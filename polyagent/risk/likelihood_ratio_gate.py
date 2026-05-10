"""Likelihood-ratio selective abstention (Heng & Soh, ICLR 2025).

Direct implementation of the doc's Problem-8 fix #1: "Know When to
Abstain: Optimal Selective Classification with Likelihood Ratios"
(arXiv 2505.15008). Proves the Neyman-Pearson optimal selection score
under covariate shift is a **likelihood ratio**, not a confidence
threshold. The paper provides two practical scores:

  1. **RLog** — ratio of accepted-population log-likelihood to
     full-population log-likelihood at the input.
  2. **Δ-KNN-RLog** — local density-aware variant; replace the global
     log-likelihoods with k-NN-window kernel estimates.

We implement the lighter **RLog** scoring since Polyagent's signal
volume (~tens of thousands of `signal_outcomes` rows) is too small for
stable density estimation, and we want the gate to be fast at decision
time. Δ-KNN-RLog is left as a TODO for when we get to the ~hundreds-of-
thousands scale.

Why this matters for Polyagent
------------------------------
The existing `SelectiveGate` is a width-based confidence threshold
(width below the rolling quantile ⇒ admit). That is the El-Yaniv-Wiener
2010 / Geifman-El-Yaniv 2017 recipe. Heng-Soh's contribution is to show
that **under covariate shift** (which Polyagent absolutely has — train
data is resolved markets, live decisions are on different question
mixes, calendar regimes, etc.) the optimal selection score is a
likelihood ratio. The confidence-threshold recipe is a *special case*
that only matches the LR when train and live distributions are
identical.

Concretely:
  RLog(x) = log p_accepted(x) - log p_full(x)

where:
  - p_full is the joint density of *all* (signal, market, feature)
    rows we've seen.
  - p_accepted is the joint density of rows where the model was
    *correct* (or, equivalently, profit-positive) on holdout.

A signal is admitted iff RLog(x) is in the top-k% — i.e. the
input looks more like the historical "we got it right" distribution
than the historical "we saw it at all" distribution.

We approximate p_full and p_accepted with **Gaussian-mixture density**
on the feature vector (4-d default: model_p, market_p, |gap|,
liquidity_proxy). This is the cheapest density approximation that
still respects the LR formulation. For a more correct implementation
swap in scikit-learn's `BayesianGaussianMixture` or `KernelDensity`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger()


def _safe_logdet(cov: np.ndarray) -> float:
    """log|Σ| with a floor for numerical stability."""
    sign, ld = np.linalg.slogdet(cov)
    if sign <= 0:
        # singular or near-singular — fall back to diag
        return float(np.sum(np.log(np.clip(np.diag(cov), 1e-9, None))))
    return float(ld)


@dataclass
class _GaussianFit:
    mean: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    log_norm: float        # constant in log N(x | μ, Σ)
    n_samples: int

    @classmethod
    def from_samples(cls, X: np.ndarray) -> "_GaussianFit":
        """Maximum-likelihood Gaussian fit with diagonal regularization
        for stability under low n."""
        n, d = X.shape
        mean = X.mean(axis=0)
        cov = np.cov(X.T) if n > 1 else np.eye(d)
        # Regularize: add 1e-4 to diagonal to avoid singularity when
        # features are highly correlated (e.g. model_p and combined_p).
        cov = cov + 1e-4 * np.eye(d)
        inv = np.linalg.pinv(cov)
        log_norm = -0.5 * (d * math.log(2.0 * math.pi) + _safe_logdet(cov))
        return cls(mean=mean, cov=cov, inv_cov=inv, log_norm=log_norm, n_samples=n)

    def log_prob(self, x: np.ndarray) -> float:
        diff = x - self.mean
        m = float(diff @ self.inv_cov @ diff)
        return self.log_norm - 0.5 * m


@dataclass
class LikelihoodRatioGate:
    """Heng-Soh RLog selective abstention.

    Args:
        feature_dim: dimensionality of the feature vector passed to
            `add_observation` / `score`. Default 4 = (model_p,
            market_p, abs_gap, liquidity_proxy). Choose features that
            shift between train and live regimes.
        coverage: fraction of incoming signals to ADMIT. e.g. 0.40
            means we keep the top 40% by RLog score.
        burn_in: number of *both-accepted-and-rejected* observations
            required before the gate trips. Until then, admit all.
        refit_every: refit the two Gaussian densities every N
            new observations (cheap; full rebuild from the deque).
        accept_buffer: max #(features, was_correct) pairs to keep.
        score_quantile_window: max #recent RLog scores to keep for
            the rolling-quantile threshold.
    """
    feature_dim: int = 4
    coverage: float = 0.40
    burn_in: int = 200
    refit_every: int = 50
    accept_buffer: int = 5000
    score_quantile_window: int = 2000

    _samples: deque = field(default_factory=lambda: deque(maxlen=5000))
    _is_correct: deque = field(default_factory=lambda: deque(maxlen=5000))
    _scores: deque = field(default_factory=lambda: deque(maxlen=2000))

    _fit_full: _GaussianFit | None = None
    _fit_accept: _GaussianFit | None = None
    _since_last_fit: int = 0

    n_seen: int = 0
    n_admitted: int = 0

    def __post_init__(self) -> None:
        if self._samples.maxlen != self.accept_buffer:
            self._samples = deque(maxlen=self.accept_buffer)
            self._is_correct = deque(maxlen=self.accept_buffer)
        if self._scores.maxlen != self.score_quantile_window:
            self._scores = deque(maxlen=self.score_quantile_window)

    # ── ingest historical outcomes ─────────────────────────────────────
    def add_observation(self, features, was_correct: bool) -> None:
        """Record one resolved holdout row. `was_correct` is whether
        the prediction (signal direction or profit sign) was right."""
        x = np.asarray(features, dtype=float).flatten()
        if x.size != self.feature_dim:
            log.warning(
                "lr_gate_dim_mismatch",
                got=int(x.size),
                expected=self.feature_dim,
            )
            return
        if not np.all(np.isfinite(x)):
            return
        self._samples.append(x)
        self._is_correct.append(bool(was_correct))
        self._since_last_fit += 1
        if self._since_last_fit >= self.refit_every:
            self._refit()

    def _refit(self) -> None:
        """Re-fit p_full and p_accepted from current buffers."""
        self._since_last_fit = 0
        if len(self._samples) < max(self.burn_in, self.feature_dim + 2):
            return
        X = np.array(self._samples)
        correct_mask = np.array(self._is_correct, dtype=bool)
        if correct_mask.sum() < self.feature_dim + 2:
            return
        try:
            self._fit_full = _GaussianFit.from_samples(X)
            self._fit_accept = _GaussianFit.from_samples(X[correct_mask])
        except Exception as e:
            log.warning("lr_gate_refit_failed", err=str(e))

    # ── score & gate ────────────────────────────────────────────────────
    def rlog(self, features) -> float | None:
        """Compute the RLog score at `features`. Returns None during
        burn-in or if features are malformed."""
        if self._fit_full is None or self._fit_accept is None:
            return None
        x = np.asarray(features, dtype=float).flatten()
        if x.size != self.feature_dim or not np.all(np.isfinite(x)):
            return None
        lp_accept = self._fit_accept.log_prob(x)
        lp_full = self._fit_full.log_prob(x)
        score = lp_accept - lp_full
        self._scores.append(score)
        return float(score)

    def admit(self, features) -> bool:
        """Decide whether to admit a candidate based on RLog."""
        self.n_seen += 1
        if self._fit_full is None or self._fit_accept is None or len(self._scores) < self.burn_in:
            self.n_admitted += 1
            return True
        score = self.rlog(features)
        if score is None:
            self.n_admitted += 1
            return True
        threshold = float(np.quantile(self._scores, 1.0 - self.coverage))
        admit = score >= threshold
        if admit:
            self.n_admitted += 1
        return admit

    def summary(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "n_admitted": self.n_admitted,
            "global_admit_rate": round(self.n_admitted / max(1, self.n_seen), 3),
            "coverage_target": self.coverage,
            "n_samples": len(self._samples),
            "n_correct": int(sum(self._is_correct)),
            "fit_ready": self._fit_full is not None and self._fit_accept is not None,
        }
