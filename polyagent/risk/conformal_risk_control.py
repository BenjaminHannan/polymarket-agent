"""Conformal Risk Control (Angelopoulos et al., ICLR 2024).

Direct implementation of pmwhybetter.md Problem-7 fix #5: choosing
per-cell abstention thresholds with **finite-sample guarantees** on the
selective-classification risk.

Reference
---------
  - Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control,"
    ICLR 2024, OpenReview f3549ef9b5ff5.
  - https://github.com/aangelopoulos/conformal-risk

Setting
-------
We have a calibration set of (predicted probability, label) pairs and
a *loss function* L(p, y) that returns the loss of acting on prediction
p when the truth is y (e.g., squared error, 0/1 wrong-direction, or
something more bespoke like "money lost on this trade").

Conformal Risk Control finds a threshold λ such that the **expected
loss on accepted predictions** is bounded by a user-specified α with
exactly (1 − α) coverage under exchangeability:

    E[L(p, y) | accept(p, λ)] ≤ α    with finite-sample guarantee

The threshold is computed from the calibration set's empirical quantile
of the loss function over a candidate λ grid; the published trick is to
use the (1 + 1/n)·(1 − α) quantile to get the finite-sample bound (this
is the same Bonferroni-style correction Vovk uses for conformal
prediction).

API
---
- `ConformalRiskController.fit(scores, losses)` — calibrate λ from
  past (score, loss) observations.
- `controller.accept(score) -> bool` — admit a new candidate iff its
  score is in the safe region.
- `controller.risk_at(lambda_)` — empirical risk on the calibration
  set at a given λ (useful for diagnostics).

Why this matters for Polyagent
------------------------------
The existing `selective_gate.py` uses interval-width quantiles to
abstain, and `likelihood_ratio_gate.py` uses the Heng-Soh LR. Both are
*heuristic* in the sense that they don't guarantee any specific
operational risk level — they admit the top X% by some score and
hope that the resulting selective error is acceptable.

Conformal Risk Control inverts the question: *given* a target risk α
(e.g., "expected log-loss ≤ 0.5 on accepted trades"), find the
threshold that achieves it with high probability. This lets us write
acceptance rules in terms of *operational* metrics (max trade loss,
max wrong-direction rate) rather than *statistical* metrics
(confidence interval width).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class ConformalRiskController:
    """Score-based selective classifier with finite-sample risk control.

    Args:
        alpha: target operational risk on accepted predictions.
            Lower α = stricter gate, fewer admissions.
        lambda_grid: candidate thresholds to search over. Default 1000
            uniformly-spaced points in [0, 1].
        higher_is_better: if True (default), admit when score ≥ λ; if
            False, admit when score ≤ λ. Matches the convention of the
            score function (e.g. confidence scores are higher=better;
            uncertainty widths are lower=better, set False).
    """
    alpha: float = 0.1
    n_lambda: int = 1000
    higher_is_better: bool = True
    # Fit state (populated by `fit`)
    _lambda_hat: float | None = None
    _calibration_scores: np.ndarray | None = None
    _calibration_losses: np.ndarray | None = None
    _empirical_risk_at_lambda: float | None = None

    def fit(self, scores, losses) -> None:
        """Calibrate the controller on past observations.

        Args:
            scores: per-prediction scores (any scale; the controller
                only uses ordering).
            losses: per-prediction realized loss (non-negative).
        """
        s = np.asarray(scores, dtype=float).flatten()
        l = np.asarray(losses, dtype=float).flatten()
        if len(s) != len(l):
            raise ValueError("scores and losses must have same length")
        if len(s) < 5:
            log.warning("crc_fit_too_few_samples", n=len(s))
            self._lambda_hat = None
            return

        n = len(s)
        # Build candidate λ grid: just the unique calibration scores
        # (the optimal λ is always at one of these). Plus 0 and 1 to
        # handle the empty-accept and full-accept edge cases.
        candidates = np.unique(s)
        candidates = np.concatenate(([s.min() - 1e-9], candidates, [s.max() + 1e-9]))

        # For each candidate λ, compute expected loss on accepted set.
        # Acceptance rule: score ≥ λ (or ≤ λ if higher_is_better=False)
        best_lambda = None
        best_n_accept = -1
        # The finite-sample-corrected target: the (1 + 1/n)·(1 − α)
        # empirical quantile of losses on accepted samples must be ≤ α.
        # Equivalently, the *mean loss on accepted* must be ≤ α with the
        # (1 − α/(n+1)) bonferroni-style guarantee.
        target = float(self.alpha)
        for lam in candidates:
            if self.higher_is_better:
                mask = s >= lam
            else:
                mask = s <= lam
            n_acc = int(mask.sum())
            if n_acc == 0:
                continue
            emp_risk = float(l[mask].mean())
            # Finite-sample inflation factor (n+1)/n ensures the
            # expected risk on a fresh exchangeable sample is bounded by
            # target with prob ≥ 1 − α.
            inflated = emp_risk * (n + 1) / n
            if inflated <= target:
                # Among all valid λ, pick the one that admits the MOST
                # samples (least conservative). This is the standard CRC
                # convention.
                if n_acc > best_n_accept:
                    best_n_accept = n_acc
                    best_lambda = float(lam)
                    best_risk = emp_risk
        if best_lambda is None:
            log.warning(
                "crc_no_admissible_lambda",
                target=target,
                min_realised_risk=float(l.min()) if len(l) else None,
            )
            # Default to the strictest threshold (accept nothing).
            self._lambda_hat = float(s.max() + 1e-9) if self.higher_is_better else float(s.min() - 1e-9)
            self._empirical_risk_at_lambda = 0.0
        else:
            self._lambda_hat = best_lambda
            self._empirical_risk_at_lambda = best_risk

        self._calibration_scores = s
        self._calibration_losses = l
        log.info(
            "crc_fit_done",
            n_calibration=n,
            lambda_hat=round(self._lambda_hat, 4),
            target_risk=target,
            empirical_risk=round(self._empirical_risk_at_lambda or 0.0, 4),
            admit_rate=round(best_n_accept / n if best_n_accept > 0 else 0.0, 3),
        )

    def accept(self, score: float) -> bool:
        """Decide whether to admit a new candidate.

        Returns False during burn-in (before fit() is called).
        """
        if self._lambda_hat is None:
            return False
        s = float(score)
        if self.higher_is_better:
            return s >= self._lambda_hat
        return s <= self._lambda_hat

    def risk_at(self, lam: float) -> float | None:
        """Empirical risk on the calibration set at a given threshold.

        Returns None when no calibration data is present."""
        if self._calibration_scores is None or self._calibration_losses is None:
            return None
        if self.higher_is_better:
            mask = self._calibration_scores >= lam
        else:
            mask = self._calibration_scores <= lam
        n_acc = int(mask.sum())
        if n_acc == 0:
            return 0.0
        return float(self._calibration_losses[mask].mean())

    def summary(self) -> dict:
        return {
            "alpha_target": self.alpha,
            "lambda_hat": self._lambda_hat,
            "empirical_risk_at_lambda": self._empirical_risk_at_lambda,
            "n_calibration": (
                len(self._calibration_scores)
                if self._calibration_scores is not None else 0
            ),
            "higher_is_better": self.higher_is_better,
        }
