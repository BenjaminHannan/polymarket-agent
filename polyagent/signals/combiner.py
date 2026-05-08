"""Logarithmic opinion-pool combiner (v2 §7.2).

Given K experts with calibrated probabilities p_k and weights w_k (sum to 1):

    logit(p*) = Σ_k w_k · logit(p_k)
    p*        = σ(logit(p*))

Beats linear pool empirically because it preserves Bayesian-update semantics
when experts share a prior. Use for combining model-derived probabilities
into a single tradable P(YES) before comparing to the market price.

`fit_weights` learns w_k by minimizing log-loss over labeled rows on a
simplex. Falls back to uniform weights if scipy isn't available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def log_pool(probs: list[float], weights: list[float] | None = None) -> float:
    if not probs:
        return 0.5
    if weights is None:
        weights = [1.0 / len(probs)] * len(probs)
    if len(weights) != len(probs):
        raise ValueError("len mismatch")
    s = sum(weights)
    if s <= 0:
        weights = [1.0 / len(probs)] * len(probs)
    else:
        weights = [w / s for w in weights]
    z = sum(w * _logit(p) for w, p in zip(weights, probs))
    return _sigmoid(z)


@dataclass
class LogPoolCombiner:
    weights: list[float]
    expert_names: list[str]

    def combine(self, probs: dict[str, float]) -> float:
        ordered = [probs.get(n, 0.5) for n in self.expert_names]
        return log_pool(ordered, self.weights)

    @classmethod
    def uniform(cls, expert_names: list[str]) -> "LogPoolCombiner":
        n = len(expert_names)
        return cls(weights=[1.0 / n] * n, expert_names=expert_names)


def fit_weights(
    expert_probs: np.ndarray,  # shape (N, K)
    labels: np.ndarray,  # shape (N,) in {0, 1}
    expert_names: list[str],
    seed: int = 42,
) -> LogPoolCombiner:
    """Find non-negative weights summing to 1 that minimize log-loss."""
    try:
        from scipy.optimize import minimize, Bounds, LinearConstraint
    except ImportError:
        log.warning("scipy_missing_uniform_weights")
        return LogPoolCombiner.uniform(expert_names)

    N, K = expert_probs.shape
    if N < 50:
        log.warning("fit_weights_few_samples", n=N)
        return LogPoolCombiner.uniform(expert_names)

    eps = 1e-6
    clipped = np.clip(expert_probs, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))  # (N, K)

    def neg_log_loss(w):
        z = logits @ w
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    rng = np.random.default_rng(seed)
    x0 = rng.dirichlet(np.ones(K))
    res = minimize(
        neg_log_loss,
        x0,
        method="SLSQP",
        bounds=Bounds(np.zeros(K), np.ones(K)),
        constraints=LinearConstraint(np.ones(K), 1.0, 1.0),
        options={"maxiter": 200, "ftol": 1e-8},
    )
    w = res.x.tolist() if res.success else [1.0 / K] * K
    return LogPoolCombiner(weights=w, expert_names=expert_names)
