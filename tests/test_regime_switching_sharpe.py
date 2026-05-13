"""Tests for Bayesian regime-switching Sharpe."""
from __future__ import annotations

import numpy as np

from polyagent.eval.regime_switching_sharpe import fit_regime_switching


def test_tiny_n_returns_trivial():
    res = fit_regime_switching([0.01, 0.02])
    assert res.n == 2
    # Trivial: both regimes equal
    assert res.mu[0] == res.mu[1]


def test_two_regime_recovery():
    """Synthetic two-regime returns should recover roughly the right μ's."""
    rng = np.random.default_rng(0)
    n = 600
    # First half N(0.005, 0.01), second half N(-0.005, 0.01) — clear regime shift.
    a = rng.normal(0.005, 0.01, size=n // 2)
    b = rng.normal(-0.005, 0.01, size=n // 2)
    x = np.concatenate([a, b])
    res = fit_regime_switching(x, n_iter=30, seed=0)
    # Regime 0 should be the low-mean regime (μ_0 < μ_1 by ordering)
    assert res.mu[0] < res.mu[1]
    # The two regimes' Sharpes should have opposite signs (one ~+8, one ~-8)
    assert res.regime_sharpes[0] < 0 < res.regime_sharpes[1]
    # And their magnitudes should both be substantial (>3)
    assert abs(res.regime_sharpes[0]) > 3.0
    assert res.regime_sharpes[1] > 3.0


def test_stationary_probs_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=300)
    res = fit_regime_switching(x, n_iter=15)
    assert abs(sum(res.stationary_prob) - 1.0) < 1e-6


def test_posterior_regime_in_unit():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=200)
    res = fit_regime_switching(x, n_iter=15)
    assert all(0 <= float(v) <= 1 for v in res.posterior_regime)


def test_summary_keys():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=100)
    res = fit_regime_switching(x, n_iter=10)
    s = res.summary()
    for k in ("mu", "sigma", "stationary_prob", "regime_sharpes",
              "mixture_sharpe", "n", "log_likelihood"):
        assert k in s
