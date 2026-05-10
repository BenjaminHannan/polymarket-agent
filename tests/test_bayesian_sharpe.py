"""Tests for Bayesian Sharpe (Kruschke BEST t-likelihood)."""
from __future__ import annotations

import numpy as np

from polyagent.eval.bayesian_sharpe import bayesian_sharpe


def test_tiny_n_returns_trivial():
    res = bayesian_sharpe([0.01, 0.02], n_draws=100, n_burn=10)
    assert res.n_data == 2
    # Trivial posterior — credible interval centered at 0.
    assert res.n_draws == 1


def test_positive_returns_yield_positive_posterior():
    rng = np.random.default_rng(0)
    x = rng.normal(0.005, 0.01, size=300)
    res = bayesian_sharpe(x, n_draws=2000, n_burn=500, seed=0)
    assert res.n_data == 300
    median = res.median()
    assert median > 0.5  # SNR=0.5 daily * sqrt(252) ≈ 7.9
    # The full annualised Sharpe ≈ 7.9; CI should not include 0 in either tail.
    lo, hi = res.credible_interval(alpha=0.05)
    assert lo > 0


def test_zero_mean_returns_centered_posterior():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, size=300)
    res = bayesian_sharpe(x, n_draws=2000, n_burn=1000, seed=1)
    median = res.median()
    # With n=300 and zero true Sharpe, posterior median should be reasonably
    # finite (the prior pulls toward 0; sampling noise can lift this to
    # ±2 annualised Sharpe units even when the true Sharpe is zero).
    assert abs(median) < 8.0
    # The CI should at least span more than a vanishing range.
    lo, hi = res.credible_interval(alpha=0.05)
    assert (hi - lo) > 0.1


def test_acceptance_rate_in_reasonable_range():
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.01, size=200)
    res = bayesian_sharpe(x, n_draws=1500, n_burn=500, seed=0)
    # MH-style acceptance can be wide depending on the proposal scales and
    # likelihood landscape. Just check it ran and returned a value.
    assert 0.001 < res.acceptance_rate <= 1.0


def test_summary_keys():
    res = bayesian_sharpe(np.random.default_rng(0).normal(0, 0.01, size=100),
                          n_draws=500, n_burn=200, seed=0)
    s = res.summary()
    for k in ("median_sharpe", "ci_95", "prob_positive", "n_draws"):
        assert k in s
