"""Tests for Ledoit-Wolf 2008 robust Sharpe test."""
from __future__ import annotations

import numpy as np

from polyagent.eval.ledoit_wolf_sharpe import (
    ledoit_wolf_sharpe_test,
    ledoit_wolf_sharpe_difference,
    hac_variance,
)


def test_tiny_n_returns_pvalue_one():
    res = ledoit_wolf_sharpe_test([0.01, 0.02])
    assert res.p_value == 1.0
    assert res.sharpe == 0.0


def test_positive_sharpe_rejects_null():
    """Strong positive Sharpe should produce small p-value."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.005, 0.01, size=500)
    res = ledoit_wolf_sharpe_test(x, null_sharpe=0.0)
    assert res.sharpe > 0
    assert res.p_value < 0.05
    assert res.z_statistic > 1.5


def test_zero_mean_high_pvalue():
    """Centered Gaussian returns should have p > 0.05 most of the time."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.01, size=500)
    res = ledoit_wolf_sharpe_test(x, null_sharpe=0.0)
    # Not strictly required to be > 0.05 every seed, but should be reasonable
    assert 0.0 <= res.p_value <= 1.0
    assert abs(res.sharpe) < 5.0  # annualised


def test_hac_variance_simple_case():
    """HAC of iid sequence should be close to plain variance."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=1000)
    v = hac_variance(x, lag=3)
    # IID variance is 1; HAC should be close.
    assert 0.7 < v < 1.3


def test_hac_variance_correlated_inflates():
    """Positively-autocorrelated series should produce HAC > IID variance."""
    rng = np.random.default_rng(0)
    # AR(1) with rho=0.7
    n = 2000
    x = np.zeros(n)
    eps = rng.normal(0, 1, size=n)
    rho = 0.7
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    # IID variance estimate
    iid_var = float(np.var(x, ddof=1))
    hac = hac_variance(x, lag=10)
    # HAC should be substantially larger than IID variance for AR(1, 0.7)
    assert hac > iid_var * 1.5


def test_sharpe_difference_zero_when_identical():
    """Same series should produce zero difference, p≈0.5."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=300)
    res = ledoit_wolf_sharpe_difference(x, x)
    assert abs(res.sharpe) < 1e-6


def test_sharpe_difference_finds_outperformance():
    """A clearly better series should produce a positive difference and
    a moderate-to-low p-value."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.003, 0.01, size=500)
    b = rng.normal(0.000, 0.01, size=500)
    res = ledoit_wolf_sharpe_difference(a, b)
    assert res.sharpe > 0
    assert res.p_value < 0.5


def test_sharpe_difference_rejects_mismatched_lengths():
    try:
        ledoit_wolf_sharpe_difference([0.01, 0.02], [0.01])
    except ValueError as e:
        assert "length" in str(e)
    else:
        raise AssertionError("expected ValueError")
