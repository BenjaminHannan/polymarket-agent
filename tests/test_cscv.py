"""Tests for CSCV/PBO."""
from __future__ import annotations

import numpy as np

from polyagent.eval.cscv import (
    pbo_from_returns, deflated_sharpe, _gauss_quantile,
)


def test_pbo_balanced_configs_near_half():
    """Configurations with iid Gaussian returns should give PBO roughly
    in [0.2, 0.8] — the in-sample winner is at most marginally better
    than OOS median by chance."""
    rng = np.random.default_rng(42)
    T, C = 200, 5
    R = rng.normal(0.0, 0.01, size=(T, C))
    res = pbo_from_returns(R, n_splits=8)
    # iid noise + finite n: PBO is wide; we just sanity-check it's
    # not pinned to 0 or 1 (which would indicate a bug).
    assert 0.0 < res.pbo < 1.0
    assert res.n_configs == 5
    assert res.n_periods == 200
    assert res.n_splits == 8


def test_pbo_dominant_config_low():
    """One configuration is genuinely better — PBO should be near 0."""
    rng = np.random.default_rng(0)
    T, C = 200, 4
    R = rng.normal(0.0, 0.01, size=(T, C))
    # Make column 0 dominate by 3 sd.
    R[:, 0] += 0.005
    res = pbo_from_returns(R, n_splits=8)
    assert res.pbo < 0.3


def test_pbo_rejects_one_config():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, size=(100, 1))
    try:
        pbo_from_returns(R, n_splits=4)
    except ValueError as e:
        assert "configurations" in str(e)
    else:
        raise AssertionError("expected ValueError on C=1")


def test_pbo_rejects_odd_splits():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, size=(50, 3))
    try:
        pbo_from_returns(R, n_splits=7)
    except ValueError as e:
        assert "even" in str(e)
    else:
        raise AssertionError("expected ValueError on odd n_splits")


def test_deflated_sharpe_n_trials_penalty():
    """Multi-trial penalty drives DSR below the raw Sharpe percentile."""
    dsr_one = deflated_sharpe(2.0, n_trials=1, n_observations=252)
    dsr_many = deflated_sharpe(2.0, n_trials=100, n_observations=252)
    assert dsr_one > dsr_many


def test_gauss_quantile_monotone():
    a = _gauss_quantile(0.1)
    b = _gauss_quantile(0.5)
    c = _gauss_quantile(0.9)
    assert a < b < c
    assert abs(b) < 1e-6  # median of N(0, 1) is 0
