"""Tests for hierarchical Beta-binomial calibration."""
from __future__ import annotations

import sqlite3

from polyagent.models.hierarchical_calibrator import (
    fit_and_persist,
    lookup_posterior,
    _beta_marginal_log_lik,
    _beta_quantile,
)


def test_marginal_lik_handles_invalid():
    """Negative/zero hyperparameters should return −inf."""
    assert _beta_marginal_log_lik([(10, 5)], -0.1, 10) == float("-inf")
    assert _beta_marginal_log_lik([(10, 5)], 0.5, -1) == float("-inf")


def test_beta_quantile_monotone():
    """For Beta(2, 5) the median should be below the 75th percentile."""
    q25 = _beta_quantile(2, 5, 0.25)
    q50 = _beta_quantile(2, 5, 0.50)
    q75 = _beta_quantile(2, 5, 0.75)
    assert 0.0 < q25 < q50 < q75 < 1.0


def test_pooling_shrinks_small_cells():
    """A cell with n=1 should be shrunk near the global mean μ, not its raw rate."""
    # Global: 4 cells with ~0.50 win rate (100 obs each).
    # 1 cell with n=1, k=1 — raw rate 1.0.
    obs = {
        "a": (100, 50),
        "b": (100, 49),
        "c": (100, 51),
        "d": (100, 50),
        "small": (1, 1),
    }
    posts, mu, nu = fit_and_persist(obs, conn=None)
    p_small = posts["small"].posterior_mean
    # Should be pulled toward μ (~0.5), well below the raw rate of 1.0.
    assert p_small < 0.95
    assert p_small > 0.3
    # Big cells should be near their raw rate.
    assert abs(posts["a"].posterior_mean - 0.50) < 0.05


def test_pooling_respects_large_cells():
    """A cell with n=10000 should dominate even an extreme global prior."""
    obs = {
        "a": (10, 5),
        "b": (10, 5),
        "c": (10, 5),
        "big": (10_000, 8_000),  # raw 0.80
    }
    posts, _, _ = fit_and_persist(obs, conn=None)
    # The big cell's posterior should be close to its raw rate even
    # though three small cells anchor μ around 0.5.
    assert abs(posts["big"].posterior_mean - 0.80) < 0.02


def test_persist_and_lookup(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    obs = {"sports_global_24h": (100, 60), "politics_24h": (50, 25)}
    posts, mu, nu = fit_and_persist(obs, conn=conn)
    look = lookup_posterior(conn, "sports_global_24h")
    assert look is not None
    assert look.n_obs == 100
    assert look.n_wins == 60
    assert 0.5 < look.posterior_mean < 0.65


def test_empty_observations_returns_default():
    posts, mu, nu = fit_and_persist({}, conn=None)
    assert posts == {}
    assert mu == 0.5
    assert nu == 10.0
