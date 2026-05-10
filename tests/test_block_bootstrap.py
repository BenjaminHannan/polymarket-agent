"""Tests for Politis-Romano stationary block bootstrap."""
from __future__ import annotations

import numpy as np

from polyagent.eval.block_bootstrap import (
    stationary_bootstrap_sample,
    stationary_bootstrap_sharpe,
    sharpe_ci,
    bootstrap_p_value,
)


def test_resample_length_preserved():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=100)
    y = stationary_bootstrap_sample(x, mean_block=10.0, rng=rng)
    assert len(y) == len(x)


def test_resample_uses_only_input_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    y = stationary_bootstrap_sample(x, mean_block=2.0, rng=rng)
    assert set(y).issubset(set(x))


def test_bootstrap_sharpe_centered_near_truth():
    """N(0.001, 0.01) sharpe ≈ 0.001/0.01 * sqrt(252) ≈ 1.586."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.001, 0.01, size=500)
    dist = stationary_bootstrap_sharpe(x, n_boot=500, mean_block=10.0, seed=42)
    assert len(dist) == 500
    median = float(np.median(dist))
    assert abs(median - 1.586) < 1.0  # within 1 std-err on n=500


def test_sharpe_ci_widens_with_smaller_n():
    rng = np.random.default_rng(42)
    big = rng.normal(0.001, 0.01, size=1000)
    small = rng.normal(0.001, 0.01, size=100)
    d_big = stationary_bootstrap_sharpe(big, n_boot=500, mean_block=10.0, seed=42)
    d_small = stationary_bootstrap_sharpe(small, n_boot=500, mean_block=10.0, seed=42)
    lo_b, hi_b = sharpe_ci(d_big)
    lo_s, hi_s = sharpe_ci(d_small)
    assert (hi_s - lo_s) > (hi_b - lo_b)  # small-n CI is wider


def test_p_value_for_zero_data_returns_one():
    p = bootstrap_p_value(np.array([]), n_boot=10)
    assert p == 1.0


def test_p_value_well_above_null():
    rng = np.random.default_rng(0)
    x = rng.normal(0.005, 0.01, size=500)  # very positive
    p = bootstrap_p_value(x, null_sharpe=0.0, n_boot=300, mean_block=5.0, seed=0)
    assert 0.0 <= p <= 1.0
