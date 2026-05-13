"""Tests for Generalized Venn-Abers calibration."""
from __future__ import annotations

import random

from polyagent.models.generalized_venn_abers import (
    fit_generalized_venn_abers, calibrate, quantile, interval, sample,
    _gauss_quantile,
)


def test_fit_records_data():
    state = fit_generalized_venn_abers([0.1, 0.5, 0.9], [0, 0, 1])
    assert state.n == 3
    assert len(state.sorted_pairs) == 3


def test_fit_unequal_raises():
    try:
        fit_generalized_venn_abers([0.1, 0.5], [0, 1, 0])
    except ValueError as e:
        assert "equal length" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_calibrate_returns_interval():
    # Wins concentrate above 0.7
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    state = fit_generalized_venn_abers(scores, labels)
    dist = calibrate(state, 0.8)
    assert 0.0 <= dist.p_low <= dist.point <= dist.p_high <= 1.0
    assert dist.n_support == 8


def test_calibrate_at_extremes():
    scores = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    state = fit_generalized_venn_abers(scores, labels)
    dist_low = calibrate(state, 0.05)
    dist_high = calibrate(state, 0.95)
    # High-score query should have a higher midpoint than low-score query.
    assert dist_high.point > dist_low.point


def test_quantile_in_unit():
    state = fit_generalized_venn_abers([0.1, 0.5, 0.9], [0, 0, 1])
    dist = calibrate(state, 0.5)
    q05 = quantile(dist, 0.05)
    q95 = quantile(dist, 0.95)
    assert 0.0 <= q05 <= 1.0
    assert 0.0 <= q95 <= 1.0
    assert q05 <= q95


def test_interval_monotone_in_alpha():
    state = fit_generalized_venn_abers([0.1, 0.5, 0.9], [0, 0, 1])
    dist = calibrate(state, 0.5)
    lo_50, hi_50 = interval(dist, alpha=0.50)
    lo_10, hi_10 = interval(dist, alpha=0.10)
    # 90% CI should be wider than 50% CI
    assert (hi_10 - lo_10) >= (hi_50 - lo_50) - 1e-9


def test_sample_returns_n_in_unit():
    state = fit_generalized_venn_abers([0.1, 0.9], [0, 1])
    dist = calibrate(state, 0.5)
    samples = sample(dist, n=20, rng=random.Random(0))
    assert len(samples) == 20
    assert all(0.0 <= s <= 1.0 for s in samples)


def test_gauss_quantile_extremes():
    # Standard normal quantile is unbounded but we cap.
    assert _gauss_quantile(0.0) <= -1e8
    assert _gauss_quantile(1.0) >= 1e8
    # Median ≈ 0
    assert abs(_gauss_quantile(0.5)) < 1e-6
