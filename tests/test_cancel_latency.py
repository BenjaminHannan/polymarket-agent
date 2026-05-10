"""Tests for Brownian σ√Δt cancel-latency model."""
from __future__ import annotations

import random

from polyagent.risk.cancel_latency import CancelLatencyModel


def test_drift_scales_with_sigma():
    m = CancelLatencyModel()
    d1 = m.expected_drift_bps(sigma_per_sec=0.001)
    d2 = m.expected_drift_bps(sigma_per_sec=0.002)
    # 2× sigma should produce 2× drift (linear in σ for fixed Δt).
    assert abs(d2 / d1 - 2.0) < 1e-9


def test_expected_loss_capped_by_half_spread():
    m = CancelLatencyModel()
    # Big sigma: drift dominates spread.
    loss = m.expected_loss_bps(sigma_per_sec=0.01, spread_bps=10.0)
    assert loss == 5.0  # capped at half_spread


def test_should_repost_under_threshold():
    m = CancelLatencyModel(last_look_bps=15.0)
    assert m.should_repost(observed_price_change_bps=5.0) is True
    assert m.should_repost(observed_price_change_bps=-10.0) is True


def test_should_repost_blocks_over_threshold():
    m = CancelLatencyModel(last_look_bps=15.0)
    assert m.should_repost(observed_price_change_bps=20.0) is False
    assert m.should_repost(observed_price_change_bps=-25.0) is False


def test_disabled_always_reposts():
    m = CancelLatencyModel(last_look_bps=15.0, enabled=False)
    assert m.should_repost(observed_price_change_bps=1000.0) is True


def test_fill_probability_monotone_in_distance():
    m = CancelLatencyModel()
    p_close = m.fill_probability(sigma_per_sec=0.001, distance_to_mid_bps=10.0)
    p_far = m.fill_probability(sigma_per_sec=0.001, distance_to_mid_bps=100.0)
    assert p_close > p_far  # closer to mid ⇒ more likely to be filled


def test_simulate_cancel_returns_finite():
    m = CancelLatencyModel()
    rng = random.Random(0)
    result = m.simulate_cancel(
        sigma_per_sec=0.001,
        distance_to_mid_bps=10.0,
        n_iters=200,
        rng=rng,
    )
    assert 0.0 <= result["p_filled"] <= 1.0
    assert result["avg_loss_bps"] >= 0.0
    assert result["n_iters"] == 200
