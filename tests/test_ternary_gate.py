"""Tests for ternary UP/FLAT/DOWN selective classifier."""
from __future__ import annotations

import numpy as np

from polyagent.signals.ternary_gate import TernaryGate


def test_burn_in_returns_flat():
    g = TernaryGate()
    assert g.classify(0.9, 0.1) == "FLAT"


def test_fit_finds_threshold_on_clear_signal():
    rng = np.random.default_rng(0)
    n = 500
    # Markets resolve YES when our prediction > market price.
    market = rng.uniform(0.2, 0.8, size=n)
    pred = market + rng.normal(0.0, 0.05, size=n)
    pred = np.clip(pred, 0.01, 0.99)
    # Add a clear UP signal where pred is much higher than market
    up_idx = rng.choice(n, size=80, replace=False)
    pred[up_idx] = np.clip(market[up_idx] + 0.15, 0.01, 0.99)
    outcomes = np.array([
        1 if (p > m + 0.1 and rng.random() < 0.85) else
        0 if (p < m - 0.1 and rng.random() < 0.85) else
        int(rng.random() < m)
        for p, m in zip(pred, market)
    ])
    g = TernaryGate(min_hit_rate=0.60, min_edge_abs=0.05, coverage_floor=0.05)
    g.fit(pred, market, outcomes)
    # Should have found at least one threshold
    assert g._up_threshold is not None or g._down_threshold is not None


def test_classify_in_three_buckets():
    g = TernaryGate()
    g._up_threshold = 0.1
    g._down_threshold = 0.1
    assert g.classify(0.7, 0.5) == "UP"
    assert g.classify(0.3, 0.5) == "DOWN"
    assert g.classify(0.55, 0.5) == "FLAT"


def test_admits_only_matching_side():
    g = TernaryGate()
    g._up_threshold = 0.1
    g._down_threshold = 0.1
    # UP candidate, BUY side: admit
    assert g.admits(0.7, 0.5, "BUY") is True
    # UP candidate, SELL side: reject (direction mismatch)
    assert g.admits(0.7, 0.5, "SELL") is False
    # FLAT candidate: reject either side
    assert g.admits(0.55, 0.5, "BUY") is False


def test_summary_keys():
    g = TernaryGate()
    s = g.summary()
    for k in ("up_threshold", "down_threshold", "min_hit_rate"):
        assert k in s
