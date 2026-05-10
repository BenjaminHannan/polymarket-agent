"""Tests for VPIN toxicity gate."""
from __future__ import annotations

import time

from polyagent.risk.vpin_gate import VPINGate


def _flood_buys(gate: VPINGate, token: str, n: int, size: float = 100.0):
    for _ in range(n):
        gate.record_trade(token, "BUY", size)


def _flood_sells(gate: VPINGate, token: str, n: int, size: float = 100.0):
    for _ in range(n):
        gate.record_trade(token, "SELL", size)


def test_cold_start_allows_quote():
    g = VPINGate(bucket_volume=500, n_buckets=5, min_buckets=3)
    # No trades recorded — gate should not trip.
    allow, info = g.allow_quote("tok", "BUY")
    assert allow is True
    assert info["decision"] == "allow"
    assert info["reason"] == "cold_start"


def test_balanced_flow_allows():
    g = VPINGate(bucket_volume=500, n_buckets=10, min_buckets=3, vpin_max=0.6)
    for _ in range(20):
        g.record_trade("tok", "BUY", 50)
        g.record_trade("tok", "SELL", 50)
    allow, info = g.allow_quote("tok", "BUY")
    assert allow is True
    assert info["vpin"] is None or info["vpin"] < 0.6


def test_one_sided_buy_flow_blocks_sell_quote():
    g = VPINGate(bucket_volume=500, n_buckets=10, min_buckets=3, vpin_max=0.5,
                 direction_quality=1.0)
    # Hammer one side hard.
    _flood_buys(g, "tok", 200, size=100)
    v = g.vpin("tok")
    assert v is not None and v >= 0.5, f"expected high VPIN got {v}"
    # SELL quote — flow is buy-heavy, against our offer ⇒ block.
    allow, info = g.allow_quote("tok", "SELL")
    assert allow is False
    assert info["decision"] == "block"


def test_one_sided_buy_flow_allows_buy_quote():
    g = VPINGate(bucket_volume=500, n_buckets=10, min_buckets=3, vpin_max=0.5,
                 direction_quality=1.0)
    _flood_buys(g, "tok", 200, size=100)
    # BUY quote — flow is buy-heavy but ALIGNED with our bid ⇒ allow.
    # (we'd love to be filled by sellers; if buyers are running we
    # are the wrong-direction maker, but the gate's purpose is to
    # block quotes that will be picked off — a BUY quote sitting
    # below mid won't get hit by buyers.)
    allow, info = g.allow_quote("tok", "BUY")
    assert allow is True
    assert info["decision"] == "allow"


def test_direction_quality_partial_correction():
    g_low = VPINGate(direction_quality=0.59)
    g_high = VPINGate(direction_quality=1.0)
    # Same trade flow into both.
    for _ in range(100):
        g_low.record_trade("tok", "BUY", 100)
        g_high.record_trade("tok", "BUY", 100)
    v_low = g_low.vpin("tok")
    v_high = g_high.vpin("tok")
    # Both should saturate near 1 since all flow is BUY.
    assert v_low is not None and v_high is not None
    # Equally extreme — quality correction shouldn't flip the sign.
    assert v_low > 0.5 and v_high > 0.5


def test_mark_direction_quality_clamps():
    g = VPINGate()
    g.mark_direction_quality(0.2)
    assert g.direction_quality == 0.5  # clamped up
    g.mark_direction_quality(1.5)
    assert g.direction_quality == 1.0  # clamped down


def test_summary_shape():
    g = VPINGate()
    s = g.summary()
    assert "n_tokens" in s
    assert "vpin_max" in s
    assert "direction_quality" in s
