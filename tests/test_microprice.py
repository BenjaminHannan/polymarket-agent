"""Tests for micro-price / VAMP / queue-imbalance features."""
from __future__ import annotations

from dataclasses import dataclass, field

from polyagent.models.microprice import (
    micro_price, queue_imbalance, vamp, compute_features,
)


@dataclass
class _Book:
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)


def test_micro_price_symmetric():
    """Symmetric volume ⇒ micro-price = mid."""
    b = _Book(bids={0.48: 100}, asks={0.52: 100})
    mp = micro_price(b)
    assert mp is not None
    assert abs(mp - 0.50) < 1e-9


def test_micro_price_inverse_weighted():
    """Heavy bid ⇒ micro pulled toward ask. Inverse weighting:
    (ask_vol * bid + bid_vol * ask) / (ask_vol + bid_vol).
    """
    b = _Book(bids={0.48: 200}, asks={0.52: 100})
    mp = micro_price(b)
    # (100*0.48 + 200*0.52) / 300 = (48 + 104)/300 = 152/300 ≈ 0.5067
    assert mp is not None
    assert abs(mp - 152.0 / 300.0) < 1e-6
    assert mp > 0.50  # pulled toward ask, away from the heavy bid


def test_micro_price_empty_side_none():
    b = _Book(bids={}, asks={0.52: 100})
    assert micro_price(b) is None


def test_queue_imbalance_in_unit():
    b = _Book(bids={0.48: 100}, asks={0.52: 100})
    qi = queue_imbalance(b)
    assert qi == 0.5
    b2 = _Book(bids={0.48: 300}, asks={0.52: 100})
    qi2 = queue_imbalance(b2)
    assert qi2 == 0.75  # 300/(300+100)


def test_vamp_walks_book():
    b = _Book(bids={}, asks={0.50: 100, 0.51: 200, 0.52: 300})
    # Walking 100 USDC at 0.50: full fill at 0.50.
    v = vamp(b, "BUY", target_notional=50.0)
    assert v == 0.50
    # Walking $150 notional: 100 at 0.50 ($50) + 100 at 0.51 ($51); $50+$51 = $101.
    # Actually: $50 of asks * $0.50 = 100 shares costing $50.
    # Remaining $100; at $0.51 buys 100/0.51 = 196.08 shares.
    # Average price = ($50 + $100) / (100 + 196.08) = $150/296.08 = $0.5066.
    v2 = vamp(b, "BUY", target_notional=150.0)
    assert v2 is not None
    assert 0.50 < v2 < 0.51


def test_vamp_unknown_side_none():
    b = _Book(bids={0.48: 100}, asks={0.52: 100})
    assert vamp(b, "FLOAT", target_notional=10.0) is None


def test_compute_features_full():
    b = _Book(bids={0.48: 100, 0.47: 200}, asks={0.52: 100, 0.53: 150})
    f = compute_features(b, vamp_notional=20.0)
    assert f.mid is not None and abs(f.mid - 0.50) < 1e-9
    assert f.spread is not None and abs(f.spread - 0.04) < 1e-9
    assert f.bid_levels == 2
    assert f.ask_levels == 2
    assert f.queue_imbalance is not None
    assert f.micro is not None
    assert f.vamp_buy is not None
    assert f.vamp_sell is not None


def test_compute_features_empty_book():
    b = _Book()
    f = compute_features(b)
    assert f.mid is None
    assert f.micro is None
    assert f.queue_imbalance is None
    assert f.bid_levels == 0
    assert f.ask_levels == 0
