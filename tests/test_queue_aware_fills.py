"""Tests for the queue-aware fill simulator."""
from __future__ import annotations

from polyagent.orderbook import OrderBook as Book
from polyagent.risk.queue_aware_fills import (
    compare_fill_models,
    simulate_passive_fill,
    walk_book_taker,
)


def _book(bids: dict[float, float], asks: dict[float, float]) -> Book:
    b = Book(token_id="t")
    b.bids = {float(p): float(s) for p, s in bids.items()}
    b.asks = {float(p): float(s) for p, s in asks.items()}
    return b


# ── walk_book_taker ────────────────────────────────────────────────────

def test_walk_taker_buy_full_at_top_level():
    # 100 size at 0.50 ask is enough; no slippage
    b = _book(bids={0.49: 100}, asks={0.50: 100, 0.51: 200})
    r = walk_book_taker(b, "BUY", 50)
    assert r is not None
    assert r.filled_size == 50
    assert r.vwap_price == 0.50
    assert r.top_of_book_price == 0.50
    assert r.levels_walked == 1
    assert not r.partial
    assert r.slippage_bps == 0


def test_walk_taker_buy_walks_two_levels():
    # ask top has 30 at 0.50, next 200 at 0.51; we want 100 → walks 2 levels
    b = _book(bids={0.49: 100}, asks={0.50: 30, 0.51: 200})
    r = walk_book_taker(b, "BUY", 100)
    assert r is not None
    assert r.filled_size == 100
    # vwap = (30 * 0.50 + 70 * 0.51) / 100 = 0.507
    assert abs(r.vwap_price - 0.507) < 1e-9
    assert r.top_of_book_price == 0.50
    assert r.levels_walked == 2
    # slippage = (0.507 - 0.50) / 0.50 * 10000 = 140 bps
    assert abs(r.slippage_bps - 140.0) < 1e-6


def test_walk_taker_partial_when_book_too_thin():
    # Only 50 total ask depth; we want 200
    b = _book(bids={0.49: 100}, asks={0.50: 30, 0.51: 20})
    r = walk_book_taker(b, "BUY", 200)
    assert r is not None
    assert r.filled_size == 50
    assert r.partial
    # vwap = (30*0.50 + 20*0.51) / 50 = 0.504
    assert abs(r.vwap_price - 0.504) < 1e-9


def test_walk_taker_max_price_truncates():
    # Same setup as 2-level walk but with a max_price=0.505 cap on BUY
    b = _book(bids={0.49: 100}, asks={0.50: 30, 0.51: 200})
    r = walk_book_taker(b, "BUY", 100, max_price=0.505)
    assert r is not None
    assert r.filled_size == 30  # only the 0.50 level passes the cap
    assert r.partial
    assert r.vwap_price == 0.50


def test_walk_taker_sell_walks_bids_descending():
    b = _book(bids={0.50: 30, 0.49: 200}, asks={0.51: 100})
    r = walk_book_taker(b, "SELL", 100)
    assert r is not None
    # vwap = (30 * 0.50 + 70 * 0.49) / 100 = 0.4930
    assert abs(r.vwap_price - 0.4930) < 1e-9
    assert r.top_of_book_price == 0.50
    assert r.levels_walked == 2


def test_walk_taker_returns_none_on_empty_side():
    b = _book(bids={0.49: 100}, asks={})
    assert walk_book_taker(b, "BUY", 10) is None
    b2 = _book(bids={}, asks={0.51: 100})
    assert walk_book_taker(b2, "SELL", 10) is None


# ── simulate_passive_fill ──────────────────────────────────────────────

def test_passive_fill_with_flow_rate_short_horizon():
    # Posting 50 BUY at 0.49; 100 already at 0.49 ahead of us;
    # opp-side flow at 5/sec; horizon 60s → 300 size cleared.
    # clear_time = (100 + 50) / 5 = 30s; horizon 60 ≥ clear_time → near-cert.
    b = _book(bids={0.49: 100, 0.48: 200}, asks={0.50: 100, 0.51: 200})
    r = simulate_passive_fill(
        b, "BUY", 0.49, 50, horizon_sec=60, recent_opp_volume_per_sec=5.0
    )
    assert r is not None
    assert r.fill_prob > 0.9
    assert r.expected_wait_sec == 30.0
    assert r.queue_ahead == 100


def test_passive_fill_with_flow_rate_horizon_too_short():
    # Same but horizon=10s, clear_time=30s → fill_prob ~ 10/30 = 0.33
    b = _book(bids={0.49: 100, 0.48: 200}, asks={0.50: 100, 0.51: 200})
    r = simulate_passive_fill(
        b, "BUY", 0.49, 50, horizon_sec=10, recent_opp_volume_per_sec=5.0
    )
    assert r is not None
    assert 0.20 < r.fill_prob < 0.50


def test_passive_fill_imbalance_fallback_no_flow_rate():
    # Bid-heavy book: posting on the bid side should be HARDER (less likely
    # to be hit since opp-flow is sparse).
    b = _book(bids={0.49: 1000, 0.48: 1000}, asks={0.51: 50, 0.52: 30})
    r = simulate_passive_fill(b, "BUY", 0.49, 50, horizon_sec=60)
    assert r is not None
    # Bid-heavy → bid_share high → posting BUY harder → bias negative on prob
    assert r.fill_prob < 0.5

    # Ask-heavy: posting BUY on bid side should be easier
    b2 = _book(bids={0.49: 50, 0.48: 30}, asks={0.51: 1000, 0.52: 1000})
    r2 = simulate_passive_fill(b2, "BUY", 0.49, 50, horizon_sec=60)
    assert r2 is not None
    assert r2.fill_prob > 0.5


def test_passive_fill_returns_none_on_empty_book():
    b = _book(bids={}, asks={0.51: 100})
    assert simulate_passive_fill(b, "BUY", 0.49, 50) is None


# ── compare_fill_models ────────────────────────────────────────────────

def test_compare_fill_models_returns_full_breakdown():
    b = _book(bids={0.49: 100}, asks={0.50: 30, 0.51: 200})
    out = compare_fill_models(b, "BUY", 100)
    assert out["available"] is True
    assert out["top_of_book"] == 0.50
    assert abs(out["walked_vwap"] - 0.507) < 1e-9
    assert out["filled_size"] == 100
    assert out["levels_walked"] == 2
    assert out["partial"] is False
    # Walked slippage should be > 0 (we paid more than top of book)
    assert out["slippage_bps_walked"] > 0


def test_compare_fill_models_unavailable_on_empty_book():
    b = _book(bids={}, asks={})
    assert compare_fill_models(b, "BUY", 100)["available"] is False
