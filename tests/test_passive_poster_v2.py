"""Tests for passive_poster_v2 — Avellaneda-Stoikov math + eligibility."""
from __future__ import annotations

from unittest.mock import MagicMock

from polyagent.gamma import Market
from polyagent.orderbook import OrderBook
from polyagent.strategies.passive_poster_v2 import (
    PassivePosterV2,
    avellaneda_stoikov_half_spread,
    avellaneda_stoikov_skew,
)


# ── Avellaneda-Stoikov math ─────────────────────────────────────────────

def test_skew_zero_when_inventory_zero():
    assert avellaneda_stoikov_skew(0.0, gamma=0.05, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600) == 0.0


def test_skew_negative_when_long():
    # Long inventory → reservation drops below mid → we lean to attract bids
    s = avellaneda_stoikov_skew(100.0, gamma=0.05, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600)
    assert s < 0


def test_skew_positive_when_short():
    s = avellaneda_stoikov_skew(-100.0, gamma=0.05, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600)
    assert s > 0


def test_skew_scales_with_inventory():
    # 2x inventory → 2x skew
    s1 = avellaneda_stoikov_skew(50.0, gamma=0.05, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600)
    s2 = avellaneda_stoikov_skew(100.0, gamma=0.05, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600)
    assert abs(s2 / s1 - 2.0) < 1e-9


def test_half_spread_increases_with_volatility():
    # Bypass clamps so we observe the underlying monotonicity in σ²;
    # in production the clamps are intended (prevent multi-dollar quotes
    # on a 0–1 venue).
    low_vol = avellaneda_stoikov_half_spread(
        gamma=2.0, sigma_sq_per_sec=0.00001, time_to_horizon_sec=10,
        k_arrival_per_sec=2.0, min_half_spread=0.0, max_half_spread=10.0,
    )
    hi_vol = avellaneda_stoikov_half_spread(
        gamma=2.0, sigma_sq_per_sec=0.00005, time_to_horizon_sec=10,
        k_arrival_per_sec=2.0, min_half_spread=0.0, max_half_spread=10.0,
    )
    assert hi_vol > low_vol


def test_half_spread_handles_pathological_inputs():
    # zero gamma: returns risk_term + 0.005 fallback (not divide-by-zero)
    out = avellaneda_stoikov_half_spread(
        gamma=0.0, sigma_sq_per_sec=0.0001, time_to_horizon_sec=600, k_arrival_per_sec=0.5
    )
    assert out > 0  # didn't blow up


# ── Eligibility ─────────────────────────────────────────────────────────

def _market(category: str = "sports_global") -> Market:
    return Market(
        condition_id="0xabc", question="dummy?",
        yes_token_id="t_yes", no_token_id="t_no",
        end_date_iso="2030-01-01T00:00:00Z",
        liquidity=10000.0, volume_24h=10000.0,
        accepting_orders=True, category=category,
    )


def _book(bids: dict[float, float], asks: dict[float, float], age_sec: float = 5.0) -> OrderBook:
    import time as _t
    b = OrderBook(token_id="t")
    b.bids = {float(p): float(s) for p, s in bids.items()}
    b.asks = {float(p): float(s) for p, s in asks.items()}
    b.last_update_ts = _t.time() - age_sec
    return b


def _make_v2(certified: set[str] | None) -> PassivePosterV2:
    book_store = MagicMock()
    book_store.books = {}
    broker = MagicMock()
    broker.positions = {}
    return PassivePosterV2(
        book_store=book_store,
        broker=broker,
        markets_by_token={},
        certified_categories=certified,
    )


def test_eligibility_rejects_uncertified_category():
    v2 = _make_v2(certified={"sports_global"})
    book = _book(bids={0.49: 100}, asks={0.50: 100})
    ok, reason = v2._eligible_market(_market("crypto"), book)
    assert not ok
    assert reason == "uncertified_category"


def test_eligibility_accepts_certified_category():
    v2 = _make_v2(certified={"sports_global"})
    book = _book(bids={0.49: 100}, asks={0.50: 100})
    ok, reason = v2._eligible_market(_market("sports_global"), book)
    assert ok
    assert reason == "ok"


def test_eligibility_rejects_no_book():
    v2 = _make_v2(certified=None)
    ok, reason = v2._eligible_market(_market(), None)
    assert not ok
    assert reason == "no_book"


def test_eligibility_rejects_stale_book():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.50: 100}, age_sec=200)
    ok, reason = v2._eligible_market(_market(), book)
    assert not ok
    assert reason == "stale_book"


def test_eligibility_rejects_wide_spread():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.40: 100}, asks={0.50: 100})  # 10pp spread > 6pp default
    ok, reason = v2._eligible_market(_market(), book)
    assert not ok
    assert reason == "spread_wide"


def test_eligibility_rejects_thin_book():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 5}, asks={0.50: 5})  # only 10 total depth
    ok, reason = v2._eligible_market(_market(), book)
    assert not ok
    assert reason == "thin_book"


def test_eligibility_disabled_when_no_allowlist():
    # certified_categories=None means no gate → trade everything
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.50: 100})
    ok, reason = v2._eligible_market(_market("anything"), book)
    assert ok


# ── Quote computation ───────────────────────────────────────────────────

def test_compute_quote_produces_two_sided():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    q = v2._compute_quote(_market(), book, inventory_yes=0.0)
    assert q is not None
    assert q.bid_price > 0
    assert q.ask_price > q.bid_price
    # Quote should sit inside the existing spread (i.e., not cross top of book)
    assert q.bid_price <= 0.51 - 0.01
    assert q.ask_price >= 0.49 + 0.01


def test_compute_quote_inventory_skew_long_lowers_bid():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    q_flat = v2._compute_quote(_market(), book, inventory_yes=0.0)
    q_long = v2._compute_quote(_market(), book, inventory_yes=200.0)
    assert q_flat is not None and q_long is not None
    # Long inventory → reservation lower → both quotes shifted down
    assert q_long.bid_price <= q_flat.bid_price
    assert q_long.ask_price <= q_flat.ask_price


def test_compute_quote_inventory_skew_short_raises_ask():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    q_flat = v2._compute_quote(_market(), book, inventory_yes=0.0)
    q_short = v2._compute_quote(_market(), book, inventory_yes=-200.0)
    assert q_flat is not None and q_short is not None
    assert q_short.bid_price >= q_flat.bid_price
    assert q_short.ask_price >= q_flat.ask_price
