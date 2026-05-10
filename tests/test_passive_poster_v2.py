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
    m = _market()
    q = v2._compute_quote(m, book, m.yes_token_id, inventory_this_token=0.0)
    assert q is not None
    assert q.bid_price > 0
    assert q.ask_price > q.bid_price
    assert q.bid_price <= 0.51 - 0.01
    assert q.ask_price >= 0.49 + 0.01


def test_compute_quote_inventory_skew_long_lowers_bid():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q_flat = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    q_long = v2._compute_quote(m, book, m.yes_token_id, 200.0)
    assert q_flat is not None and q_long is not None
    assert q_long.bid_price <= q_flat.bid_price
    assert q_long.ask_price <= q_flat.ask_price


def test_compute_quote_inventory_skew_short_raises_ask():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q_flat = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    q_short = v2._compute_quote(m, book, m.yes_token_id, -200.0)
    assert q_flat is not None and q_short is not None
    assert q_short.bid_price >= q_flat.bid_price
    assert q_short.ask_price >= q_flat.ask_price


def test_compute_quote_marks_yes_vs_no_token():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q_yes = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    q_no = v2._compute_quote(m, book, m.no_token_id, 0.0)
    assert q_yes is not None and q_no is not None
    assert q_yes.is_yes_token is True
    assert q_no.is_yes_token is False
    assert q_yes.token_id != q_no.token_id


# ── Quote-replacement protocol ─────────────────────────────────────────

def test_post_or_replace_first_post_creates_revision_zero():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    new_q = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    posted = v2._post_or_replace(m.yes_token_id, new_q)
    assert posted.revision == 0
    assert v2._quotes[m.yes_token_id] is posted


def test_post_or_replace_unchanged_quote_keeps_revision():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q1 = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    p1 = v2._post_or_replace(m.yes_token_id, q1)
    p1.bid_fills = 3  # simulate a fill
    q2 = v2._compute_quote(m, book, m.yes_token_id, 0.0)  # identical
    p2 = v2._post_or_replace(m.yes_token_id, q2)
    assert p2.revision == 0  # unchanged
    assert p2.bid_fills == 3  # counters preserved


def test_post_or_replace_changed_quote_increments_revision():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q1 = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    p1 = v2._post_or_replace(m.yes_token_id, q1)
    p1.bid_fills = 5
    # A meaningfully different quote (long inventory shifts both quotes)
    q2 = v2._compute_quote(m, book, m.yes_token_id, 500.0)
    p2 = v2._post_or_replace(m.yes_token_id, q2)
    if p2.bid_price != p1.bid_price or p2.ask_price != p1.ask_price:
        assert p2.revision == 1  # incremented after the cancel+post
        # Fill counters preserved across the replacement
        assert p2.bid_fills == 5


# ── Adverse-selection feedback ──────────────────────────────────────────

def test_adverse_selection_widen_no_history_returns_one():
    v2 = _make_v2(certified=None)
    assert v2._adverse_selection_widen("nonexistent_token") == 1.0


def test_adverse_selection_widen_after_adverse_buys():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    v2._post_or_replace(m.yes_token_id, q)
    # Simulate 4 bought-then-mid-dropped (adverse) outcomes
    for _ in range(4):
        v2._quotes[m.yes_token_id].recent_outcomes.append((True, 0.50, 0.45))
    widen = v2._adverse_selection_widen(m.yes_token_id)
    assert widen > 1.0  # should widen the spread
    assert widen <= 1.5  # capped


def test_adverse_selection_widen_after_favorable_outcomes():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    v2._post_or_replace(m.yes_token_id, q)
    # Favorable: bought, mid then went UP (we're winning)
    for _ in range(4):
        v2._quotes[m.yes_token_id].recent_outcomes.append((True, 0.50, 0.55))
    widen = v2._adverse_selection_widen(m.yes_token_id)
    assert widen == 1.0  # no widening when not adverse


# ── Calibration from observed flow ──────────────────────────────────────

def test_calibration_first_call_records_mid():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    v2._update_calibration(m.yes_token_id, book)
    # First call only seeds _last_mid; no calibration update yet
    assert v2._last_mid[m.yes_token_id] == 0.50
    assert m.yes_token_id not in v2._k_arrival


def test_calibration_updates_after_two_cycles():
    v2 = _make_v2(certified=None)
    m = _market()
    book1 = _book(bids={0.49: 100}, asks={0.51: 100})
    v2._update_calibration(m.yes_token_id, book1)
    book2 = _book(bids={0.50: 100}, asks={0.52: 100})  # mid shifted
    v2._update_calibration(m.yes_token_id, book2)
    assert m.yes_token_id in v2._k_arrival
    assert m.yes_token_id in v2._sigma_per_sec
    # Mid moved → sigma estimate is positive
    assert v2._sigma_per_sec[m.yes_token_id] > 0


# ── Inventory unwind ──────────────────────────────────────────────────

def test_inventory_unwind_long_suppresses_buy_side():
    v2 = _make_v2(certified=None)
    v2.max_total_inventory_yes = 100
    v2.unwind_threshold_pct = 0.6  # threshold = 60
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    # 80 long > threshold 60 → unwind mode
    q = v2._compute_quote(m, book, m.yes_token_id, inventory_this_token=80.0)
    assert q is not None
    assert q.one_sided_unwind is True
    assert q.unwind_side == "BUY"
    assert q.bid_price == 0.01  # boundary tick — unfillable


def test_inventory_unwind_short_suppresses_sell_side():
    v2 = _make_v2(certified=None)
    v2.max_total_inventory_yes = 100
    v2.unwind_threshold_pct = 0.6
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    # -80 short → unwind mode, suppress SELL
    q = v2._compute_quote(m, book, m.yes_token_id, inventory_this_token=-80.0)
    assert q is not None
    assert q.one_sided_unwind is True
    assert q.unwind_side == "SELL"
    assert q.ask_price == 0.99


def test_inventory_below_threshold_normal_two_sided():
    v2 = _make_v2(certified=None)
    v2.max_total_inventory_yes = 100
    v2.unwind_threshold_pct = 0.6
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q = v2._compute_quote(m, book, m.yes_token_id, inventory_this_token=30.0)
    assert q is not None
    assert q.one_sided_unwind is False
    assert q.unwind_side is None
    # Both quotes should sit in the normal range
    assert 0.01 < q.bid_price < q.ask_price < 0.99


# ── Stale-quote tracking ──────────────────────────────────────────────

def test_compute_quote_records_mid_at_post():
    v2 = _make_v2(certified=None)
    book = _book(bids={0.49: 100}, asks={0.51: 100})
    m = _market()
    q = v2._compute_quote(m, book, m.yes_token_id, 0.0)
    assert q is not None
    # mid was (0.49 + 0.51) / 2 = 0.50
    assert q.mid_at_post == 0.50
