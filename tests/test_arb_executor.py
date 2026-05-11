"""Tests for the multi-detector arb executor."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from polyagent.strategies.arb_executor import (
    ArbExecutor, _BasketLeg, _PricedMarket,
)


@dataclass
class _MockBook:
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)

    def best_bid(self):
        if not self.bids:
            return None
        p = max(self.bids.keys())
        return (p, self.bids[p])

    def best_ask(self):
        if not self.asks:
            return None
        p = min(self.asks.keys())
        return (p, self.asks[p])


class _MockBookStore:
    def __init__(self):
        self.books = {}


@dataclass
class _MockMarket:
    yes_token_id: str
    no_token_id: str = ""
    condition_id: str = "c1"
    question: str = ""
    category: str = "sports_global"
    event_id: str | None = None
    neg_risk_event_id: str | None = None


class _MockBroker:
    """Records submit calls and lets test set per-call fill amounts."""
    def __init__(self):
        self.calls = []
        self.fills_to_return = []   # consumed in order

    async def submit(self, **kwargs):
        self.calls.append(kwargs)
        if self.fills_to_return:
            return self.fills_to_return.pop(0)
        return float(kwargs.get("max_size", 0.0))


def test_cooldown_blocks_repeat():
    bs = _MockBookStore()
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs)
    e._mark_used("k1")
    assert e._cooldown_ok("k1") is False
    assert e._cooldown_ok("k2") is True


def test_summary_starts_zero():
    e = ArbExecutor(broker=_MockBroker(), book_store=_MockBookStore())
    s = e.summary()
    assert s["baskets_executed"] == 0
    assert s["legs_executed"] == 0
    assert s["legs_unwound"] == 0


def test_handle_basket_full_fills():
    e = ArbExecutor(broker=_MockBroker(), book_store=_MockBookStore())
    legs = [
        _BasketLeg(token_id="t1", condition_id="c1", side="BUY",
                   quote_price=0.4, target_size=50),
        _BasketLeg(token_id="t2", condition_id="c1", side="BUY",
                   quote_price=0.5, target_size=50),
    ]
    asyncio.run(e._handle_basket_result("b1", legs, [50.0, 50.0]))
    assert e._baskets_executed == 1
    assert e._legs_executed == 2
    assert e._legs_unwound == 0


def test_handle_basket_partial_triggers_unwind():
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=_MockBookStore())
    legs = [
        _BasketLeg(token_id="t1", condition_id="c1", side="BUY",
                   quote_price=0.4, target_size=50),
        _BasketLeg(token_id="t2", condition_id="c1", side="BUY",
                   quote_price=0.5, target_size=50),
    ]
    # One leg filled, one didn't
    asyncio.run(e._handle_basket_result("b1", legs, [50.0, 0.0]))
    # Should NOT count as basket_executed (partial)
    assert e._baskets_executed == 0
    # Should attempt to unwind the one filled leg
    assert e._legs_unwound == 1
    # The unwind call should be opposite-side SELL
    assert br.calls[-1]["side"] == "SELL"
    assert br.calls[-1]["strategy"] == "arb_unwind"


def test_inplay_skip_when_no_groups():
    """Empty inplay_groups means the in-play scanner is a no-op."""
    e = ArbExecutor(
        broker=_MockBroker(), book_store=_MockBookStore(),
        inplay_groups=[],
    )
    # _scan_inplay shouldn't raise on empty groups
    asyncio.run(e._scan_inplay())


def test_negrisk_basket_executes_full_legs():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.32: 100})
    bs.books["t2"] = _MockBook(asks={0.32: 100})
    bs.books["t3"] = _MockBook(asks={0.32: 100})
    # Sum YES = 0.96 → arb gap 400 bps (inside the anti-phantom band:
    # sum >= 0.70 floor AND gap <= 500 bps cap)
    markets = [
        _MockMarket(yes_token_id="t1", condition_id="c1", neg_risk_event_id="ev1"),
        _MockMarket(yes_token_id="t2", condition_id="c2", neg_risk_event_id="ev1"),
        _MockMarket(yes_token_id="t3", condition_id="c3", neg_risk_event_id="ev1"),
    ]
    br = _MockBroker()
    e = ArbExecutor(
        broker=br, book_store=bs, markets=markets,
        negrisk_min_bps=50.0, min_leg_size=50.0, max_basket_notional=200.0,
    )
    asyncio.run(e._scan_negrisk())
    # Should have placed 3 BUY orders (LONG_YES_ALL)
    n_buys = sum(1 for c in br.calls if c.get("side") == "BUY")
    assert n_buys == 3
    assert e._baskets_executed == 1


def test_negrisk_skips_under_min_gap():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.49: 100})
    bs.books["t2"] = _MockBook(asks={0.50: 100})
    # Sum = 0.99 → arb gap 100 bps; with floor at 200 we skip
    markets = [
        _MockMarket(yes_token_id="t1", condition_id="c1", neg_risk_event_id="ev2"),
        _MockMarket(yes_token_id="t2", condition_id="c2", neg_risk_event_id="ev2"),
    ]
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs, markets=markets,
                    negrisk_min_bps=200.0)
    asyncio.run(e._scan_negrisk())
    assert len(br.calls) == 0


def test_negrisk_skips_partial_group_under_30pct():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.10: 100})
    bs.books["t2"] = _MockBook(asks={0.15: 100})
    # Sum = 0.25 < 0.30 → assume partial group, skip even though gap is huge
    markets = [
        _MockMarket(yes_token_id="t1", condition_id="c1", neg_risk_event_id="ev3"),
        _MockMarket(yes_token_id="t2", condition_id="c2", neg_risk_event_id="ev3"),
    ]
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs, markets=markets,
                    negrisk_min_bps=50.0)
    asyncio.run(e._scan_negrisk())
    assert len(br.calls) == 0


def test_negrisk_skips_partial_long_below_sum_floor():
    """LONG_YES_ALL needs sum_yes >= 0.70 to defeat partial-group artifacts."""
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.10: 100})
    bs.books["t2"] = _MockBook(asks={0.20: 100})
    # Sum = 0.30 → arb gap 7000 bps but it's the partial-group case
    markets = [
        _MockMarket(yes_token_id="t1", condition_id="c1", neg_risk_event_id="ev_part_long"),
        _MockMarket(yes_token_id="t2", condition_id="c2", neg_risk_event_id="ev_part_long"),
    ]
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs, markets=markets,
                    negrisk_min_bps=50.0,
                    negrisk_min_sum_long=0.70,
                    negrisk_max_gap_bps=10000.0)
    asyncio.run(e._scan_negrisk())
    # 0.30 < 0.70 floor ⇒ blocked even though gap is large
    assert len(br.calls) == 0


def test_negrisk_skips_phantom_wide_gap():
    """Gaps over the max-gap cap are phantom partial-group artifacts."""
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.05: 100})
    bs.books["t2"] = _MockBook(asks={0.05: 100})
    # Sum = 0.10 → 9000 bps gap; but sum floor is the first guard.
    # Force the phantom path: high sum_yes but huge gap.
    bs.books["t3"] = _MockBook(asks={0.99: 100})
    bs.books["t4"] = _MockBook(asks={0.99: 100})
    bs.books["t5"] = _MockBook(asks={0.99: 100})
    # Sum = 2.97 → 19700 bps SHORT_YES_ALL gap; sum_yes > max_sum_short=1.30
    markets = [
        _MockMarket(yes_token_id="t3", condition_id="c3", neg_risk_event_id="ev_phantom"),
        _MockMarket(yes_token_id="t4", condition_id="c4", neg_risk_event_id="ev_phantom"),
        _MockMarket(yes_token_id="t5", condition_id="c5", neg_risk_event_id="ev_phantom"),
    ]
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs, markets=markets,
                    negrisk_min_bps=50.0,
                    negrisk_max_sum_short=1.30,
                    negrisk_max_gap_bps=500.0)
    asyncio.run(e._scan_negrisk())
    # sum 2.97 > 1.30 max_sum_short ⇒ blocked
    assert len(br.calls) == 0


def test_negrisk_executes_within_gap_band():
    """A legitimate small-gap arb in the safe band should still trigger."""
    bs = _MockBookStore()
    # 3 legs summing to 0.97 → 300 bps LONG gap; well inside band
    bs.books["t1"] = _MockBook(asks={0.33: 100})
    bs.books["t2"] = _MockBook(asks={0.32: 100})
    bs.books["t3"] = _MockBook(asks={0.32: 100})
    markets = [
        _MockMarket(yes_token_id="t1", condition_id="c1", neg_risk_event_id="ev_ok"),
        _MockMarket(yes_token_id="t2", condition_id="c2", neg_risk_event_id="ev_ok"),
        _MockMarket(yes_token_id="t3", condition_id="c3", neg_risk_event_id="ev_ok"),
    ]
    br = _MockBroker()
    e = ArbExecutor(broker=br, book_store=bs, markets=markets,
                    negrisk_min_bps=50.0,
                    negrisk_min_sum_long=0.70,
                    negrisk_max_gap_bps=500.0,
                    min_leg_size=50.0)
    asyncio.run(e._scan_negrisk())
    # 3 BUY legs placed
    assert sum(1 for c in br.calls if c.get("side") == "BUY") == 3


def test_cond_for_token_lookup():
    markets = [_MockMarket(yes_token_id="tk", condition_id="cx")]
    e = ArbExecutor(broker=_MockBroker(), book_store=_MockBookStore(),
                    markets=markets)
    assert e._cond_for_token("tk") == "cx"
    assert e._cond_for_token("missing") is None
