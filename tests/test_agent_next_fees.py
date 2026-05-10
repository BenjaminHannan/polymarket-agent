"""Tests for agent-next-style exact-Polymarket fee model."""
from __future__ import annotations

from polyagent.risk.agent_next_fees import (
    agent_next_fee, effective_taker_fee_bps,
)


def test_zero_fee_on_invalid_price():
    f = agent_next_fee(price=0, side="BUY", shares=100, category="sports")
    assert f.taker_fee_usd == 0.0
    f2 = agent_next_fee(price=1.0, side="BUY", shares=100, category="sports")
    assert f2.taker_fee_usd == 0.0


def test_fee_uses_min_side():
    """Fee = bps × min(p, 1-p) × shares."""
    # sports_global = 75 bps, p=0.3, shares=100
    # fee = 75/10000 × 0.3 × 100 = 0.225
    f = agent_next_fee(price=0.3, side="BUY", shares=100, category="sports_global")
    assert abs(f.taker_fee_usd - 0.225) < 1e-6


def test_fee_symmetric_in_price():
    """Fee at p should equal fee at 1-p (same min side)."""
    a = agent_next_fee(price=0.3, side="BUY", shares=100, category="sports_global")
    b = agent_next_fee(price=0.7, side="BUY", shares=100, category="sports_global")
    assert abs(a.taker_fee_usd - b.taker_fee_usd) < 1e-9


def test_maker_collects_rebate():
    f_taker = agent_next_fee(price=0.5, side="BUY", shares=100, category="politics", is_maker=False)
    f_maker = agent_next_fee(price=0.5, side="BUY", shares=100, category="politics", is_maker=True)
    assert f_taker.effective_fee_usd > 0
    assert f_maker.effective_fee_usd < 0
    # Maker rebate = rebate_share × taker_fee
    assert abs(f_maker.effective_fee_usd + f_taker.taker_fee_usd * 0.22) < 1e-6


def test_unknown_category_falls_back_to_other():
    f = agent_next_fee(price=0.5, side="BUY", shares=100, category="bogus_category")
    # other = 100 bps; fee = 1% × 0.5 × 100 = 0.5
    assert abs(f.taker_fee_usd - 0.5) < 1e-6


def test_effective_bps_of_notional_lower_on_favourites():
    """Buying favourites pays much less fee per $ than buying coin-flips."""
    bps_coin = effective_taker_fee_bps(price=0.5, category="sports_global")
    bps_fav = effective_taker_fee_bps(price=0.95, category="sports_global")
    # bps_fav = 75 × 0.05 / 0.95 ≈ 3.95; bps_coin = 75
    assert bps_fav < bps_coin
    assert bps_fav < 5.0


def test_rate_bps_used_is_nominal():
    f = agent_next_fee(price=0.3, side="BUY", shares=100, category="sports_global")
    assert f.rate_bps_used == 75.0
