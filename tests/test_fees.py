"""Tests for per-category fees + maker rebate + cancel-latency penalty."""
from __future__ import annotations

import os

from polyagent.risk.fees import (
    DEFAULT_REBATE_SHARE,
    cancel_latency_penalty,
    compute_fees,
    _category_rate,
)


# ── Per-category rate resolution ────────────────────────────────────────

def test_rate_known_category():
    assert _category_rate("crypto") == 0.018
    assert _category_rate("sports_global") == 0.0075
    assert _category_rate("politics_us") == 0.010


def test_rate_unknown_category_defaults_to_other():
    assert _category_rate("not_a_real_category") == 0.010
    assert _category_rate(None) == 0.010
    assert _category_rate("") == 0.010


def test_rate_env_override(monkeypatch):
    monkeypatch.setenv("FEE_RATE_SPORTS_GLOBAL", "0.005")
    assert _category_rate("sports_global") == 0.005


def test_rate_env_override_invalid_falls_through(monkeypatch):
    monkeypatch.setenv("FEE_RATE_SPORTS_GLOBAL", "not_a_number")
    assert _category_rate("sports_global") == 0.0075


# ── compute_fees ────────────────────────────────────────────────────────

def test_taker_pays_full_fee_no_rebate():
    f = compute_fees(notional=100.0, category="sports_global", is_maker=False)
    assert f.is_maker is False
    assert f.taker_fee_paid == 100.0 * 0.0075  # $0.75
    assert f.maker_rebate_credited == 0.0


def test_maker_zero_fee_receives_rebate():
    f = compute_fees(notional=100.0, category="sports_global", is_maker=True)
    assert f.is_maker is True
    assert f.taker_fee_paid == 0.0
    # 22% of the would-be taker fee on $100 notional at 0.75%
    assert abs(f.maker_rebate_credited - 100.0 * 0.0075 * DEFAULT_REBATE_SHARE) < 1e-9


def test_crypto_taker_fee_higher_than_sports():
    crypto_taker = compute_fees(notional=100.0, category="crypto", is_maker=False)
    sports_taker = compute_fees(notional=100.0, category="sports_global", is_maker=False)
    assert crypto_taker.taker_fee_paid > sports_taker.taker_fee_paid
    assert crypto_taker.taker_fee_rate == 0.018
    assert sports_taker.taker_fee_rate == 0.0075


def test_rebate_share_override():
    f = compute_fees(notional=100.0, category="sports_global", is_maker=True, rebate_share=0.30)
    assert f.maker_rebate_rate == 0.30
    assert abs(f.maker_rebate_credited - 100.0 * 0.0075 * 0.30) < 1e-9


# ── cancel_latency_penalty ──────────────────────────────────────────────

def test_cancel_latency_buy_pays_more():
    # BUY at 0.50 with 0.005/sec realized vol, 2-sec block:
    # drift = 0.005 * sqrt(2) ≈ 0.00707
    eff = cancel_latency_penalty(0.50, "BUY", 0.005)
    assert eff > 0.50
    assert abs(eff - 0.5071) < 1e-3


def test_cancel_latency_sell_receives_less():
    eff = cancel_latency_penalty(0.50, "SELL", 0.005)
    assert eff < 0.50
    assert abs(eff - 0.4929) < 1e-3


def test_cancel_latency_clamped_to_unit_interval():
    # Extreme vol can't push price outside [0, 1]
    eff_buy = cancel_latency_penalty(0.99, "BUY", 1.0)
    assert eff_buy <= 1.0
    eff_sell = cancel_latency_penalty(0.01, "SELL", 1.0)
    assert eff_sell >= 0.0


def test_cancel_latency_with_no_vol_uses_default():
    # No realized_vol → falls back to default 0.005
    eff_buy = cancel_latency_penalty(0.50, "BUY", None)
    eff_buy_default = cancel_latency_penalty(0.50, "BUY", 0.005)
    assert abs(eff_buy - eff_buy_default) < 1e-9
