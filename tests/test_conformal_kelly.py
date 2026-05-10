"""Tests for conformal-Kelly sizing (Sun & Boyd 2019)."""
from __future__ import annotations

from polyagent.risk.conformal_kelly import (
    _kelly_fraction, robust_kelly_fraction, conformal_kelly_sizing,
)


def test_kelly_fraction_no_edge_zero():
    # If p = price, edge = 0, Kelly = 0.
    f = _kelly_fraction(p=0.5, price=0.5, side="BUY")
    assert abs(f) < 1e-9


def test_kelly_fraction_positive_edge_positive_size():
    f = _kelly_fraction(p=0.6, price=0.5, side="BUY")
    assert f > 0


def test_kelly_fraction_negative_edge_negative_size():
    f = _kelly_fraction(p=0.4, price=0.5, side="BUY")
    assert f < 0  # we'd actually short, not long


def test_robust_kelly_uses_low_end_for_buy():
    dec = robust_kelly_fraction(p_low=0.55, p_high=0.65, price=0.50,
                                side="BUY", kelly_mult=1.0)
    # Worst case for BUY is p_low (smallest probability ⇒ smallest size).
    assert dec.p_worst == 0.55
    assert dec.fraction > 0
    assert abs(dec.edge_at_worst - 0.05) < 1e-9


def test_robust_kelly_uses_high_end_for_sell():
    dec = robust_kelly_fraction(p_low=0.35, p_high=0.45, price=0.50,
                                side="SELL", kelly_mult=1.0)
    # Worst case for SELL is p_high (largest YES prob ⇒ smallest NO bet size).
    assert dec.p_worst == 0.45
    assert abs(dec.edge_at_worst - 0.05) < 1e-9


def test_robust_kelly_no_edge_zero():
    dec = robust_kelly_fraction(p_low=0.40, p_high=0.60, price=0.50,
                                side="BUY", kelly_mult=0.5)
    # p_worst = 0.40 ⇒ no edge ⇒ no bet.
    assert dec.fraction == 0
    assert dec.rationale == "no_edge_at_worst_case"


def test_kelly_mult_scales_fraction():
    f_full = robust_kelly_fraction(p_low=0.55, p_high=0.65, price=0.50,
                                   side="BUY", kelly_mult=1.0).fraction
    f_half = robust_kelly_fraction(p_low=0.55, p_high=0.65, price=0.50,
                                   side="BUY", kelly_mult=0.5).fraction
    assert abs(f_half - f_full * 0.5) < 1e-9


def test_conformal_sizing_caps_at_max_fraction():
    s = conformal_kelly_sizing(p_low=0.80, p_high=0.90, price=0.50,
                               side="BUY", bankroll=10_000,
                               kelly_mult=2.0, max_fraction=0.05)
    assert s["fraction"] == 0.05
    assert s["notional"] == 500.0


def test_conformal_sizing_below_min_notional_zero():
    s = conformal_kelly_sizing(p_low=0.51, p_high=0.52, price=0.50,
                               side="BUY", bankroll=100,
                               kelly_mult=0.1, max_fraction=0.05,
                               min_notional=10.0)
    # tiny edge × tiny bankroll < min_notional
    assert s["notional"] == 0
    assert s["rationale"] == "below_min_notional"


def test_conformal_sizing_no_bankroll():
    s = conformal_kelly_sizing(p_low=0.5, p_high=0.5, price=0.5, side="BUY",
                               bankroll=0)
    assert s["notional"] == 0
    assert s["rationale"] == "no_bankroll"
