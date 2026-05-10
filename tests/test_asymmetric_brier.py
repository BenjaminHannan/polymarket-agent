"""Tests for asymmetric / cost-sensitive Brier."""
from __future__ import annotations

from polyagent.eval.asymmetric_brier import (
    cost_weighted_brier,
    linear_economic_brier,
    coverage_asymmetric_brier,
    evaluate_strategy,
)


def test_cost_weighted_zero_when_perfect():
    assert cost_weighted_brier(1.0, 1, side="BUY") == 0.0
    assert cost_weighted_brier(0.0, 0, side="BUY") == 0.0


def test_cost_weighted_wrong_side_charged_more():
    # Buying YES, NO wins → high cost
    bad = cost_weighted_brier(0.9, 0, side="BUY")
    # Right call, lower cost
    good = cost_weighted_brier(0.1, 0, side="BUY")
    assert bad > good


def test_linear_economic_returns_finite():
    r = linear_economic_brier(0.5, 0.4, 1, side="BUY", notional=100)
    assert r is not None
    assert abs(r) < 1000


def test_coverage_asym_lower_when_abstained():
    """An abstain penalty should be smaller than a confident-wrong Brier."""
    abstain = coverage_asymmetric_brier(0.5, 1, was_admitted=False, coverage=0.4)
    wrong = coverage_asymmetric_brier(0.9, 0, was_admitted=True)
    assert abstain < wrong


def test_evaluate_strategy_returns_dataclass():
    pred = [0.6, 0.7, 0.3, 0.4]
    out = [1, 1, 0, 0]
    res = evaluate_strategy(pred, out)
    assert res.n == 4
    assert 0.0 <= res.mean_cost_brier <= 1.0


def test_evaluate_strategy_realized_pnl_with_notionals():
    pred = [0.6, 0.4]
    mkt = [0.5, 0.5]
    out = [1, 0]
    sides = ["BUY", "BUY"]
    notionals = [100, 100]
    res = evaluate_strategy(
        pred, out, sides=sides,
        market_prices=mkt, notionals=notionals,
    )
    # BUY YES at 0.5, YES wins: profit = (1-0.5)/0.5 × 100 = 100
    # BUY YES at 0.5, NO wins: loss = -100
    # Net: 0
    assert abs(res.total_realized_pnl_usd) < 1e-6


def test_evaluate_strategy_empty_returns_zeros():
    res = evaluate_strategy([], [])
    assert res.n == 0
    assert res.mean_cost_brier == 0.0
