"""Asymmetric / cost-sensitive Brier score (pmwhybetter.md Problem-8 #4).

Standard Brier weights false-positives and false-negatives symmetrically:

    Brier(p, y) = (p − y)²

In trading this is *wrong* whenever the cost of a wrong-side bet
differs from the cost of an over-confident-but-right call. Two
common asymmetries on Polymarket:

1. **Longshot bias** — a 0.05 market resolving YES pays 19× the
   notional; a 0.95 market resolving YES pays 0.053×. Same Brier
   on both sides hides the asymmetric P&L impact.

2. **Coverage cost** — refusing to act has a cost (lost opportunity)
   different from acting wrongly. Standard scoring lumps both into
   the same loss.

This module implements three published asymmetric variants:

- **Cost-weighted Brier** — multiply each term by a per-trade cost
  matrix C[predicted_direction, true_outcome]. Default cost matrix
  encodes the Polymarket fee schedule: BUY-YES on 0.50 ask + 1% fee
  ⇒ cost(YES, YES) = 0.01, cost(YES, NO) = 0.50.

- **Linear-economic Brier** — replace `(p − y)²` with the *realized
  Kelly-edge loss*: |Δedge| × |position|. Lifts directly from the
  ACM Computing Surveys 2025 trading-via-selective-classification
  doi:10.1145/3727633.

- **Coverage-asymmetric Brier** — inverse-coverage-weighted variant
  for selective classification: the score *rewards* a model that
  covers more high-confidence cells (Heng-Soh-style abstention pairing).

API
---
- `cost_weighted_brier(p, y, cost_matrix)` → float
- `linear_economic_brier(p, market_p, y, side, notional)` → float
- `coverage_asymmetric_brier(p, y, was_admitted, coverage)` → float
- `evaluate_strategy(predictions, outcomes, side, notionals)` → dict
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


# Default cost matrix C[predicted_direction, true_outcome].
# Rows: our action (BUY=1, NO_ACTION=0, SELL=-1)
# Cols: outcome (YES=1, NO=0)
# Values: bps cost. Buying YES when YES wins ⇒ 100 bps (1% taker fee).
#         Buying YES when NO wins ⇒ 10000 bps (we lose the whole stake
#         minus rebate, which paper mode ignores).
DEFAULT_COST_MATRIX = {
    ("BUY", 1): 0.01,    # right call → fees only
    ("BUY", 0): 1.0,     # wrong call → full loss
    ("SELL", 1): 1.0,    # wrong call short → full loss
    ("SELL", 0): 0.01,   # right call short → fees only
    ("NO_ACTION", 1): 0.0,  # abstain → no realised loss
    ("NO_ACTION", 0): 0.0,
}


def cost_weighted_brier(
    p: float,
    y: int,
    *,
    side: str = "BUY",
    cost_matrix: dict | None = None,
) -> float:
    """Cost-weighted Brier: (p − y)² × C[side, y]."""
    p = float(p)
    y = int(y)
    side = side.upper()
    cost = (cost_matrix or DEFAULT_COST_MATRIX).get((side, y), 1.0)
    return float((p - y) ** 2 * cost)


def linear_economic_brier(
    p: float,
    market_p: float,
    y: int,
    *,
    side: str = "BUY",
    notional: float = 100.0,
) -> float:
    """Realized economic loss (USD) of a one-sided bet relative to no
    action.

    If we BUY YES at market_p and YES wins, our realized profit per
    notional = (1 − market_p) / market_p × notional. If NO wins, loss
    = −notional. The "Brier" here is the *negative* of expected
    realized P&L under the model's p — i.e. how badly we'd score this
    decision against the true outcome.
    """
    side = side.upper()
    notional = float(notional)
    if notional <= 0 or market_p <= 0 or market_p >= 1:
        return 0.0
    if side == "BUY":
        payoff = ((1 - market_p) / market_p) * notional if y == 1 else -notional
    elif side == "SELL":
        # Symmetric — sold YES at market_p = bought NO at (1 − market_p)
        payoff = (market_p / (1 - market_p)) * notional if y == 0 else -notional
    else:
        return 0.0
    expected = p * payoff if side == "BUY" and y == 1 else \
               (1 - p) * payoff if side == "BUY" and y == 0 else \
               (1 - p) * payoff if side == "SELL" and y == 0 else \
               p * payoff
    return -float(expected)


def coverage_asymmetric_brier(
    p: float,
    y: int,
    was_admitted: bool,
    *,
    coverage: float = 0.4,
) -> float:
    """Coverage-asymmetric Brier (paired with selective classification).

    Standard Brier when admitted. When abstained, charge a *coverage
    cost* = (1 − coverage) × baseline_brier(0.5, y). This rewards a
    model that covers more — i.e. abstains less — because abstaining
    costs less than being wrong, but it's not free."""
    p = float(p)
    y = int(y)
    if was_admitted:
        return float((p - y) ** 2)
    # Abstain cost: a fraction of the worst-case naive Brier (which is
    # 0.25 at p=0.5). Smaller coverage → higher abstain penalty.
    base = (0.5 - y) ** 2
    return float((1 - coverage) * base)


@dataclass
class StrategyEvaluation:
    n: int
    mean_cost_brier: float
    mean_linear_economic: float
    mean_coverage_asym: float
    total_realized_pnl_usd: float


def evaluate_strategy(
    predictions: list[float],
    outcomes: list[int],
    *,
    sides: list[str] | None = None,
    market_prices: list[float] | None = None,
    notionals: list[float] | None = None,
    admitted: list[bool] | None = None,
    coverage: float = 0.4,
) -> StrategyEvaluation:
    """One-pass evaluator that computes all three asymmetric Briers
    and the total realized P&L (USD) implied by the linear-economic
    formulation. Lists must be equal length."""
    n = len(predictions)
    if n == 0 or len(outcomes) != n:
        return StrategyEvaluation(0, 0.0, 0.0, 0.0, 0.0)
    sides = sides or ["BUY"] * n
    market_prices = market_prices or [0.5] * n
    notionals = notionals or [100.0] * n
    admitted = admitted or [True] * n

    cwb, leb, cab = [], [], []
    realized = 0.0
    for i in range(n):
        cwb.append(cost_weighted_brier(predictions[i], outcomes[i], side=sides[i]))
        # linear_economic_brier returns the *expected* loss; the
        # *realized* P&L is just the deterministic payoff at the
        # outcome with no expectation.
        if admitted[i]:
            leb.append(linear_economic_brier(
                predictions[i], market_prices[i], outcomes[i],
                side=sides[i], notional=notionals[i],
            ))
            # Realized P&L of the trade
            mp = float(market_prices[i])
            n_ = float(notionals[i])
            s = sides[i].upper()
            if mp <= 0 or mp >= 1:
                realized += 0
            elif s == "BUY":
                realized += ((1 - mp) / mp) * n_ if outcomes[i] == 1 else -n_
            elif s == "SELL":
                realized += (mp / (1 - mp)) * n_ if outcomes[i] == 0 else -n_
        else:
            leb.append(0.0)
        cab.append(coverage_asymmetric_brier(
            predictions[i], outcomes[i], admitted[i], coverage=coverage,
        ))

    return StrategyEvaluation(
        n=n,
        mean_cost_brier=float(np.mean(cwb)) if cwb else 0.0,
        mean_linear_economic=float(np.mean(leb)) if leb else 0.0,
        mean_coverage_asym=float(np.mean(cab)) if cab else 0.0,
        total_realized_pnl_usd=float(realized),
    )
