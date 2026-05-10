"""Conformal-Kelly sizing (Vovk 2025; Sun & Boyd arXiv 1812.10371).

Direct implementation of the doc's Problem-7 fix #4. Standard Kelly
betting requires a *point* estimate of win probability; the
distributionally-robust Kelly recipe (Sun & Boyd 2019) operates on a
*set* of plausible probabilities and chooses the bet that maximises
expected log-growth under the worst case in the set.

Conformal-Kelly: use a **conformal predictive distribution** for p
(rather than a single calibrated probability) and compute Kelly sizing
that is robust over the full conformal interval.

Why this matters for Polyagent
------------------------------
The existing Kelly sizing in `paper_broker.py` uses `KELLY_MULT × edge
/ var` with the model's point probability. That is mis-specified when:

  1. The model's confidence interval is wide (the Venn-Abers
     [p_low, p_high] is broad).
  2. The model has been mis-calibrated historically in this category
     ("Kelly Betting Can Be Too Conservative" — arXiv 1710.01786 —
     showed that with mis-specified models even fractional Kelly can
     blow up).

The conformal interval [p_low, p_high] already exists in our pipeline.
The robust Kelly bet under that set is the bet at the *worst-case
end-of-interval*, which is always smaller than point-estimate Kelly.

API
---
- `robust_kelly_fraction(p_low, p_high, price, side, kelly_mult)`
  Returns the fraction of bankroll to bet under the worst-case
  probability in [p_low, p_high].
- `conformal_kelly_sizing(p_point, p_low, p_high, price, side,
   bankroll, max_fraction)` — convenience that returns USDC notional.

References:
  - Vovk, V., "Conformal e-prediction," arXiv 2001.05989, May 2025.
  - Sun, Q. & Boyd, S., "Distributional Robust Kelly Gambling,"
    arXiv 1812.10371, 2018.
  - Lopez de Prado, "Kelly Betting Can Be Too Conservative,"
    arXiv 1710.01786 (the conservative-fail mode this addresses).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class KellyDecision:
    fraction: float            # of bankroll (signed; 0 means don't bet)
    side: str                  # "BUY" (taking YES) or "SELL" (taking NO)
    p_worst: float             # worst-case probability used
    edge_at_worst: float       # signed edge at p_worst
    rationale: str


def _kelly_fraction(p: float, price: float, side: str) -> float:
    """Vanilla Kelly fraction for a single binary outcome.

    On Polymarket the YES token at price `price` resolves to $1 if YES
    wins, $0 otherwise. If we BUY at `price` we win (1−price)/price
    per unit on YES. Standard Kelly for a discrete bet with payout b
    and win prob p:

        f* = (b·p − q) / b      where  q = 1 − p

    For our BUY:
        b = (1 − price) / price
        f* = (b·p − (1−p)) / b
           = p − (1−p)/b
           = p − price (1−p)/(1−price)
        Simplification check: (b p − q)/b = p − q/b
            = p − (1−p) price / (1−price)
        So f* = p − (1−p) · price / (1 − price)

    For SELL (we are selling YES, i.e. buying NO at 1−price):
        replace price ← 1−price and p ← 1−p.
    """
    p = float(p)
    if side.upper() == "SELL":
        p = 1.0 - p
        price = 1.0 - price
    if price <= 0 or price >= 1:
        return 0.0
    # f* = (b p − q) / b with b = (1−price)/price
    b = (1.0 - price) / price
    q = 1.0 - p
    f = (b * p - q) / b
    return float(f)


def robust_kelly_fraction(
    p_low: float,
    p_high: float,
    price: float,
    side: str,
    kelly_mult: float = 0.5,
) -> KellyDecision:
    """Worst-case Kelly fraction across the conformal interval
    [p_low, p_high]. For a BUY (long YES) the worst case is p_low;
    for a SELL (short YES) the worst case is p_high.

    `kelly_mult` is the fractional-Kelly multiplier (default 0.5 =
    half-Kelly; Polyagent's current default).
    """
    side_norm = side.upper().strip()
    if side_norm == "BUY":
        p_worst = float(p_low)
    elif side_norm == "SELL":
        p_worst = float(p_high)
    else:
        return KellyDecision(0.0, side_norm, 0.0, 0.0, "unknown_side")
    if not (0.0 < price < 1.0):
        return KellyDecision(0.0, side_norm, p_worst, 0.0, "invalid_price")
    f_star = _kelly_fraction(p_worst, price, side_norm)
    f_scaled = max(0.0, f_star * float(kelly_mult))
    edge = (p_worst - price) if side_norm == "BUY" else (price - p_worst)
    rationale = (
        "robust_kelly_positive" if f_scaled > 0
        else "no_edge_at_worst_case"
    )
    return KellyDecision(
        fraction=f_scaled,
        side=side_norm,
        p_worst=p_worst,
        edge_at_worst=float(edge),
        rationale=rationale,
    )


def conformal_kelly_sizing(
    p_low: float,
    p_high: float,
    price: float,
    side: str,
    *,
    bankroll: float,
    kelly_mult: float = 0.5,
    max_fraction: float = 0.05,
    min_notional: float = 1.0,
) -> dict:
    """High-level wrapper. Returns a dict with notional, fraction,
    p_worst, and rationale — directly usable by `paper_broker.submit`.

    Args:
        p_low, p_high: conformal-Venn-Abers interval for P(YES).
        price: top-of-book quote for the side we'd take.
        side: 'BUY' or 'SELL'.
        bankroll: total bankroll for sizing (typically NAV − reserved).
        kelly_mult: fractional-Kelly multiplier.
        max_fraction: hard cap on bankroll fraction per trade.
        min_notional: minimum USDC to act on; below this, return 0.
    """
    if bankroll <= 0:
        return {"notional": 0.0, "fraction": 0.0, "p_worst": p_low,
                "rationale": "no_bankroll"}
    dec = robust_kelly_fraction(p_low, p_high, price, side, kelly_mult)
    fraction = min(dec.fraction, max_fraction)
    notional = bankroll * fraction
    if notional < min_notional:
        return {"notional": 0.0, "fraction": fraction,
                "p_worst": dec.p_worst,
                "edge_at_worst": dec.edge_at_worst,
                "rationale": "below_min_notional"}
    return {
        "notional": float(notional),
        "fraction": float(fraction),
        "p_worst": float(dec.p_worst),
        "edge_at_worst": float(dec.edge_at_worst),
        "rationale": dec.rationale,
    }
