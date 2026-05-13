"""Micro-price, VAMP, and queue-imbalance features.

Direct implementation of the doc's Problem-10 fix #4 — OFI on multi-level
book; Cont-Kukanov-Stoikov 2014; Gould & Bonart 2015; arXiv 2602.00776
(Dec 2025: cross-asset SHAP-stable patterns show order-book-imbalance
and adverse-selection features rank stably across BTC/LTC/ETC/ENJ/ROSE).

The mid-price `(bid + ask) / 2` is a crude estimator of "true" price
in markets where the book is asymmetric or thin. Three published
refinements:

1. **Micro-price** (Stoikov 2017): inverse-volume-weighted mid.
   When the bid has thicker volume than the ask, the true price is
   biased toward the ask (because the next trade is more likely a
   buyer crossing the spread).

        micro = (ask_vol * bid + bid_vol * ask) / (ask_vol + bid_vol)

   Note the *inverse* weighting — the side with MORE volume drags the
   estimator toward the OPPOSITE side.

2. **VAMP — Volume-Adjusted Mid-Price**: walks the book to a target
   notional and returns the volume-weighted average price of the
   shares that would be filled. Useful for "where is the market for
   $X notional?" — a more honest fair-value when the touch is thin.

3. **Queue imbalance** (Gould & Bonart 2015): bid_vol / (bid_vol +
   ask_vol). A documented one-tick-ahead price predictor: high QI
   (heavy bid) ⇒ next tick more likely up.

For Polymarket's [0, 1] price scale, all of these still apply
unchanged — the only adjustment is to clamp results to [0.01, 0.99]
before feeding them to downstream models that assume valid
probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


def _top(book_side: dict[float, float], reverse: bool) -> tuple[float, float] | None:
    """Return (best_price, size_at_best) or None if empty.
    `reverse=True` for bids (sort high→low), False for asks (low→high).
    """
    if not book_side:
        return None
    prices = sorted(book_side.keys(), reverse=reverse)
    p = prices[0]
    s = float(book_side.get(p, 0.0))
    if s <= 0:
        return None
    return float(p), s


@dataclass
class MicrostructureFeatures:
    """Container for the three documented microstructure features."""
    mid: float | None
    micro: float | None
    vamp_buy: float | None       # at `vamp_notional` for buying YES
    vamp_sell: float | None      # at `vamp_notional` for selling YES
    queue_imbalance: float | None  # bid_vol / (bid_vol + ask_vol)
    spread: float | None
    bid_levels: int
    ask_levels: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def micro_price(book) -> float | None:
    """Inverse-volume-weighted micro-price (Stoikov 2017)."""
    bid = _top(getattr(book, "bids", {}), reverse=True)
    ask = _top(getattr(book, "asks", {}), reverse=False)
    if bid is None or ask is None:
        return None
    bp, bv = bid
    ap, av = ask
    denom = bv + av
    if denom <= 0:
        return None
    micro = (av * bp + bv * ap) / denom
    return float(max(0.0, min(1.0, micro)))


def queue_imbalance(book) -> float | None:
    """Top-of-book queue imbalance (Gould & Bonart 2015).
    bid_vol / (bid_vol + ask_vol) in [0, 1]. > 0.5 ⇒ next tick more
    likely up."""
    bid = _top(getattr(book, "bids", {}), reverse=True)
    ask = _top(getattr(book, "asks", {}), reverse=False)
    if bid is None or ask is None:
        return None
    _, bv = bid
    _, av = ask
    denom = bv + av
    if denom <= 0:
        return None
    return float(bv / denom)


def vamp(book, side: str, target_notional: float) -> float | None:
    """Volume-Adjusted Mid-Price: walk the book up to `target_notional`
    USDC on `side` ('BUY' or 'SELL') and return the size-weighted
    average price.

    'BUY' walks the asks (lowest first); 'SELL' walks the bids
    (highest first).
    """
    side_norm = side.upper().strip()
    if side_norm == "BUY":
        levels = getattr(book, "asks", {})
        reverse = False
    elif side_norm == "SELL":
        levels = getattr(book, "bids", {})
        reverse = True
    else:
        return None
    if not levels or target_notional <= 0:
        return None
    prices = sorted(levels.keys(), reverse=reverse)
    remaining = float(target_notional)
    cost = 0.0
    shares = 0.0
    for p in prices:
        sz = float(levels.get(p, 0.0))
        if sz <= 0:
            continue
        notional_here = p * sz
        if notional_here >= remaining:
            # partial-fill this level
            sz_take = remaining / p
            cost += sz_take * p
            shares += sz_take
            remaining = 0.0
            break
        cost += notional_here
        shares += sz
        remaining -= notional_here
    if shares <= 0:
        return None
    avg_price = cost / shares
    return float(max(0.0, min(1.0, avg_price)))


def compute_features(
    book,
    *,
    vamp_notional: float = 500.0,
) -> MicrostructureFeatures:
    """Compute all microstructure features for a given book.

    Args:
        book: object with `.bids` and `.asks` dicts {price: size}.
        vamp_notional: USDC notional to walk the book for VAMP.

    Returns:
        MicrostructureFeatures with possibly-None entries when
        either side is empty.
    """
    bids = getattr(book, "bids", {}) or {}
    asks = getattr(book, "asks", {}) or {}
    bid = _top(bids, reverse=True)
    ask = _top(asks, reverse=False)
    if bid is not None and ask is not None:
        mid = (bid[0] + ask[0]) / 2.0
        spread = ask[0] - bid[0]
    else:
        mid = None
        spread = None
    return MicrostructureFeatures(
        mid=float(mid) if mid is not None else None,
        micro=micro_price(book),
        vamp_buy=vamp(book, "BUY", vamp_notional),
        vamp_sell=vamp(book, "SELL", vamp_notional),
        queue_imbalance=queue_imbalance(book),
        spread=float(spread) if spread is not None else None,
        bid_levels=len(bids),
        ask_levels=len(asks),
    )
