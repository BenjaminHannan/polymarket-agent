"""Queue-aware fill probability + cancel-slippage modeling.

Cont/Kukanov/Stoikov 2014 + Gould/Bonart 2016: probability that a passive
limit order at the top-of-book fills before being canceled or invalidated
by a mid-price move depends on:

  - queue position (how much size is ahead of you at the same level)
  - top-of-book imbalance (predictive of next mid-price tick direction)
  - micro-volatility (faster vol = higher cancel risk)

This module exposes two functions:

  fill_prob_top_of_book(queue_ahead, opp_size, imbalance) -> p
  pessimistic_fill_price(book, side, size, *, queue_loss_bps) -> price

The first is used at signal time to estimate whether a posted passive
limit will actually fill. The second is used by the broker to compute a
worst-case execution price for the shadow ledger.

These are simplified models. The literature has more sophisticated
formulations (semi-Markov queue models, intensity-based diffusions); we
use closed-form approximations that capture the dominant behavior.
"""

from __future__ import annotations

import math


def fill_prob_top_of_book(
    queue_ahead: float,
    opp_size: float,
    imbalance: float,
    *,
    queue_density: float = 1.0,
) -> float:
    """Estimate probability that a top-of-book passive limit fills.

    Heuristic — Gould-Bonart-style:
      base_prob = opp_size / (queue_ahead + opp_size)
      adjusted by imbalance: bid-heavy book increases bid fill probability,
      ask-heavy decreases it (because the next move is more likely to lift
      the ask, leaving us stranded).

    `imbalance` is bid_share ∈ [0, 1]; >0.5 means bid-heavy.
    `queue_density` represents how compact the queue is — higher means
    more cancellations / replenishment, faster turnover.
    """
    if queue_ahead < 0:
        queue_ahead = 0.0
    base = opp_size / (queue_ahead + opp_size + 1e-6)
    # Imbalance correction: deviation from 0.5, scaled
    imb_bias = (imbalance - 0.5) * 0.6
    p = base + imb_bias * (1 - base)
    return max(0.01, min(0.99, p * queue_density / max(1.0, queue_density)))


# Polygon mainnet ~2s block time. Cancels can't land before next block, so a
# resting limit can't dodge adverse mid moves faster than this. Modeled as
# extra slippage when realized vol implies the mid likely moved during the
# block window.
POLYGON_BLOCK_SEC = 2.0


def cancel_latency_slippage(
    realized_vol: float | None,
    block_sec: float = POLYGON_BLOCK_SEC,
) -> float:
    """Expected adverse mid drift during the cancel-latency window.

    Brownian approximation: σ × √Δt. If realized_vol is None we use a small
    default. Returns dollars-per-share of expected adverse drift.
    """
    if realized_vol is None or realized_vol <= 0:
        # Default: 0.5pp/s realized vol, modest assumption
        sigma = 0.005
    else:
        sigma = realized_vol
    import math
    return sigma * math.sqrt(block_sec)


def pessimistic_fill_price(
    side: str,
    best_bid: float | None,
    best_ask: float | None,
    book_size_consumed: float,
    book_depth_at_top: float,
    *,
    queue_loss_bps: float = 50.0,
    realized_vol: float | None = None,
    block_sec: float = POLYGON_BLOCK_SEC,
) -> float | None:
    """Compute a pessimistic effective fill price.

    Models:
      - half-spread crossed
      - queue-position loss proportional to consumed depth
      - book-walk slippage
      - cancel latency: 1 Polygon block of adverse drift

    The cancel-latency component is what the doc highlighted: a resting
    limit can't dodge adverse moves faster than the next block.
    """
    if best_bid is None or best_ask is None:
        return None
    spread = best_ask - best_bid
    half_spread = spread / 2.0
    queue_factor = 1.0 + (queue_loss_bps / 10000.0)
    if book_depth_at_top <= 0:
        slip = half_spread
    else:
        slip = half_spread * min(1.0, book_size_consumed / book_depth_at_top)
    cancel_slip = cancel_latency_slippage(realized_vol, block_sec)
    if side == "BUY":
        return min(1.0, best_ask + half_spread * (queue_factor - 1.0) + slip + cancel_slip)
    else:
        return max(0.0, best_bid - half_spread * (queue_factor - 1.0) - slip - cancel_slip)
