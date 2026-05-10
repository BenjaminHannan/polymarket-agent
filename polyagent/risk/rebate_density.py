"""Per-market quote-density rebate model (refinement on risk/fees.py).

The simple model in risk/fees.py credits 22% of the would-be taker fee
on every maker fill. Polymarket's real program is more nuanced: the
daily rebate pool on a market is bounded by the total taker fees on
that market, and is distributed across makers proportionally to their
quote density (size × quote-uptime, weighted toward inside-spread
posts).

For paper-mode we can't observe other makers' density, so this module
models our SHARE of the available rebate pool as:

  our_share = our_quote_time_size / max(visible_book_top_size, our_size)

which gives us close to 100% credit when our size dominates the inside
spread (we're the dominant maker) and a small fraction when the book
is deep with other makers.

The simple 22% × notional model in risk/fees.py is the *upper bound*
on what we could earn. This module gives a *more honest paper-P&L
estimate* by scaling that upper bound by our visible market share.

Schema: a per-fill column on fills_shadow_queue is added so we can
A/B compare the simple vs density-adjusted rebate stream. The
broker continues to use the simple model for the cash-flow side
(it's the upper bound, makes paper P&L generous in a known way);
this module produces the *honest* number for cert validation.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


def ensure_columns(conn: sqlite3.Connection) -> None:
    """Add density-adjusted rebate column to fills_shadow_queue if absent."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fills_shadow_queue)").fetchall()}
    if "rebate_density_adjusted" not in cols:
        conn.execute(
            "ALTER TABLE fills_shadow_queue ADD COLUMN rebate_density_adjusted REAL DEFAULT 0"
        )
        conn.commit()


@dataclass
class DensityAdjustedRebate:
    upper_bound: float       # the simple 22% × notional × fee_rate value
    market_share: float      # our_size / (our_size + other_visible_size)
    adjusted_rebate: float   # upper_bound × market_share
    other_visible_size: float
    our_size: float


def compute_density_share(
    *,
    our_quote_size: float,
    visible_book_size_at_or_better: float,
) -> DensityAdjustedRebate | None:
    """Estimate our share of the maker pool on this market at fill time.

    `visible_book_size_at_or_better` is the total displayed size at our
    quote price level OR better (i.e. the queue ahead of us plus the
    rest of the displayed maker liquidity competing for the same flow).
    Our SHARE is our_size / (our_size + others), where `others` is the
    visible-book-size minus our own contribution.

    Real Polymarket rebate distribution is complex (time-weighted,
    inside-spread bonuses, daily aggregation). This is a paper-mode
    upper-bound estimator; the cert-validator can use it instead of
    the optimistic 22%-flat to honestly bound expected rebate income.
    """
    if our_quote_size <= 0:
        return None
    others = max(0.0, visible_book_size_at_or_better - our_quote_size)
    total = our_quote_size + others
    market_share = our_quote_size / total if total > 0 else 1.0
    # The upper-bound is the existing 22% × notional × fee_rate, supplied
    # by the caller (we don't recompute fees here)
    return DensityAdjustedRebate(
        upper_bound=0.0,  # filled in by caller
        market_share=market_share,
        adjusted_rebate=0.0,  # filled in by caller after upper_bound is known
        other_visible_size=others,
        our_size=our_quote_size,
    )


def adjust_rebate(
    *,
    upper_bound: float,
    our_quote_size: float,
    visible_book_size_at_or_better: float,
) -> DensityAdjustedRebate:
    """Scale the upper-bound (22% × notional × fee_rate) by our market share."""
    share = compute_density_share(
        our_quote_size=our_quote_size,
        visible_book_size_at_or_better=visible_book_size_at_or_better,
    )
    if share is None:
        return DensityAdjustedRebate(
            upper_bound=upper_bound, market_share=0.0,
            adjusted_rebate=0.0, other_visible_size=0.0, our_size=0.0,
        )
    share.upper_bound = upper_bound
    share.adjusted_rebate = upper_bound * share.market_share
    return share


def density_adjusted_summary(conn: sqlite3.Connection, strategy: str) -> dict:
    """Aggregate the adjusted-vs-upper-bound rebate gap for one strategy."""
    ensure_columns(conn)
    row = conn.execute(
        """SELECT
              COALESCE(SUM(maker_rebate_credited), 0)            AS upper_bound_total,
              COALESCE(SUM(rebate_density_adjusted), 0)          AS adjusted_total,
              COUNT(*)                                           AS n_maker_fills
           FROM fills_shadow_queue q
           JOIN fills f ON f.id = q.fill_id
           WHERE q.is_maker = 1 AND f.strategy = ?""",
        (strategy,),
    ).fetchone()
    return {
        "strategy": strategy,
        "n_maker_fills": int(row[2] or 0),
        "rebate_upper_bound": round(float(row[0]), 4),
        "rebate_density_adjusted": round(float(row[1]), 4),
        "adjustment_haircut": (
            round(1.0 - (float(row[1]) / float(row[0])), 4)
            if row[0] and float(row[0]) > 0 else None
        ),
    }
