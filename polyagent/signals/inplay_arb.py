"""In-play (live) sports arb detector (pmwhybetter.md Problem-3 #2,
Problem-6 #2; Yang/Cheng/Zou 2026, SSRN 6624718).

Direct implementation of the doc's Yang/Cheng/Zou 2026 NBA finding:
*in-game arbitrage opportunities occur with median episode duration
of 3.6 seconds, concentrated in the final minutes of games.*

Their methodology: real-time order-book-snapshot detection (not
executed-trade reconstruction). For a live sports event with
mutually-exclusive outcomes (team A wins, team B wins, draw), the
sum of YES prices across the three outcomes should equal 1.0. When
the sum drifts to (1 + ε), an arb exists: sell each leg at its YES
price, collect ε × notional per round-trip when one outcome resolves.

Capital constraint
------------------
The smallest leg's available size at the stale price caps the
executable size. Even a 5% gap may be only $200 deep — the doc
explicitly flags this as the realistic constraint.

Latency floor
-------------
Yang-Cheng-Zou measured 3.6-second median episode duration. Our paper
broker has *no* latency model on the matching side; the queue-aware
shadow flag this through `cancel_latency.py` (Brownian σ√Δt + 2s
Polygon blocks). An arb visible to us for ≥3.6s on the WSS feed is,
by the time we'd actually execute on-chain, ~2× as wide as it appears.

Polymarket structure
--------------------
Most sports markets on Polymarket use the NegRisk pattern (3+
exclusive outcomes share a condition_id). This module is a thin
specialisation of `negrisk_clustering.py` that:

  1. Polls the BookStore on every tick.
  2. Computes sum-of-YES per known NegRisk group.
  3. Filters on **in-game-window** flag (game start ≤ now < game end).
  4. Emits an `InPlayArb` candidate with the leg-by-leg trade and
     the min-leg-size executability bound.

API
---
- `find_inplay_arbs(book_store, negrisk_groups)` →
    list[InPlayArb]
- `InPlayArb.size_capped` — executable size after min-leg constraint
- `InPlayArb.bps_gap` — gap in basis points
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class InPlayArb:
    """An in-play arb opportunity across N legs of a NegRisk group."""
    condition_id: str
    legs: list[dict]               # [{token_id, yes_price, ask_size}, ...]
    sum_yes_prices: float
    arb_gap: float                 # sum − 1.0 (positive = arb)
    bps_gap: float                 # arb_gap × 10000
    min_leg_size_at_stale: float   # executability cap
    detected_ts: float
    detected_during_game: bool     # in-game window flag (T/F)

    @property
    def size_capped(self) -> float:
        return self.min_leg_size_at_stale

    @property
    def captured_pnl_estimate(self) -> float:
        """Per-USDC arb yield × min-leg size. Ignores fees + slippage."""
        return self.arb_gap * self.min_leg_size_at_stale


def _is_in_game_window(
    game_start_ts: float | None, game_end_ts: float | None,
) -> bool:
    """True iff `now` is inside the game window. Defaults to False
    when either bound is missing (treat as "pre-market" not in-play)."""
    now = time.time()
    if game_start_ts is None or game_end_ts is None:
        return False
    return float(game_start_ts) <= now <= float(game_end_ts)


def find_inplay_arbs(
    book_store,
    negrisk_groups: list[dict],
    *,
    min_bps_gap: float = 20.0,
    in_game_only: bool = True,
) -> list[InPlayArb]:
    """Scan the BookStore for active in-play arbs across NegRisk groups.

    Args:
        book_store: polyagent.orderbook.BookStore with live books.
        negrisk_groups: list of {condition_id, leg_token_ids[],
                                  game_start_ts, game_end_ts}.
        min_bps_gap: minimum arb gap in bps to surface (default 20 =
            0.20% gap, which is small but covers fees only in a
            best-case real-money scenario).
        in_game_only: if True (default), only surface arbs detected
            during the active game window. Set False to catch
            pre-market arbs (which last longer but are less specific
            to the Yang-Cheng-Zou finding).
    """
    out: list[InPlayArb] = []
    for group in negrisk_groups:
        cid = group.get("condition_id")
        tokens = group.get("leg_token_ids", []) or []
        if not cid or len(tokens) < 2:
            continue
        in_window = _is_in_game_window(
            group.get("game_start_ts"), group.get("game_end_ts"),
        )
        if in_game_only and not in_window:
            continue

        legs = []
        any_missing = False
        for tok in tokens:
            book = book_store.books.get(tok)
            if book is None:
                any_missing = True
                break
            ask = book.best_ask()
            if ask is None:
                any_missing = True
                break
            legs.append({
                "token_id": tok,
                "yes_price": float(ask[0]),
                "ask_size": float(ask[1]),
            })
        if any_missing:
            continue

        sum_yes = sum(leg["yes_price"] for leg in legs)
        gap = sum_yes - 1.0
        bps = gap * 10_000.0
        if bps < min_bps_gap:
            continue
        # Executability: smallest available size at the quoted price.
        min_size = min(leg["ask_size"] for leg in legs)
        if min_size <= 0:
            continue

        arb = InPlayArb(
            condition_id=cid,
            legs=legs,
            sum_yes_prices=sum_yes,
            arb_gap=float(gap),
            bps_gap=float(bps),
            min_leg_size_at_stale=float(min_size),
            detected_ts=time.time(),
            detected_during_game=in_window,
        )
        out.append(arb)
    if out:
        log.info(
            "inplay_arb_candidates",
            n=len(out),
            top_bps=round(max(a.bps_gap for a in out), 1),
            top_size=round(max(a.size_capped for a in out), 1),
        )
    return out
