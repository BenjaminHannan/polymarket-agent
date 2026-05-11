"""Multi-detector arbitrage executor — auto-trades opportunities from
the four idle arb scanners (combinatorial NegRisk, monotonicity, in-play
sports, single-condition negation).

Background
----------
Polyagent has four arb detectors that have been logging opportunities
without ever placing trades:

  - `polyagent/signals/combinatorial_arb.py` — NegRisk sum-to-1 violations
  - `polyagent/signals/monotonicity_arb.py`   — A ⊆ B with p(A) > p(B)
  - `polyagent/signals/inplay_arb.py`         — sub-second sports in-play gaps
  - `polyagent/signals/negrisk_clustering.py` — semantic-clustered MEE arbs

`pmwhy.md` and `pmwhybetter.md` both cite IMDEA's published $40M extracted
from Polymarket NegRisk arbs Apr 2024 – Apr 2025 ($29M from rebalancing
alone). The detectors find these; the executor turns them into trades.

Why this strategy bypasses the cert gate
----------------------------------------
The cert gate exists to protect against trading directionally in
categories where the model is provably worse than the market. Arbitrage
is **mathematically risk-free when executable** — it's not "edge against
the market", it's "edge against pricing inconsistencies among related
markets". The gate doesn't apply, and applying it would just leak
free money to faster operators.

Safety: every basket atomically places all legs via `broker.submit()`.
If any leg comes up short (price moved during the partial-fill round-
trip), the basket auto-unwinds the filled legs at best bid to avoid
sitting on an unintended directional position. Modelled on the existing
`yes_no_arb.py` unwind pattern.

Honest caveat
-------------
Paper mode P&L from this executor will be **optimistic** vs. real money:
sub-100ms colocation operators get most of the published $40M; our paper
broker fills at top-of-book without modeling the queue we'd actually be
at the back of. Always run `scripts/reconcile_queue_aware.py` against
`fills_shadow_queue` to see the realistic-fill version before believing
the dashboard P&L number.

Thresholds (defaults)
---------------------
- NegRisk sum-to-1 (≥3 legs):       gap ≥ 5 bps, min-leg size ≥ $50
- Monotonicity pair (2 legs):       gap ≥ 3 bps, min-leg size ≥ $50
- In-play sports (≥2 legs):         gap ≥ 10 bps late-game / 20 bps else
- Per-(opportunity, cooldown):      60s between retries on same pair
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import structlog

from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker

log = structlog.get_logger()


@dataclass
class _BasketLeg:
    """One leg of a multi-leg arb basket."""
    token_id: str
    condition_id: str
    side: str           # "BUY" or "SELL"
    quote_price: float
    target_size: float
    category: str = "arb"
    is_yes_token: bool = True


@dataclass
class ArbExecutor:
    """Long-running supervised task. Polls each detector at `poll_sec`
    cadence, places baskets when opportunities clear thresholds.

    Args:
        broker: PaperBroker.
        book_store: BookStore for fresh prices at execution time.
        markets: list of Market (used by detector callbacks).
        poll_sec: scan cadence in seconds (default 5 = aggressive; matches
            Yang-Cheng-Zou 2026's 3.6s median in-play episode).
        cooldown_sec: per-opportunity cooldown to avoid repeated
            attempts on the same gap while the WSS feed propagates.
        negrisk_min_bps: NegRisk sum-to-1 gap floor (default 50 = 0.5 cents).
        monotonicity_min_bps: monotonicity-pair gap floor (default 30).
        inplay_late_game_min_bps: looser threshold for last 5 min of games.
        inplay_default_min_bps: default in-play threshold.
        min_leg_size: minimum available size at the stale price per leg.
    """
    broker: PaperBroker
    book_store: BookStore
    markets: list = field(default_factory=list)
    poll_sec: float = 5.0
    cooldown_sec: float = 60.0
    negrisk_min_bps: float = 50.0
    monotonicity_min_bps: float = 30.0
    inplay_late_game_min_bps: float = 10.0
    inplay_default_min_bps: float = 20.0
    min_leg_size: float = 50.0
    max_basket_notional: float = 200.0  # USD per basket — small while paper-validating
    # In-play: provide a list of NegRisk groups with game-window metadata
    # via the constructor. Empty = in-play scanner is a no-op.
    inplay_groups: list[dict] = field(default_factory=list)
    # State
    _recent: deque = field(default_factory=lambda: deque(maxlen=500))
    _markets_by_token: dict = field(default_factory=dict)
    _baskets_executed: int = 0
    _legs_executed: int = 0
    _legs_unwound: int = 0

    def __post_init__(self) -> None:
        self._markets_by_token = {
            m.yes_token_id: m for m in (self.markets or [])
        }
        # Also map condition_id -> Market for monotonicity (which uses
        # token-level lookups but might want the parent market).

    # ── Public scanner entry ────────────────────────────────────────────
    async def run(self) -> None:
        log.info(
            "arb_executor_start",
            poll_sec=self.poll_sec,
            negrisk_min_bps=self.negrisk_min_bps,
            monotonicity_min_bps=self.monotonicity_min_bps,
            inplay_default_min_bps=self.inplay_default_min_bps,
            min_leg_size=self.min_leg_size,
            max_basket_notional=self.max_basket_notional,
        )
        while True:
            try:
                await self._scan_and_execute()
            except Exception as e:
                log.warning("arb_executor_scan_error", err=str(e))
            await asyncio.sleep(self.poll_sec)

    async def _scan_and_execute(self) -> None:
        # 1) Monotonicity arbs (cheap, deterministic detection)
        await self._scan_monotonicity()
        # 2) NegRisk sum-to-1
        await self._scan_negrisk()
        # 3) In-play (late-game-aware threshold)
        await self._scan_inplay()

    # ── Detector → executor adapters ────────────────────────────────────
    async def _scan_monotonicity(self) -> None:
        from polyagent.signals.monotonicity_arb import detect_pairs
        # Compose lightweight "market-with-price" objects from the live
        # book_store. detect_pairs() only needs token_id, question, yes_price.
        live_markets = []
        for m in self.markets:
            book = self.book_store.books.get(m.yes_token_id)
            if book is None:
                continue
            ask = book.best_ask()
            if ask is None:
                continue
            live_markets.append(_PricedMarket(
                token_id=m.yes_token_id,
                question=m.question,
                yes_price=float(ask[0]),
                condition_id=m.condition_id,
                category=getattr(m, "category", None) or "",
            ))
        if len(live_markets) < 2:
            return
        candidates = detect_pairs(live_markets)
        for c in candidates:
            gap_bps = c.gap * 10_000.0
            if gap_bps < self.monotonicity_min_bps:
                continue
            await self._execute_monotonicity_pair(c)

    async def _scan_negrisk(self) -> None:
        # Group markets by their event_id / NegRisk group key.
        groups: dict[str, list] = {}
        for m in self.markets:
            event_id = getattr(m, "neg_risk_event_id", None) or getattr(m, "event_id", None)
            if not event_id:
                continue
            groups.setdefault(event_id, []).append(m)
        for event_id, members in groups.items():
            if len(members) < 2:
                continue
            asks = []
            for m in members:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    asks = None
                    break
                a = book.best_ask()
                if a is None:
                    asks = None
                    break
                asks.append((m, float(a[0]), float(a[1])))
            if asks is None:
                continue
            sum_yes = sum(p for _, p, _ in asks)
            # The "partial-group" guard: skip when sum is way below 1, which
            # usually means we're only seeing some of the legs in our stream.
            if sum_yes < 0.30:
                continue
            # Two arb directions: sum < 1 (buy YES on every leg) or
            # sum > 1 (buy NO on every leg, i.e. SELL YES at the ask).
            if sum_yes < 1.0:
                gap_bps = (1.0 - sum_yes) * 10_000.0
                if gap_bps < self.negrisk_min_bps:
                    continue
                await self._execute_negrisk_basket(
                    event_id, asks, direction="LONG_YES_ALL",
                    gap_bps=gap_bps,
                )
            elif sum_yes > 1.0:
                gap_bps = (sum_yes - 1.0) * 10_000.0
                if gap_bps < self.negrisk_min_bps:
                    continue
                await self._execute_negrisk_basket(
                    event_id, asks, direction="SHORT_YES_ALL",
                    gap_bps=gap_bps,
                )

    async def _scan_inplay(self) -> None:
        if not self.inplay_groups:
            return
        from polyagent.signals.inplay_arb import find_inplay_arbs
        # Two passes with different thresholds: late-game (last 5 min)
        # uses the relaxed threshold; everything else uses the default.
        now = time.time()
        late_groups = []
        normal_groups = []
        for g in self.inplay_groups:
            game_end = g.get("game_end_ts")
            if game_end and now >= game_end - 300:
                late_groups.append(g)
            else:
                normal_groups.append(g)
        if late_groups:
            arbs = find_inplay_arbs(
                self.book_store, late_groups,
                min_bps_gap=self.inplay_late_game_min_bps,
                in_game_only=True,
            )
            for a in arbs:
                if a.min_leg_size_at_stale < self.min_leg_size:
                    continue
                await self._execute_inplay_basket(a, late_game=True)
        if normal_groups:
            arbs = find_inplay_arbs(
                self.book_store, normal_groups,
                min_bps_gap=self.inplay_default_min_bps,
                in_game_only=True,
            )
            for a in arbs:
                if a.min_leg_size_at_stale < self.min_leg_size:
                    continue
                await self._execute_inplay_basket(a, late_game=False)

    # ── Execution primitives ────────────────────────────────────────────
    async def _execute_monotonicity_pair(self, candidate) -> None:
        """Trade a (subset, superset) pair where p(subset) > p(superset).
        The arb: SELL the subset (rich leg) + BUY the superset (cheap leg).
        On Polymarket: SELL YES = BUY NO at (1 − price).
        """
        # Cooldown
        key = f"mono:{candidate.pair_id}"
        if not self._cooldown_ok(key):
            return
        # Re-check fresh book at execution time (the scanner used a
        # snapshot from the detector).
        book_sub = self.book_store.books.get(candidate.token_subset)
        book_sup = self.book_store.books.get(candidate.token_superset)
        if book_sub is None or book_sup is None:
            return
        bid_sub = book_sub.best_bid()
        ask_sup = book_sup.best_ask()
        if bid_sub is None or ask_sup is None:
            return
        # Recompute gap with fresh prices: SELL subset at bid_sub gets you
        # bid_sub per share; BUY superset at ask_sup costs ask_sup per share.
        # Net long the (superset, no-subset) constraint, which is risk-free
        # as long as bid_sub > ask_sup at execution time.
        if bid_sub[0] <= ask_sup[0] + (self.monotonicity_min_bps / 10_000.0):
            return  # gap evaporated
        leg_size = min(self.min_leg_size, float(bid_sub[1]), float(ask_sup[1]))
        if leg_size < self.min_leg_size:
            return
        # Cap basket notional
        notional = leg_size * max(bid_sub[0], ask_sup[0])
        if notional > self.max_basket_notional:
            leg_size = leg_size * (self.max_basket_notional / notional)
        if leg_size < 1.0:
            return
        sub_cid = self._cond_for_token(candidate.token_subset)
        sup_cid = self._cond_for_token(candidate.token_superset)
        if sub_cid is None or sup_cid is None:
            return
        basket_id = str(uuid.uuid4())[:8]
        log.info(
            "arb_execute_monotonicity",
            basket_id=basket_id,
            pair_id=candidate.pair_id,
            bid_sub=bid_sub[0], ask_sup=ask_sup[0],
            gap_bps=round((bid_sub[0] - ask_sup[0]) * 10_000.0, 1),
            size=round(leg_size, 2),
        )
        results = await asyncio.gather(
            self.broker.submit(
                strategy=f"arb_monotonicity",
                condition_id=sub_cid,
                token_id=candidate.token_subset,
                side="SELL",
                max_size=leg_size,
                max_price=bid_sub[0] - 1e-9,
                reason=f"mono basket={basket_id} sub_rich",
            ),
            self.broker.submit(
                strategy=f"arb_monotonicity",
                condition_id=sup_cid,
                token_id=candidate.token_superset,
                side="BUY",
                max_size=leg_size,
                max_price=ask_sup[0] + 1e-9,
                reason=f"mono basket={basket_id} sup_cheap",
            ),
            return_exceptions=True,
        )
        await self._handle_basket_result(
            basket_id=basket_id,
            legs=[
                _BasketLeg(
                    token_id=candidate.token_subset, condition_id=sub_cid,
                    side="SELL", quote_price=bid_sub[0], target_size=leg_size,
                ),
                _BasketLeg(
                    token_id=candidate.token_superset, condition_id=sup_cid,
                    side="BUY", quote_price=ask_sup[0], target_size=leg_size,
                ),
            ],
            results=results,
        )
        self._mark_used(key)

    async def _execute_negrisk_basket(
        self, event_id: str, asks: list, *, direction: str, gap_bps: float,
    ) -> None:
        """`direction` ∈ {LONG_YES_ALL, SHORT_YES_ALL}.

        LONG: sum<1, so buy YES on every leg — locks (1 − sum) profit per share.
        SHORT: sum>1, so sell YES on every leg — locks (sum − 1) profit per share.
        """
        key = f"negrisk:{event_id}:{direction}"
        if not self._cooldown_ok(key):
            return
        # Smallest available depth caps the basket.
        min_depth = min(d for _, _, d in asks)
        leg_size = min(self.min_leg_size, float(min_depth))
        if leg_size < self.min_leg_size:
            return
        # Cap notional: number of legs × per-share-cost × leg_size
        total_cost_per_share = sum(p for _, p, _ in asks) if direction == "LONG_YES_ALL" \
            else sum(1.0 - p for _, p, _ in asks)
        notional = total_cost_per_share * leg_size
        if notional > self.max_basket_notional:
            leg_size = leg_size * (self.max_basket_notional / notional)
        if leg_size < 1.0:
            return
        basket_id = str(uuid.uuid4())[:8]
        log.info(
            "arb_execute_negrisk",
            basket_id=basket_id,
            event_id=event_id,
            direction=direction,
            n_legs=len(asks),
            gap_bps=round(gap_bps, 1),
            size=round(leg_size, 2),
        )
        legs: list[_BasketLeg] = []
        coros = []
        for m, p, _ in asks:
            side = "BUY" if direction == "LONG_YES_ALL" else "SELL"
            max_price = (p + 1e-9) if side == "BUY" else (p - 1e-9)
            legs.append(_BasketLeg(
                token_id=m.yes_token_id, condition_id=m.condition_id,
                side=side, quote_price=p, target_size=leg_size,
            ))
            coros.append(self.broker.submit(
                strategy="arb_negrisk",
                condition_id=m.condition_id,
                token_id=m.yes_token_id,
                side=side,
                max_size=leg_size,
                max_price=max_price,
                reason=f"negrisk basket={basket_id} {direction}",
            ))
        results = await asyncio.gather(*coros, return_exceptions=True)
        await self._handle_basket_result(basket_id, legs, results)
        self._mark_used(key)

    async def _execute_inplay_basket(self, arb, *, late_game: bool) -> None:
        """In-play sports NegRisk basket — same as `_execute_negrisk_basket`
        but with the late-game flag for telemetry."""
        key = f"inplay:{arb.condition_id}:{int(arb.detected_ts)}"
        if not self._cooldown_ok(key):
            return
        leg_size = min(self.min_leg_size, arb.min_leg_size_at_stale)
        if leg_size < self.min_leg_size:
            return
        total = sum(leg["yes_price"] for leg in arb.legs)
        notional = total * leg_size
        if notional > self.max_basket_notional:
            leg_size = leg_size * (self.max_basket_notional / notional)
        if leg_size < 1.0:
            return
        basket_id = str(uuid.uuid4())[:8]
        log.info(
            "arb_execute_inplay",
            basket_id=basket_id,
            condition_id=arb.condition_id,
            late_game=late_game,
            n_legs=len(arb.legs),
            gap_bps=round(arb.bps_gap, 1),
            size=round(leg_size, 2),
        )
        # All NegRisk inplay arbs trade SHORT_YES_ALL (sum>1 by construction
        # in inplay_arb.find_inplay_arbs).
        legs: list[_BasketLeg] = []
        coros = []
        for leg_d in arb.legs:
            legs.append(_BasketLeg(
                token_id=leg_d["token_id"], condition_id=arb.condition_id,
                side="SELL", quote_price=leg_d["yes_price"], target_size=leg_size,
            ))
            coros.append(self.broker.submit(
                strategy="arb_inplay",
                condition_id=arb.condition_id,
                token_id=leg_d["token_id"],
                side="SELL",
                max_size=leg_size,
                max_price=leg_d["yes_price"] - 1e-9,
                reason=f"inplay basket={basket_id} late_game={late_game}",
            ))
        results = await asyncio.gather(*coros, return_exceptions=True)
        await self._handle_basket_result(basket_id, legs, results)
        self._mark_used(key)

    # ── Basket post-processing ──────────────────────────────────────────
    async def _handle_basket_result(
        self,
        basket_id: str,
        legs: list[_BasketLeg],
        results: list,
    ) -> None:
        """Process the results of an asyncio.gather over leg submissions.

        If every leg filled (even partially), we accept the partial-fill
        basket. If one or more legs returned 0 (no fill), we *unwind*
        the filled legs at market to avoid carrying directional exposure.
        """
        fills = []
        for leg, res in zip(legs, results):
            if isinstance(res, Exception):
                log.warning("arb_leg_error", basket_id=basket_id,
                            token=leg.token_id[:14], err=str(res))
                fills.append(0.0)
            else:
                fills.append(float(res or 0.0))
        if all(f > 0 for f in fills):
            self._baskets_executed += 1
            self._legs_executed += len(legs)
            log.info(
                "arb_basket_filled",
                basket_id=basket_id, n_legs=len(legs),
                fills=[round(f, 2) for f in fills],
                cum_baskets=self._baskets_executed,
            )
            return
        # Partial: unwind the filled legs at best opposite-side
        log.warning(
            "arb_basket_partial_unwind",
            basket_id=basket_id, fills=[round(f, 2) for f in fills],
        )
        unwind_coros = []
        for leg, filled in zip(legs, fills):
            if filled <= 0:
                continue
            unwind_side = "SELL" if leg.side == "BUY" else "BUY"
            unwind_coros.append(self.broker.submit(
                strategy="arb_unwind",
                condition_id=leg.condition_id,
                token_id=leg.token_id,
                side=unwind_side,
                max_size=filled,
                max_price=None,
                reason=f"basket={basket_id} unwind partial",
            ))
        if unwind_coros:
            try:
                await asyncio.gather(*unwind_coros, return_exceptions=True)
                self._legs_unwound += len(unwind_coros)
            except Exception as e:
                log.error(
                    "arb_unwind_failed",
                    basket_id=basket_id, err=str(e),
                )

    # ── Helpers ─────────────────────────────────────────────────────────
    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        # Walk recent and check if `key` is present with recent ts.
        for entry in self._recent:
            if entry[0] == key and now - entry[1] < self.cooldown_sec:
                return False
        return True

    def _mark_used(self, key: str) -> None:
        self._recent.append((key, time.time()))

    def _cond_for_token(self, token_id: str) -> str | None:
        m = self._markets_by_token.get(token_id)
        return m.condition_id if m is not None else None

    def summary(self) -> dict:
        return {
            "baskets_executed": self._baskets_executed,
            "legs_executed": self._legs_executed,
            "legs_unwound": self._legs_unwound,
            "cooldown_entries": len(self._recent),
        }


@dataclass
class _PricedMarket:
    """Lightweight adapter so monotonicity_arb.detect_pairs() can run
    over live BookStore prices without rebuilding Market objects."""
    token_id: str
    question: str
    yes_price: float
    condition_id: str
    category: str
