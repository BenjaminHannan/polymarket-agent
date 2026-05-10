"""Passive maker — two-sided Avellaneda-Stoikov quoting, paper-mode.

Distinct from polyagent/strategies/passive_poster.py (which is signal-driven:
only posts when the combined-signal model gives it a directional candidate).
This version runs as a standalone two-sided market maker — quotes a bid AND
an ask on every targeted token simultaneously, with inventory-aware skew so
neither side runs away.

Architecture per the senior-quant review (pmwhy.md §B1, §B2):
  - Avellaneda-Stoikov 2008 used as a SIZING / SKEW framework, NOT a pricing
    model. The Polymarket discrete tick + binary settlement violates AS's
    Brownian-mid assumption, so we use AS to set how WIDE we quote and how
    MUCH we lean (inventory skew), not the absolute prices.
  - simulate_passive_fill() from polyagent.risk.queue_aware_fills supplies
    the per-cycle fill probability for each posted side (Cont-Kukanov-Stoikov
    queue-position model with imbalance fallback).
  - Targeted at LOW-VOLATILITY markets only — adverse selection on high-vol
    markets eats spread capture (Bartlett & O'Hara 2026).
  - Paper-mode: a "fill" is a Bernoulli draw per cycle weighted by fill_prob;
    on virtual fills we route through broker.submit() with max_price set to
    our quote so the recorded fill price matches the maker quote.
  - Tracks cumulative spread captured + estimated maker rebate (Polymarket
    rebate program: 20-25% of taker fees, daily USDC, per market).

Default OFF behind ENABLE_PASSIVE_POSTER_V2. Gated by the same
strategy_certificates / category allowlist as combined_trader.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from polyagent.gamma import Market
from polyagent.orderbook import BookStore, OrderBook
from polyagent.paper_broker import PaperBroker
from polyagent.risk.queue_aware_fills import simulate_passive_fill

log = structlog.get_logger()


# ── Avellaneda-Stoikov 2008 closed-form helpers ────────────────────────
# r = s - q × γ × σ² × (T-t)              -- reservation price
# δ_half = γ × σ² × (T-t) / 2 + (1/γ)·ln(1 + γ/k)   -- half-spread
# bid = r - δ_half, ask = r + δ_half
#
# Notation:
#   s     = current mid price
#   q     = inventory (positive = long)
#   γ     = risk aversion (we set this small; large γ → wider quotes)
#   σ²    = realized variance per unit time
#   T-t   = remaining time horizon (we cap at ~600s of "trading window")
#   k     = order arrival intensity (taker arrivals per second crossing
#           top of book) — estimated from book churn

def avellaneda_stoikov_skew(
    inventory: float, gamma: float, sigma_sq_per_sec: float, time_to_horizon_sec: float
) -> float:
    """Reservation-price OFFSET from mid, in $/share. Positive when long
    (we lean down to attract bids and reduce inventory)."""
    return -inventory * gamma * sigma_sq_per_sec * max(0.0, time_to_horizon_sec)


def avellaneda_stoikov_half_spread(
    gamma: float,
    sigma_sq_per_sec: float,
    time_to_horizon_sec: float,
    k_arrival_per_sec: float,
    *,
    min_half_spread: float = 0.005,
    max_half_spread: float = 0.05,
) -> float:
    """Half-spread in $/share, **clamped for prediction-market scale**.

    The vanilla Avellaneda-Stoikov formula assumes price/volatility units
    that don't translate to a 0–1 prediction market — its intensity term
    `(1/γ)·ln(1 + γ/k)` is unbounded in the natural log and produces
    multi-dollar half-spreads on tiny γ. We bound to a tick floor and a
    half-of-maximum-spread ceiling so quotes always sit inside a plausible
    spread on a 0.01-tick venue.
    """
    risk_term = gamma * sigma_sq_per_sec * max(0.0, time_to_horizon_sec) / 2.0
    if k_arrival_per_sec <= 0 or gamma <= 0:
        raw = risk_term + min_half_spread
    else:
        intensity_term = (1.0 / gamma) * math.log1p(gamma / k_arrival_per_sec)
        raw = float(risk_term + intensity_term)
    return max(min_half_spread, min(max_half_spread, raw))


@dataclass
class MakerQuote:
    """A two-sided quote currently posted on ONE token (either YES or NO).

    The maker keeps independent quotes for the YES and NO tokens of each
    market. They are not redundant: Polymarket's outcome tokens trade
    in separate books with their own taker flow, so a fill on YES at
    0.40 BUY does NOT imply a fill on NO at 0.60 SELL.
    """
    market: Market
    token_id: str           # the specific outcome token this quote covers
    is_yes_token: bool      # True for YES, False for NO (used for inventory naming)
    bid_price: float
    ask_price: float
    quote_size: float
    posted_ts: float = field(default_factory=time.time)
    inventory: float = 0.0  # long position on THIS token
    cumulative_spread_captured: float = 0.0
    bid_fills: int = 0
    ask_fills: int = 0
    # Adverse-selection tracking (last N (was_buy, mid_at_fill, mid_after) tuples)
    recent_outcomes: list[tuple[bool, float, float]] = field(default_factory=list)
    # Counter for the quote-replacement protocol (each cancel+repost increments)
    revision: int = 0
    # Mid at the moment we posted — used for stale-quote detection.
    mid_at_post: float = 0.0
    # Inventory-unwind state: True when inventory exceeds unwind_threshold
    # and we've suppressed the inventory-adding side.
    one_sided_unwind: bool = False
    unwind_side: str | None = None  # "BUY" (suppress BUY) or "SELL" (suppress SELL)


def _quote_changed(old: MakerQuote | None, new: MakerQuote, *, tol: float = 0.005) -> bool:
    """Whether the desired quote differs enough from the active one to
    justify a cancel + new post. Tolerance prevents churn on every cycle."""
    if old is None:
        return True
    if abs(old.bid_price - new.bid_price) > tol:
        return True
    if abs(old.ask_price - new.ask_price) > tol:
        return True
    if abs(old.quote_size - new.quote_size) > tol * 100:
        return True
    return False


@dataclass
class PassivePosterV2:
    book_store: BookStore
    broker: PaperBroker
    markets_by_token: dict[str, Market]
    target_tokens: list[str] = field(default_factory=list)  # the universe to quote on
    # Avellaneda-Stoikov knobs
    gamma: float = 0.05                 # risk aversion (small — we want tight quotes)
    horizon_sec: float = 600.0          # remaining "trading window" assumption
    # Quote sizing
    quote_size: float = 25.0            # shares per side per quote
    max_total_inventory_yes: float = 200.0
    # Targeting
    max_realized_vol: float = 0.02      # skip markets above this realized vol
    min_book_depth: float = 50.0
    max_spread_to_quote: float = 0.06
    # Cycle
    poll_sec: float = 30.0
    fill_horizon_sec: float = 30.0      # passive fill prob computed over this
    cooldown_sec: float = 60.0
    # Maker rebate assumption (Polymarket: 20-25% of taker fees per market)
    rebate_share_of_fee: float = 0.22
    # Inventory-unwind threshold (fraction of max_total_inventory_yes).
    # Above this, we suppress the inventory-adding side and tighten the
    # reducing side until inventory comes back below the threshold.
    unwind_threshold_pct: float = 0.6
    # Stale-quote cancel: if mid moves more than this many ticks since
    # post, force a cancel + recompute even if cooldown hasn't elapsed.
    stale_mid_ticks: float = 2.0
    # Cert allowlist (built from strategy_certificates)
    certified_categories: set[str] | None = None
    # Optional VPIN toxicity gate (Bartlett-O'Hara 2026 / Barzykin
    # arXiv 2508.20225). When set, we skip posting a side when flow
    # is toxically one-sided against it. The gate is supplied by main.py
    # and shared with BookStore which feeds it taker-initiated trades.
    vpin_gate: object | None = None
    # Optional sqlite handle for wash-graph suppression lookups. When
    # set, the per-token wash_share suppresses our quote size linearly:
    # size_effective = quote_size × (1 − wash_share).
    wash_graph_conn: object | None = None
    # Optional Polymarket maker-rewards tracker (pmwhybetter.md Problem-10
    # #3; $12M/yr distributed via quadratic-spread formula). When set,
    # each quote update samples its own (spread_bps, size, time) into
    # the tracker so the dashboard can show what we'd be earning if we
    # were a live MM seat.
    maker_rewards: object | None = None
    # Strategy name for logs / fills.strategy
    strategy_name: str = "passive_poster_v2"
    # Per-token quote state
    _quotes: dict[str, MakerQuote] = field(default_factory=dict)
    _last_cycle_ts: dict[str, float] = field(default_factory=dict)
    # AS calibration: per-token taker arrival rate estimate (Poisson). Updated
    # from observed mid-tick changes + book updates as a proxy for taker flow.
    _k_arrival: dict[str, float] = field(default_factory=dict)
    # Per-token rolling counter of mid-changes since last calibration update
    _midchanges_since: dict[str, list[float]] = field(default_factory=dict)
    _last_mid: dict[str, float] = field(default_factory=dict)
    # Realized vol estimate per token (per-second, EWMA)
    _sigma_per_sec: dict[str, float] = field(default_factory=dict)

    def _eligible_market(self, m: Market, book: OrderBook | None) -> tuple[bool, str]:
        """Returns (eligible, reason)."""
        if (
            self.certified_categories is not None
            and (m.category or "") not in self.certified_categories
        ):
            return False, "uncertified_category"
        if book is None or book.last_update_ts is None:
            return False, "no_book"
        if time.time() - book.last_update_ts > 120:
            return False, "stale_book"
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return False, "one_sided_book"
        spread = ba[0] - bb[0]
        if spread > self.max_spread_to_quote:
            return False, "spread_wide"
        if spread <= 0:
            return False, "crossed_book"
        # Volatility check
        rv = book.realized_vol(300) if hasattr(book, "realized_vol") else None
        if rv is not None and rv > self.max_realized_vol:
            return False, "high_volatility"
        # Depth at top
        depth = sum(s for _, s in [bb, ba])
        if depth < self.min_book_depth:
            return False, "thin_book"
        return True, "ok"

    def _compute_quote(
        self, m: Market, book: OrderBook, token_id: str, inventory_this_token: float,
        *, k_arrival_per_sec: float | None = None,
    ) -> MakerQuote | None:
        """Compute a two-sided quote for ONE token (YES or NO).

        Each token has its own book + own inventory. NO is not derived
        from YES — they have independent flow on Polymarket and we quote
        each independently (per pmwhy.md B1).
        """
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return None
        mid = (bb[0] + ba[0]) / 2.0
        rv = book.realized_vol(300) if hasattr(book, "realized_vol") else None
        sigma = rv if (rv and rv > 0) else 0.005
        sigma_sq_per_sec = sigma * sigma
        # Per-token taker arrival rate (calibrated upstream; default 0.5/s).
        k = k_arrival_per_sec if (k_arrival_per_sec and k_arrival_per_sec > 0) else 0.5
        # Per-token adverse-selection penalty: widen θ when recent fills
        # on this token preceded adverse mid moves.
        as_widen = self._adverse_selection_widen(token_id)
        skew = avellaneda_stoikov_skew(
            inventory=inventory_this_token,
            gamma=self.gamma,
            sigma_sq_per_sec=sigma_sq_per_sec,
            time_to_horizon_sec=self.horizon_sec,
        )
        half_spread = avellaneda_stoikov_half_spread(
            gamma=self.gamma,
            sigma_sq_per_sec=sigma_sq_per_sec,
            time_to_horizon_sec=self.horizon_sec,
            k_arrival_per_sec=k,
        ) * as_widen
        reservation = mid + skew
        TICK = 0.01
        # Polymarket prices are bounded to [0.01, 0.99] — hard clamp.
        # A quote at 0.0 or 1.0 would never fill and is a sign that
        # the AS skew + half-spread pushed off-axis (e.g., very deep
        # favorites). When clamps engage, we still post but the quote
        # becomes the boundary tick.
        MIN_PRICE = 0.01
        MAX_PRICE = 0.99
        bid_price = max(MIN_PRICE, min(reservation - half_spread, ba[0] - TICK))
        ask_price = min(MAX_PRICE, max(reservation + half_spread, bb[0] + TICK))
        if ask_price <= bid_price:
            return None
        is_yes = (token_id == m.yes_token_id)
        # Inventory unwind: when |inventory| exceeds unwind_threshold ×
        # max_total_inventory_yes, switch to one-sided mode that
        # suppresses the inventory-adding side and tightens the
        # reducing side. Long → suppress BUY (we'd add YES), tighten SELL.
        # Short → suppress SELL, tighten BUY.
        unwind_thr = self.unwind_threshold_pct * self.max_total_inventory_yes
        one_sided = abs(inventory_this_token) >= unwind_thr
        unwind_side: str | None = None
        if one_sided:
            if inventory_this_token > 0:
                # Long → suppress BUY (post bid at floor that won't fill),
                # tighten SELL by halving the half-spread on that side
                unwind_side = "BUY"
                bid_price = MIN_PRICE  # effectively unfillable
                # Tighten ask: post 1 tick inside the existing best_ask
                ask_price = max(MIN_PRICE + TICK, ba[0] - TICK)
            else:
                unwind_side = "SELL"
                ask_price = MAX_PRICE
                bid_price = min(MAX_PRICE - TICK, bb[0] + TICK)
            if ask_price <= bid_price:
                return None
        return MakerQuote(
            market=m,
            token_id=token_id,
            is_yes_token=is_yes,
            bid_price=round(bid_price, 4),
            ask_price=round(ask_price, 4),
            quote_size=self._effective_quote_size(token_id),
            inventory=inventory_this_token,
            mid_at_post=mid,
            one_sided_unwind=one_sided,
            unwind_side=unwind_side,
        )

    # ── Quote-replacement protocol ─────────────────────────────────────
    # In paper-mode there's no real cancel/post round-trip; in real-money
    # mode each replacement is a CLOB cancel + new post. The protocol
    # here matches what the live broker would do so we can swap broker
    # backends cleanly later: (1) compare desired vs active quote, (2) if
    # changed by more than `tol`, log a "cancel" event for the active
    # quote and a "post" event for the new one, (3) bump revision.
    def _post_or_replace(self, token_id: str, new_q: MakerQuote) -> MakerQuote:
        old = self._quotes.get(token_id)
        if not _quote_changed(old, new_q):
            # Inherit identity + counters from the existing active quote
            new_q.revision = old.revision
            new_q.bid_fills = old.bid_fills
            new_q.ask_fills = old.ask_fills
            new_q.recent_outcomes = old.recent_outcomes
            new_q.cumulative_spread_captured = old.cumulative_spread_captured
            self._quotes[token_id] = new_q
            return new_q
        if old is not None:
            log.info(
                "passive_v2_cancel",
                token=token_id[:14], rev=old.revision,
                old_bid=old.bid_price, old_ask=old.ask_price,
            )
            new_q.revision = old.revision + 1
            new_q.bid_fills = old.bid_fills
            new_q.ask_fills = old.ask_fills
            new_q.recent_outcomes = old.recent_outcomes
            new_q.cumulative_spread_captured = old.cumulative_spread_captured
        else:
            new_q.revision = 0
        log.info(
            "passive_v2_post",
            token=token_id[:14], rev=new_q.revision,
            bid=new_q.bid_price, ask=new_q.ask_price, size=new_q.quote_size,
        )
        self._quotes[token_id] = new_q
        return new_q

    def _update_calibration(self, token_id: str, book: OrderBook) -> None:
        """Update per-token estimates of taker arrival rate (k) and
        realized vol (σ) from observed mid changes. Uses a simple
        rolling-window approach: each cycle we record the mid-change
        magnitude and time delta; k is approximated by the rate of
        non-zero mid changes per second; σ is the EWMA of |Δmid|.

        These are noisy on a single-cycle basis, so we EWMA-smooth.
        """
        cur_mid = book.mid()
        if cur_mid is None:
            return
        prev_mid = self._last_mid.get(token_id)
        self._last_mid[token_id] = cur_mid
        if prev_mid is None:
            return
        d_mid = abs(cur_mid - prev_mid)
        # EWMA on σ_per_sec — assume cycle ≈ poll_sec apart
        prev_sigma = self._sigma_per_sec.get(token_id, 0.005)
        sample_sigma = d_mid / max(self.poll_sec, 1.0)
        new_sigma = 0.9 * prev_sigma + 0.1 * sample_sigma
        self._sigma_per_sec[token_id] = max(1e-5, new_sigma)
        # Maintain a deque of recent mid-change samples; rate of nonzero
        # samples per second approximates k. Keep last 20 samples.
        window = self._midchanges_since.setdefault(token_id, [])
        window.append(d_mid)
        if len(window) > 20:
            del window[0]
        # k ≈ (nonzero / total) / poll_sec — i.e. how often the mid
        # actually moves divided by the polling cadence.
        nonzero = sum(1 for x in window if x > 1e-9)
        k = nonzero / max(len(window), 1) / max(self.poll_sec, 1.0)
        # Floor / ceil to plausible range
        self._k_arrival[token_id] = max(0.05, min(5.0, k))

    def _adverse_selection_widen(self, token_id: str) -> float:
        """Return a multiplier on half-spread that widens after recent
        adverse fills on this token. 1.0 = neutral; 1.5+ = widened.

        Heuristic: of the last N fills on this token, fraction that
        preceded a mid-move against us within `as_lookback_sec`.
        """
        old = self._quotes.get(token_id)
        if old is None or not old.recent_outcomes:
            return 1.0
        # recent_outcomes entries are (was_buy, mid_at_fill, mid_after)
        adverse = 0
        for was_buy, mid_at, mid_after in old.recent_outcomes[-10:]:
            if was_buy and mid_after < mid_at:    # bought, mid dropped → bad
                adverse += 1
            elif not was_buy and mid_after > mid_at:  # sold, mid rose → bad
                adverse += 1
        n = min(10, len(old.recent_outcomes))
        adverse_rate = adverse / max(n, 1)
        # Linear widening: 0% adverse → 1.0, 50% → 1.25, 100% → 1.5
        return 1.0 + 0.5 * adverse_rate

    async def _draw_passive_fill(
        self,
        quote: MakerQuote,
        book: OrderBook,
        side: str,           # "BUY" (we are buying YES at our bid) or "SELL"
        post_price: float,
    ) -> bool:
        """Bernoulli draw of whether this side's resting limit got filled
        in the last fill_horizon_sec of trading flow."""
        result = simulate_passive_fill(
            book,
            side=side,
            post_price=post_price,
            size_target=quote.quote_size,
            horizon_sec=self.fill_horizon_sec,
        )
        if result is None:
            return False
        # Per-cycle Bernoulli draw using fill_prob × (cycle/horizon) — we
        # decay the per-cycle probability since we run faster than horizon.
        per_cycle_p = result.fill_prob * (self.poll_sec / self.fill_horizon_sec)
        per_cycle_p = max(0.0, min(0.99, per_cycle_p))
        return random.random() < per_cycle_p

    async def _record_fill(
        self,
        quote: MakerQuote,
        side: str,
        post_price: float,
        book: OrderBook,
    ) -> None:
        """Route the virtual fill through the broker. The broker handles
        fee/rebate/cancel-latency accounting; we track inventory + the
        adverse-selection outcomes for the quote."""
        mid_at_fill = book.mid()
        filled = await self.broker.submit(
            strategy=self.strategy_name,
            condition_id=quote.market.condition_id,
            token_id=quote.token_id,
            side=side,
            max_size=quote.quote_size,
            max_price=post_price + 1e-9 if side == "BUY" else post_price - 1e-9,
            reason=f"passive_v2 {side} @ {post_price:.4f} (maker)",
            is_maker=True,
            category=quote.market.category,
        )
        if filled <= 0:
            return
        if side == "BUY":
            quote.bid_fills += 1
            quote.inventory += filled
        else:
            quote.ask_fills += 1
            quote.inventory -= filled
        # Record an "outcome pending" entry for adverse-selection tracking.
        # The mid at fill time is recorded; the mid_after is filled in by
        # the post-fill check on the next cycle.
        quote.recent_outcomes.append((side == "BUY", mid_at_fill or post_price, mid_at_fill or post_price))
        # Trim to keep memory bounded
        if len(quote.recent_outcomes) > 30:
            quote.recent_outcomes = quote.recent_outcomes[-30:]
        log.info(
            "passive_v2_fill",
            condition_id=quote.market.condition_id,
            token=quote.token_id[:14],
            yes_token=quote.is_yes_token,
            side=side,
            price=round(post_price, 4),
            size=round(filled, 2),
            inventory_after=round(quote.inventory, 2),
        )

    def _update_adverse_selection_outcomes(
        self, token_id: str, current_mid: float | None
    ) -> None:
        """Walk the active quote's recent_outcomes; for any entry whose
        mid_after still equals mid_at, update it with the current mid
        — this is the "where did the price actually go after our fill"
        feedback that drives the AS widening multiplier."""
        if current_mid is None:
            return
        q = self._quotes.get(token_id)
        if q is None or not q.recent_outcomes:
            return
        updated = []
        for was_buy, mid_at, mid_after in q.recent_outcomes:
            # If we haven't filled in a real "after" yet (still same as at),
            # use the current mid as the after-mid sample.
            if abs(mid_after - mid_at) < 1e-9:
                updated.append((was_buy, mid_at, current_mid))
            else:
                updated.append((was_buy, mid_at, mid_after))
        q.recent_outcomes = updated

    async def _cycle_one_token(self, m: Market, token_id: str) -> None:
        """Run one quote cycle for a single token (either YES or NO)."""
        book = self.book_store.books.get(token_id)
        ok, reason = self._eligible_market(m, book)
        if not ok:
            # If this token is no longer eligible, cancel any active quote
            if token_id in self._quotes:
                old = self._quotes.pop(token_id)
                log.info("passive_v2_cancel", token=token_id[:14],
                         rev=old.revision, reason=f"ineligible:{reason}")
            return

        cur_mid = book.mid()

        # Stale-quote cancel: if mid has moved >= stale_mid_ticks ticks
        # since we posted, bypass the cooldown and recompute now. This
        # is the "if mid moves through us, cancel before we get picked
        # off" protection that paper-mode owes a real maker.
        active = self._quotes.get(token_id)
        bypass_cooldown = False
        if active is not None and active.mid_at_post and cur_mid is not None:
            mid_delta_ticks = abs(cur_mid - active.mid_at_post) / 0.01
            if mid_delta_ticks >= self.stale_mid_ticks:
                log.info(
                    "passive_v2_cancel",
                    token=token_id[:14], rev=active.revision,
                    reason=f"stale_mid:{mid_delta_ticks:.1f}_ticks",
                    mid_at_post=round(active.mid_at_post, 4),
                    cur_mid=round(cur_mid, 4),
                )
                self._quotes.pop(token_id, None)
                active = None
                bypass_cooldown = True

        if not bypass_cooldown:
            last = self._last_cycle_ts.get(token_id, 0.0)
            if time.time() - last < self.cooldown_sec:
                return

        # Update calibration (taker arrival rate, σ) from observed mid
        # changes since the last cycle.
        self._update_calibration(token_id, book)

        # Update adverse-selection feedback (mid_after on prior fills) BEFORE
        # we recompute, so the new quote uses up-to-date AS state.
        self._update_adverse_selection_outcomes(token_id, cur_mid)

        pos = self.broker.positions.get(token_id)
        inv = pos.size if pos else 0.0
        if abs(inv) >= self.max_total_inventory_yes:
            log.info("passive_v2_skip_inventory_cap", token=token_id[:14], inv=inv)
            return

        # Calibrated taker arrival rate from observed flow on this token
        k_est = self._k_arrival.get(token_id)

        new_q = self._compute_quote(m, book, token_id, inv, k_arrival_per_sec=k_est)
        if new_q is None:
            return
        quote = self._post_or_replace(token_id, new_q)

        # Sample into the maker-rewards tracker so we can dashboard how
        # much of the $12M/yr maker-rewards pool our quotes would have
        # earned (Polymarket quadratic-spread formula).
        if self.maker_rewards is not None and cur_mid is not None:
            try:
                # Half-spread in bps from mid.
                quote_mid = (quote.bid_price + quote.ask_price) / 2.0
                half_spread_bps = abs(quote.ask_price - quote.bid_price) / 2.0 * 10_000.0
                self.maker_rewards.sample(
                    token_id=token_id,
                    our_quote_spread_bps=half_spread_bps,
                    our_quote_size=quote.quote_size,
                )
            except Exception:
                pass

        # Try virtual fills on each side, but skip the side suppressed
        # by the inventory-unwind logic (its quote is at the boundary
        # tick and effectively unfillable). Also consult the VPIN gate:
        # a maker quote against toxic one-sided flow is blocked before
        # it ever gets the chance to be picked off (pmwhybetter.md
        # Problem-1 fix #5; Bartlett-O'Hara 2026).
        if quote.unwind_side != "BUY" and self._vpin_allow(token_id, "BUY"):
            if await self._draw_passive_fill(quote, book, "BUY", quote.bid_price):
                await self._record_fill(quote, "BUY", quote.bid_price, book)
        if quote.unwind_side != "SELL" and self._vpin_allow(token_id, "SELL"):
            if await self._draw_passive_fill(quote, book, "SELL", quote.ask_price):
                if inv > 0:
                    await self._record_fill(quote, "SELL", quote.ask_price, book)
        self._last_cycle_ts[token_id] = time.time()

    def _vpin_allow(self, token_id: str, side: str) -> bool:
        """Consult the VPIN gate if attached. Returns True (allow) when
        the gate is absent or permits the quote."""
        if self.vpin_gate is None:
            return True
        try:
            allow, _ = self.vpin_gate.allow_quote(token_id, side)
        except Exception as e:
            log.warning("vpin_gate_error", err=str(e))
            return True
        if not allow:
            log.info("passive_v2_vpin_blocked", token=token_id[:14], side=side)
        return bool(allow)

    def _effective_quote_size(self, token_id: str) -> float:
        """Apply wash-graph suppression to the configured quote_size.
        Returns the size to actually post; `quote_size × (1 − wash_share)`
        clamped to a minimum 1 share so we don't post zero-size quotes."""
        if self.wash_graph_conn is None:
            return float(self.quote_size)
        try:
            from polyagent.risk.wash_graph import suppression_factor
            factor = suppression_factor(self.wash_graph_conn, token_id)
        except Exception:
            return float(self.quote_size)
        size = max(1.0, self.quote_size * factor)
        return float(size)

    async def _cycle_token(self, token_id: str) -> None:
        """Backwards-compat shim: iterate both YES and NO sides of the
        market this token belongs to, so each cycle quotes both outcomes."""
        m = self.markets_by_token.get(token_id)
        if m is None:
            return
        # Quote on YES side
        await self._cycle_one_token(m, m.yes_token_id)
        # Quote on NO side (independent inventory + book)
        await self._cycle_one_token(m, m.no_token_id)

    async def run(self) -> None:
        if not self.target_tokens:
            log.warning("passive_v2_no_targets")
            await asyncio.Event().wait()
            return
        log.info(
            "passive_v2_start",
            n_targets=len(self.target_tokens),
            poll_sec=self.poll_sec,
            allowed_categories=sorted(self.certified_categories or []),
        )
        while True:
            for tok in self.target_tokens:
                try:
                    await self._cycle_token(tok)
                except Exception as e:
                    log.warning("passive_v2_cycle_error", token=tok[:14], err=str(e))
            await asyncio.sleep(self.poll_sec)
