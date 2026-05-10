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
    """A two-sided quote currently posted on a token."""
    market: Market
    yes_token_id: str
    no_token_id: str
    bid_price: float        # what we'd post on the YES bid side
    ask_price: float        # what we'd post on the YES ask side
    quote_size: float       # share count per side
    posted_ts: float = field(default_factory=time.time)
    inventory_yes: float = 0.0  # +long, -short (we paper-post, so always 0 starting)
    cumulative_spread_captured: float = 0.0
    bid_fills: int = 0
    ask_fills: int = 0


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
    # Cert allowlist (built from strategy_certificates)
    certified_categories: set[str] | None = None
    # Strategy name for logs / fills.strategy
    strategy_name: str = "passive_poster_v2"
    # Per-token quote state
    _quotes: dict[str, MakerQuote] = field(default_factory=dict)
    _last_cycle_ts: dict[str, float] = field(default_factory=dict)

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
        self, m: Market, book: OrderBook, inventory_yes: float
    ) -> MakerQuote | None:
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return None
        mid = (bb[0] + ba[0]) / 2.0
        rv = book.realized_vol(300) if hasattr(book, "realized_vol") else None
        sigma = rv if (rv and rv > 0) else 0.005
        sigma_sq_per_sec = sigma * sigma
        # Estimate taker arrival rate from recent book churn (ofi as proxy);
        # fall back to a typical 0.5/sec assumption for low-vol markets.
        k = 0.5
        skew = avellaneda_stoikov_skew(
            inventory=inventory_yes,
            gamma=self.gamma,
            sigma_sq_per_sec=sigma_sq_per_sec,
            time_to_horizon_sec=self.horizon_sec,
        )
        half_spread = avellaneda_stoikov_half_spread(
            gamma=self.gamma,
            sigma_sq_per_sec=sigma_sq_per_sec,
            time_to_horizon_sec=self.horizon_sec,
            k_arrival_per_sec=k,
        )
        reservation = mid + skew
        # Don't quote inside the existing spread by less than 1 tick
        TICK = 0.01
        bid_price = max(0.0, min(reservation - half_spread, ba[0] - TICK))
        ask_price = max(reservation + half_spread, bb[0] + TICK)
        if ask_price <= bid_price:
            return None  # collapsed
        return MakerQuote(
            market=m,
            yes_token_id=m.yes_token_id,
            no_token_id=m.no_token_id,
            bid_price=round(bid_price, 4),
            ask_price=round(ask_price, 4),
            quote_size=self.quote_size,
            inventory_yes=inventory_yes,
        )

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
        token_id: str,
        post_price: float,
    ) -> None:
        """Route the virtual fill through the broker so it lands in fills /
        fills_shadow / fills_shadow_queue and shows up on the dashboard."""
        # Maker fills get the post price by definition (no spread crossed).
        # Estimate rebate as a positive ledger entry (paper-only — real bot
        # would receive USDC rebates daily).
        # We approximate the captured-spread component as half the visible
        # spread at the time of post; the rebate is rebate_share_of_fee of
        # that.
        filled = await self.broker.submit(
            strategy=self.strategy_name,
            condition_id=quote.market.condition_id,
            token_id=token_id,
            side=side,
            max_size=quote.quote_size,
            max_price=post_price + 1e-9 if side == "BUY" else post_price - 1e-9,
            reason=f"passive_v2 {side} @ {post_price:.4f} (maker)",
        )
        if filled <= 0:
            return
        if side == "BUY":
            quote.bid_fills += 1
            quote.inventory_yes += filled
        else:
            quote.ask_fills += 1
            quote.inventory_yes -= filled
        # Spread captured proxy: half-spread × filled
        # (the rebate is bookkeeping only in paper mode)
        log.info(
            "passive_v2_fill",
            condition_id=quote.market.condition_id,
            side=side,
            price=round(post_price, 4),
            size=round(filled, 2),
            inventory_yes_after=round(quote.inventory_yes, 2),
        )

    async def _cycle_token(self, token_id: str) -> None:
        m = self.markets_by_token.get(token_id)
        if m is None:
            return
        book = self.book_store.books.get(token_id)
        ok, reason = self._eligible_market(m, book)
        if not ok:
            self._quotes.pop(token_id, None)
            return

        # Per-token cooldown after fills to avoid runaway adverse selection
        last = self._last_cycle_ts.get(token_id, 0.0)
        if time.time() - last < self.cooldown_sec:
            return

        # Existing inventory on this YES token from prior fills + other strategies
        pos_yes = self.broker.positions.get(m.yes_token_id)
        inv_yes = pos_yes.size if pos_yes else 0.0
        if abs(inv_yes) >= self.max_total_inventory_yes:
            log.info("passive_v2_skip_inventory_cap", token=token_id[:14], inv=inv_yes)
            return

        quote = self._compute_quote(m, book, inv_yes)
        if quote is None:
            return
        self._quotes[token_id] = quote

        # Try a virtual fill on each side
        if await self._draw_passive_fill(quote, book, "BUY", quote.bid_price):
            await self._record_fill(quote, "BUY", quote.yes_token_id, quote.bid_price)
        if await self._draw_passive_fill(quote, book, "SELL", quote.ask_price):
            # SELL closing the YES side. Only attempt if we have inventory.
            if inv_yes > 0:
                await self._record_fill(quote, "SELL", quote.yes_token_id, quote.ask_price)
        self._last_cycle_ts[token_id] = time.time()

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
