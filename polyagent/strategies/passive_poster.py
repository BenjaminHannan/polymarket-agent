"""Passive (maker-side) limit poster — paper-mode simulation.

Why this exists. The dominant 2026 finding (Della Vedova, Akey et al.,
Yang) is that in zero-fee Polymarket markets the maker–taker spread
transfer accounts for roughly the *entire* P&L gap between profitable
and unprofitable wallets. Yang's headline number: skilled traders earn
~$121/market making vs ~$63 taking. The doc explicitly says: "you need
the LP code path that you said is 'blocked on real money' to be your
default code path even in paper trading."

What this does. For each combined-signal candidate that satisfies a
weaker edge bar than the taker trader, instead of crossing the spread
we record a virtual passive order at a price *inside* the spread. On
each cycle we use the queue-position fill-probability model
(Cont/Kukanov/Stoikov 2014, Gould/Bonart 2016) to estimate the
probability that a taker has hit our resting order in the elapsed
window. Bernoulli draws per cycle; on a "fill" event we route through
broker.submit() with `max_price` set to our post price so the fill
records at the post price (better than VWAP).

Critical paper-mode honesty. The post price is exactly the price we'd
quote on a real CLOB. The fill probability is conservative (we only
count downward queue erosion plus opposite-side takers) and we apply
the Polygon-block cancel-latency adverse-drift penalty when realized
vol is high. We never pretend a fill we couldn't have plausibly gotten.

This converts an information edge into "edge + half-spread captured"
exactly when it works, and burns nothing when it doesn't (rest, then
cancel). Inventory risk and adverse-selection are real and gated on:
the smart-money registry can downgrade categories with sophisticated
maker presence; quote-staleness gates apply; per-condition concurrency
caps apply.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from polyagent.gamma import Market
from polyagent.models.categorize import categorize
from polyagent.orderbook import BookStore, OrderBook
from polyagent.paper_broker import PaperBroker
from polyagent.queue_model import (
    cancel_latency_slippage,
    fill_prob_top_of_book,
)
from polyagent.risk.smart_money import SmartMoneyRegistry

log = structlog.get_logger()


@dataclass
class _PostedOrder:
    market: Market
    token_id: str
    side: str               # "BUY" or "SELL"
    post_price: float       # price at which we'd post the limit
    target_size: float      # share count we want to fill in total
    filled_size: float = 0.0
    posted_ts: float = field(default_factory=time.time)
    last_check_ts: float = field(default_factory=time.time)
    p_model: float = 0.5    # the model probability that justified this post


@dataclass
class PassivePoster:
    book_store: BookStore
    broker: PaperBroker
    markets_by_token: dict[str, Market]
    poll_sec: float = 20.0
    min_edge: float = 0.04          # weaker bar than taker — half-spread is captured
    max_concurrent: int = 8
    per_post_notional: float = 25.0
    max_total_notional: float = 300.0
    aggression: float = 0.4         # 0 = quote at mid; 1 = quote at top-of-book
    ttl_sec: float = 180.0          # cancel an unfilled post after this
    max_spread: float = 0.10
    min_spread: float = 0.005
    smart_money: SmartMoneyRegistry | None = None
    strategy_name: str = "passive_poster"

    # In-flight posts indexed by token_id (only one post per token at a time)
    _open: dict[str, _PostedOrder] = field(default_factory=dict)
    # Per-token cooldown after a post completes/cancels
    _cooldown_until: dict[str, float] = field(default_factory=dict)

    def _open_notional(self) -> float:
        return sum(
            (o.target_size - o.filled_size) * o.post_price for o in self._open.values()
        )

    def _eligible(self, book: OrderBook | None) -> bool:
        if book is None or book.last_update_ts is None:
            return False
        if time.time() - book.last_update_ts > 300:
            return False
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return False
        spread = ba[0] - bb[0]
        if spread <= self.min_spread or spread > self.max_spread:
            return False
        return True

    def _post_price(self, side: str, best_bid: float, best_ask: float) -> float:
        """Quote inside the spread, fading from mid toward the touch by
        ``aggression``.  aggression=0 → mid; aggression=1 → join the touch
        (effectively a marketable post at depth-1)."""
        mid = (best_bid + best_ask) / 2.0
        if side == "BUY":
            # Buying YES: bid at mid + aggression × (best_bid - mid)
            return mid + self.aggression * (best_bid - mid)
        else:
            return mid + self.aggression * (best_ask - mid)

    async def on_signal(
        self,
        *,
        market: Market,
        p_combined: float,
        p_market: float,
        category: str,
    ) -> None:
        """Receive a combined-signal candidate and decide whether to post a
        passive limit. Lower edge bar than the taker trader because the
        half-spread we capture compensates."""
        edge = p_combined - p_market
        if abs(edge) < self.min_edge:
            return
        # Edge sanity cap (Della Vedova 2026): refuse claims of edge
        # bigger than the market's implicit prior could plausibly admit.
        if abs(edge) > min(p_market, 1.0 - p_market):
            return

        # Determine which side of the book we'd be posting on. Buying YES
        # means we want to BUY at a price inside the (bid,ask). Buying NO is
        # equivalent to selling YES.
        if edge > 0:
            token_id = market.yes_token_id
            side = "BUY"
            p = p_combined
        else:
            token_id = market.no_token_id
            side = "BUY"
            p = 1.0 - p_combined

        if token_id in self._open:
            return  # already have a resting post on this token
        now = time.time()
        if self._cooldown_until.get(token_id, 0.0) > now:
            return
        if len(self._open) >= self.max_concurrent:
            return
        if self._open_notional() + self.per_post_notional > self.max_total_notional:
            return

        # Wash-trade hygiene (Dubach 2026)
        wf = getattr(self.book_store, "wash_filter", None)
        if wf is not None and wf.is_blacklisted(token_id):
            return
        # Stop-loss re-entry block (mirror of combined_trader)
        if self.broker.was_recently_stopped(token_id) or (
            market.condition_id and self.broker.was_recently_stopped(market.condition_id)
        ):
            return
        # Hard per-token fill cap (broker-level)
        if self.broker.is_token_buy_capped(token_id):
            return
        # Averaging-down guard: refuse to add to a losing position. The
        # most expensive losses in the previous session were passive
        # posts averaging in as the price fell from $0.41 → $0.34 →
        # stop-loss at $0.19. If we already hold this token AND the
        # current mid is below avg_cost, do not post — the previous
        # entry was wrong, more of the same isn't going to fix it.
        existing = self.broker.positions.get(token_id)
        if existing and existing.size > 0 and existing.avg_cost > 0:
            book_now = self.book_store.books.get(token_id)
            mid_now = book_now.mid() if book_now is not None else None
            if mid_now is not None and mid_now < existing.avg_cost * 0.97:
                # 3% relative drop is enough to refuse — passive is
                # designed to nibble inside the spread; any drift this
                # large means the model's edge claim is stale.
                return

        book = self.book_store.books.get(token_id)
        if not self._eligible(book):
            return
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return
        post_price = self._post_price(side, bb[0], ba[0])
        # Sanity: must be a valid probability and on the right side of mid.
        if not (0.01 < post_price < 0.99):
            return
        if side == "BUY" and post_price >= ba[0]:
            return  # would be marketable, not passive
        # Smart-money downweight: in categories where sophisticated makers
        # dominate, post deeper (smaller aggression) so we capture more
        # spread to compensate for adverse-selection.
        if (
            self.smart_money is not None
            and len(self.smart_money.smart_wallets) >= 50
            and category in ("politics_us", "geopolitics", "ai", "politics_global")
        ):
            mid = (bb[0] + ba[0]) / 2.0
            # Move halfway back toward mid → less aggressive
            post_price = (post_price + mid) / 2.0

        # Ensure expected fill price is still profitable vs model.
        if side == "BUY" and post_price >= p:
            return  # negative expected value at this post price
        target_size = self.per_post_notional / max(0.01, post_price)
        order = _PostedOrder(
            market=market,
            token_id=token_id,
            side=side,
            post_price=post_price,
            target_size=target_size,
            p_model=p,
        )
        self._open[token_id] = order
        log.info(
            "passive_post",
            condition_id=market.condition_id,
            side=side,
            post_price=round(post_price, 4),
            best_bid=round(bb[0], 4),
            best_ask=round(ba[0], 4),
            target_size=round(target_size, 2),
            p_model=round(p, 4),
            edge=round(edge, 4),
            category=category,
        )

    async def _try_fill(self, order: _PostedOrder) -> None:
        """Estimate fill probability since last check; on success route to broker."""
        book = self.book_store.books.get(order.token_id)
        if book is None or not self._eligible(book):
            return
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return

        now = time.time()
        # Cancel if mid moved through our post price (we'd be marketable now —
        # in real life the post would have been filled; in paper, we simulate
        # the fill at our post price below).
        if order.side == "BUY" and ba[0] <= order.post_price:
            await self._submit_fill(order, fill_size=order.target_size - order.filled_size,
                                    price_cap=order.post_price, reason="touched")
            return

        # Otherwise, estimate per-cycle fill prob using the queue model. The
        # opposite side at the touch is doing the takings; queue ahead is the
        # rest of the depth at our quoted level (assume we joined; for posts
        # inside the spread we sit at the touch effectively).
        if order.side == "BUY":
            opp_size = ba[1]    # asks lift to fill us
            queue_ahead = 0.0   # we posted inside the spread → no queue
        else:
            opp_size = bb[1]
            queue_ahead = 0.0
        imb = book.imbalance(5) or 0.5
        p_fill_per_window = fill_prob_top_of_book(queue_ahead, opp_size, imb)
        # Cancel-latency penalty: when realized vol is high we'd have been
        # picked off during a block; cap effective fill prob accordingly.
        rv = book.realized_vol(120) if hasattr(book, "realized_vol") else None
        adverse = cancel_latency_slippage(rv)
        # As a crude attenuation, when adverse drift exceeds half the spread,
        # halve the fill probability — the alpha that "fill" represents is
        # mostly adverse-selection in that regime.
        spread = ba[0] - bb[0]
        if adverse > 0.5 * spread:
            p_fill_per_window *= 0.5

        # Per-cycle Bernoulli draw. We poll every ``poll_sec`` so this is
        # the probability of being hit during one polling window. In real
        # life this would be continuous. For paper realism, we damp by a
        # factor reflecting that not every cycle has a real taker arrival.
        # 0.10 multiplier: rough calibration to the median 30-60s arrival
        # rate of a marketable counterparty on a $5K-liquidity book.
        p_fill = max(0.0, min(0.5, p_fill_per_window * 0.10))
        if random.random() < p_fill:
            # Partial fill: assume taker sweeps a fraction of our remaining
            # target proportional to opp_size at the touch.
            remaining = order.target_size - order.filled_size
            fill = min(remaining, max(0.5, opp_size * 0.5))
            await self._submit_fill(order, fill_size=fill, price_cap=order.post_price,
                                    reason="probabilistic")
            return

        # TTL expiry → cancel
        if now - order.posted_ts > self.ttl_sec:
            log.info(
                "passive_post_cancel",
                condition_id=order.market.condition_id,
                age_sec=round(now - order.posted_ts, 1),
                filled=round(order.filled_size, 2),
                target=round(order.target_size, 2),
            )
            self._open.pop(order.token_id, None)
            self._cooldown_until[order.token_id] = now + 60.0

    async def _submit_fill(
        self,
        order: _PostedOrder,
        *,
        fill_size: float,
        price_cap: float,
        reason: str,
    ) -> None:
        """Route to broker.submit with max_price set to the post price so
        the fill records at our maker price (better than VWAP)."""
        if fill_size <= 0:
            return
        # The broker's VWAP loop walks the book up to max_price. With
        # max_price = post_price (which is < best_ask for BUY), the broker
        # will only fill if the touch has crossed inside our limit; that's
        # exactly when our "rest" got hit in real life. Otherwise it returns 0.
        filled = await self.broker.submit(
            strategy=self.strategy_name,
            condition_id=order.market.condition_id,
            token_id=order.token_id,
            side=order.side,
            max_size=fill_size,
            max_price=price_cap,
            reason=(
                f"passive {reason} side={order.side} "
                f"post={order.post_price:.4f} p_model={order.p_model:.4f}"
            ),
        )
        if filled > 0:
            order.filled_size += filled
            log.info(
                "passive_fill",
                condition_id=order.market.condition_id,
                side=order.side,
                filled=round(filled, 2),
                cum_filled=round(order.filled_size, 2),
                post_price=round(order.post_price, 4),
            )
            if order.filled_size >= order.target_size - 1e-6:
                self._open.pop(order.token_id, None)
                self._cooldown_until[order.token_id] = time.time() + 120.0

    async def run(self) -> None:
        log.info(
            "passive_poster_start",
            poll_sec=self.poll_sec,
            min_edge=self.min_edge,
            max_concurrent=self.max_concurrent,
            per_post_notional=self.per_post_notional,
        )
        while True:
            await asyncio.sleep(self.poll_sec)
            try:
                for order in list(self._open.values()):
                    await self._try_fill(order)
            except Exception as e:
                log.warning("passive_poster_loop_error", err=str(e))
