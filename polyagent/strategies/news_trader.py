"""News-driven paper trader.

Triggers on the strongest directional candidate the matcher found for a news
event. Buys YES (or NO) at the prevailing top-of-book ask, conservative size,
strict per-market and per-day caps. No exits — positions sit until market
resolution (resolution handling is a later phase).

Direction confidence here is a heuristic (VADER + question polarity), so this
is intentionally small-stakes. We're collecting labeled paper-fill outcomes
the next-phase classifier will train on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.news_store import NewsEvent
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker
from polyagent.risk.throttle import StrategyThrottler
from polyagent.signals.direction import DirectionResult

log = structlog.get_logger()


@dataclass
class NewsTrader:
    book_store: BookStore
    broker: PaperBroker
    per_trade_notional: float = 25.0
    max_per_market_notional: float = 75.0
    max_daily_notional: float = 500.0
    max_ask_price: float = 0.85  # don't chase >85¢ — bad risk/reward
    cooldown_sec: float = 300.0
    last_trade_ts: dict[str, float] = field(default_factory=dict)
    daily_window_start: float = field(default_factory=time.time)
    daily_notional_used: float = 0.0
    throttler: StrategyThrottler | None = None
    strategy_name: str = "news_trader"
    # New gates
    max_spread: float = 0.05
    min_ask_depth_usd: float = 25.0
    min_volume_24h_usd: float = 1000.0

    def _reset_day_if_needed(self) -> None:
        # 24h rolling reset
        if time.time() - self.daily_window_start > 86_400:
            self.daily_window_start = time.time()
            self.daily_notional_used = 0.0

    async def on_signal(
        self,
        market: Market,
        direction: DirectionResult,
        evt: NewsEvent,
        score: float,
    ) -> None:
        self._reset_day_if_needed()

        if direction.direction not in ("yes", "no"):
            return

        # Cooldown per market
        now = time.time()
        last = self.last_trade_ts.get(market.condition_id, 0.0)
        if now - last < self.cooldown_sec:
            return

        if self.daily_notional_used + self.per_trade_notional > self.max_daily_notional:
            log.info("news_trade_daily_cap_reached", used=round(self.daily_notional_used, 2))
            return

        throttle = self.throttler.get_mult(self.strategy_name) if self.throttler is not None else 1.0
        if throttle <= 0:
            log.info("news_trader_throttled", strategy=self.strategy_name)
            return

        token_id = market.yes_token_id if direction.direction == "yes" else market.no_token_id
        book = self.book_store.books.get(token_id)
        if book is None:
            return
        ask = book.best_ask()
        if ask is None:
            return
        ask_price, ask_size = ask
        if ask_price > self.max_ask_price:
            return
        if ask_price * ask_size < self.min_ask_depth_usd:
            return
        if (market.volume_24h or 0) < self.min_volume_24h_usd:
            return
        bid = book.best_bid()
        if bid is not None and (ask_price - bid[0]) > self.max_spread:
            return

        # Per-market cap (apply throttle to per-trade notional too).
        existing_pos = self.broker.positions.get(token_id)
        existing_notional = existing_pos.size * existing_pos.avg_cost if existing_pos else 0.0
        budget = max(0.0, self.max_per_market_notional - existing_notional)
        notional = min(self.per_trade_notional * throttle, budget)
        if notional <= 1.0:
            return

        size = min(notional / ask_price, ask_size)
        if size <= 0:
            return

        reason = (
            f"news direction={direction.direction} conf={direction.confidence:.2f} "
            f"sent={direction.sentiment:.2f} match_score={score:.2f} "
            f"src={evt.source} hash={evt.hash()}"
        )

        # Set cooldown BEFORE submit so concurrent invocations can't both pass
        # the cooldown check during the await.
        self.last_trade_ts[market.condition_id] = now

        log.info(
            "news_trade_attempt",
            condition_id=market.condition_id,
            question=market.question[:90],
            direction=direction.direction,
            ask=ask_price,
            size=round(size, 2),
            notional=round(size * ask_price, 2),
            confidence=round(direction.confidence, 2),
            news_title=evt.title[:80],
            news_source=evt.source,
        )

        filled = await self.broker.submit(
            strategy="news_trader",
            condition_id=market.condition_id,
            token_id=token_id,
            side="BUY",
            max_size=size,
            max_price=ask_price + 1e-9,
            reason=reason,
        )
        if filled > 0:
            self.daily_notional_used += filled * ask_price
        else:
            self.last_trade_ts[market.condition_id] = 0.0  # allow retry
