"""YES + NO ask-side arb detector.

For a binary market the YES and NO outcomes must sum to $1 at resolution.
If best-ask(YES) + best-ask(NO) < $1 - threshold, taking both legs locks in
risk-free profit (modulo fees). Polymarket's most liquid markets are zero-fee,
so threshold = 1 - epsilon (default 0.99) is the right operating point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker

log = structlog.get_logger()


@dataclass
class YesNoArb:
    book_store: BookStore
    broker: PaperBroker
    markets_by_token: dict[str, Market]
    threshold: float = settings.arb_threshold
    per_trade_size: float = settings.per_trade_size
    max_per_market: float = settings.max_per_market
    cooldown_sec: float = 30.0
    last_trade_ts: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last_trade_ts is None:
            self.last_trade_ts = {}

    async def on_event(self, token_id: str) -> None:
        market = self.markets_by_token.get(token_id)
        if market is None:
            return

        yes_book = self.book_store.books.get(market.yes_token_id)
        no_book = self.book_store.books.get(market.no_token_id)
        if yes_book is None or no_book is None:
            return

        yes_ask = yes_book.best_ask()
        no_ask = no_book.best_ask()
        if yes_ask is None or no_ask is None:
            return

        total = yes_ask[0] + no_ask[0]
        if total >= self.threshold:
            return

        now = time.time()
        last = self.last_trade_ts.get(market.condition_id, 0.0)
        if now - last < self.cooldown_sec:
            return

        # Per-market notional cap
        yes_pos = self.broker.positions.get(market.yes_token_id)
        no_pos = self.broker.positions.get(market.no_token_id)
        used = (yes_pos.size * yes_pos.avg_cost if yes_pos else 0) + (
            no_pos.size * no_pos.avg_cost if no_pos else 0
        )
        budget = max(0.0, self.max_per_market - used)
        if budget <= 0:
            return

        # Cooldown is set BEFORE the submits to prevent concurrent invocations
        # from both passing the cooldown check during the await window.
        self.last_trade_ts[market.condition_id] = now

        # Symmetric size on both legs, capped by ask depth and budget
        max_size = min(yes_ask[1], no_ask[1], self.per_trade_size, budget / total)
        if max_size <= 0:
            return

        edge_per_share = 1.0 - total
        reason = (
            f"yes_ask={yes_ask[0]:.4f} no_ask={no_ask[0]:.4f} sum={total:.4f} "
            f"edge={edge_per_share:.4f}"
        )
        log.info(
            "arb_signal",
            condition_id=market.condition_id,
            question=market.question[:80],
            yes_ask=yes_ask[0],
            no_ask=no_ask[0],
            sum=round(total, 4),
            edge=round(edge_per_share, 4),
            size=round(max_size, 2),
        )

        yes_filled = await self.broker.submit(
            strategy="yes_no_arb",
            condition_id=market.condition_id,
            token_id=market.yes_token_id,
            side="BUY",
            max_size=max_size,
            max_price=yes_ask[0] + 1e-9,
            reason=reason,
        )
        if yes_filled <= 0:
            return

        no_filled = await self.broker.submit(
            strategy="yes_no_arb",
            condition_id=market.condition_id,
            token_id=market.no_token_id,
            side="BUY",
            max_size=yes_filled,
            max_price=no_ask[0] + 1e-9,
            reason=reason,
        )

        # Unhedge protection: if NO fill came up short (price moved during
        # await), unwind the excess YES leg back to best_bid so we don't sit
        # on a directional position. Better to eat half the spread than carry
        # an unintended long.
        if no_filled < yes_filled:
            excess = yes_filled - no_filled
            log.warning(
                "arb_partial_unhedge",
                condition_id=market.condition_id,
                yes_filled=yes_filled,
                no_filled=no_filled,
                excess=round(excess, 2),
            )
            unwound = await self.broker.submit(
                strategy="yes_no_arb_unwind",
                condition_id=market.condition_id,
                token_id=market.yes_token_id,
                side="SELL",
                max_size=excess,
                max_price=None,  # take whatever the bid is
                reason=f"unhedge yes_filled={yes_filled} no_filled={no_filled}",
            )
            if unwound < excess:
                log.error(
                    "arb_unwind_failed",
                    condition_id=market.condition_id,
                    excess=excess,
                    unwound=unwound,
                    note="naked YES position remains; will resolve at expiry",
                )
