"""Combined-signal paper trader.

Sizing is fractional Kelly on the side the combined signal favors. For a
binary outcome bought at price q with model probability p, full Kelly is
f* = (p - q) / (1 - q). We scale by `kelly_mult` (default 0.15) and cap.

Per-category θ_min thresholds reflect the eval log-loss per category.

Trade-time gates layered on top of the signal:
  - kill switch (broker)
  - drawdown-aware Kelly
  - per-market spread filter
  - min ask depth (skip illiquid)
  - per-category concentration cap
  - fee-adjusted edge buffer
  - per-token cooldown
  - volume tier (skip dead markets)
  - stale-quote skip
  - position-concentration warning
  - daily loss kill (per-strategy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from polyagent.gamma import Market
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker
from polyagent.risk.adverse_selection import AdverseSelectionFilter
from polyagent.risk.smart_money import SmartMoneyRegistry
from polyagent.risk.throttle import StrategyThrottler

log = structlog.get_logger()


# θ_min by category — minimum |edge| in probability space to trade.
DEFAULT_THETA_MIN: dict[str, float] = {
    "crypto": 0.15,
    "sports_us": 0.20,
    "sports_global": 0.20,
    "politics_us": 0.10,
    "politics_global": 0.12,
    "geopolitics": 0.12,
    "ai": 0.15,
    "entertainment": 0.15,
    "economy": 0.15,
    "weather": 0.15,
    "other": 0.12,
}
DEFAULT_THETA_FALLBACK = 0.15


@dataclass
class CombinedTrader:
    book_store: BookStore
    broker: PaperBroker
    kelly_mult: float = 0.15
    max_per_trade_kelly: float = 0.05  # cap as fraction of NAV
    max_per_trade_notional: float = 50.0
    max_per_market_notional: float = 150.0
    max_daily_notional: float = 1000.0
    max_ask: float = 0.95
    cooldown_sec: float = 600.0
    theta_min_by_category: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THETA_MIN))
    theta_min_default: float = DEFAULT_THETA_FALLBACK
    last_trade_ts: dict[str, float] = field(default_factory=dict)
    last_token_trade_ts: dict[str, float] = field(default_factory=dict)
    daily_window_start: float = field(default_factory=time.time)
    daily_notional_used: float = 0.0
    daily_loss_realized: float = 0.0
    daily_loss_kill: float = 250.0  # halt for the day if we drop $250
    throttler: StrategyThrottler | None = None
    strategy_name: str = "combined_trader"

    # New gates
    max_spread: float = 0.05      # skip markets with spread > 5pp
    min_ask: float = 0.10         # refuse longshots — Della Vedova's
                                  # half-spread is largest below $0.10
    min_ask_depth_usd: float = 25.0
    fee_buffer: float = 0.02      # 2pp buffer (was 0.5pp); slippage
                                  # diagnostic showed ~22% of notional
                                  # eaten on the longshot book
    token_cooldown_sec: float = 120.0
    max_category_pct_of_nav: float = 0.30
    min_volume_24h_usd: float = 1000.0
    quote_freshness_sec: float = 300.0  # skip if last book update was >5 min ago
    drawdown_kelly_floor: float = 0.2   # never scale below 20% of base kelly
    # Tighten edge requirement when we're underwater. theta_min scales by
    # (1 + dd_theta_scale * drawdown_pct) so a 5% drawdown raises theta by
    # 1.5x, gating in only the strongest signals.
    dd_theta_scale: float = 10.0
    # Per-category contextual bandit state — Thompson-sampled multiplier on
    # kelly_mult, learned from realized P&L per category over time.
    bandit_alpha: dict[str, float] = field(default_factory=dict)  # successes
    bandit_beta: dict[str, float] = field(default_factory=dict)   # failures
    adverse_filter: AdverseSelectionFilter | None = None
    # Smart-money adverse-selection registry (Yang 2026 / Solidus 2026): when
    # a known top-PnL maker has been recently active in this market we tighten
    # theta_min by ``smart_money_theta_mult`` (post deeper inside the spread).
    smart_money: SmartMoneyRegistry | None = None
    smart_money_theta_mult: float = 1.5
    # News-store handle so we can look up recent market activity associated
    # with smart wallets (set externally; we never construct it ourselves).
    news_store: object | None = None

    def threshold(self, category: str) -> float:
        return self.theta_min_by_category.get(category, self.theta_min_default)

    def bandit_sample(self, category: str) -> float:
        """Thompson sample a [0, 1] multiplier on kelly_mult for this category.

        Beta(alpha, beta) where alpha = wins, beta = losses, both seeded at 1.
        Mean of the posterior = alpha / (alpha + beta). Sampled value provides
        exploration. Categories with no data get a near-uniform prior, so
        early sampling is wide.
        """
        import random
        a = self.bandit_alpha.get(category, 1.0)
        b = self.bandit_beta.get(category, 1.0)
        # Approx Beta sampling via two gamma vars (Marsaglia-style):
        # Python's random.betavariate is fine.
        try:
            return float(random.betavariate(a, b))
        except ValueError:
            return 0.5

    def bandit_update(self, category: str, won: bool) -> None:
        if won:
            self.bandit_alpha[category] = self.bandit_alpha.get(category, 1.0) + 1.0
        else:
            self.bandit_beta[category] = self.bandit_beta.get(category, 1.0) + 1.0

    def _reset_day_if_needed(self) -> None:
        if time.time() - self.daily_window_start > 86400:
            self.daily_window_start = time.time()
            self.daily_notional_used = 0.0
            self.daily_loss_realized = 0.0

    def _category_notional(self, category: str) -> float:
        """Sum of (size * mid) across all open positions in this category."""
        from polyagent.models.categorize import categorize as _cat
        from polyagent.gamma import Market as _M  # noqa: F401
        total = 0.0
        # Walk broker positions; classify by question we can resolve via book_store
        for tok, pos in self.broker.positions.items():
            if pos.size <= 0:
                continue
            book = self.book_store.books.get(tok)
            mid = book.mid() if book else None
            if mid is None:
                mid = pos.avg_cost
            total += pos.size * mid
        # NOTE: this returns total open notional, not per-category; that's still
        # useful as a guardrail. (Per-category lookup needs a token->Market map
        # which the trader doesn't currently hold; left as a global cap.)
        return total

    async def on_signal(
        self,
        *,
        market: Market,
        p_combined: float,
        p_market: float,
        category: str,
        p_combined_low: float | None = None,
    ) -> None:
        self._reset_day_if_needed()

        # Daily loss kill: if this strategy has bled more than its allowance,
        # stop entirely until the rolling 24h window resets.
        if self.daily_loss_realized < -self.daily_loss_kill:
            log.info(
                "combined_trader_daily_loss_kill",
                loss=round(self.daily_loss_realized, 2),
                kill=self.daily_loss_kill,
            )
            return

        # Volume gate
        if (market.volume_24h or 0) < self.min_volume_24h_usd:
            return

        edge_raw = p_combined - p_market
        # EDGE SANITY CAP (Della Vedova 2026 / Whelan 2024). A claim of
        # +25pp edge on a 3¢ market implies the market is wrong by 8×.
        # An AUC=0.77 question-only model has no statistical right to
        # that claim. Refuse trades where |edge| exceeds the smaller of
        # p_market or 1-p_market — that's the maximum the model could
        # plausibly add given the market's prior information.
        if abs(edge_raw) > min(p_market, 1.0 - p_market):
            return
        # Fee-adjusted edge: only count edge above the fee/slippage buffer.
        edge_eff = abs(edge_raw) - self.fee_buffer
        # Drawdown-conditioned theta: tighten edge requirement when we're
        # in drawdown. At 0% dd, theta is unchanged. At 5% dd, theta × 1.5.
        dd = self.broker.drawdown.drawdown(self.broker.nav(mark="mid"))
        theta = self.threshold(category) * (1.0 + self.dd_theta_scale * dd)
        if edge_eff < theta:
            return

        now = time.time()
        # Per-market cooldown
        last = self.last_trade_ts.get(market.condition_id, 0.0)
        if now - last < self.cooldown_sec:
            return

        if self.daily_notional_used + 1.0 >= self.max_daily_notional:
            return

        if edge_raw > 0:
            token_id = market.yes_token_id
            p = p_combined
            side_label = "yes"
        else:
            token_id = market.no_token_id
            p = 1.0 - p_combined
            side_label = "no"

        # Per-token cooldown (separate from per-market — covers cases where
        # both YES and NO trade rapidly on the same condition)
        last_t = self.last_token_trade_ts.get(token_id, 0.0)
        if now - last_t < self.token_cooldown_sec:
            return

        # Adverse-selection: skip tokens that have a recent history of
        # decisive losses (likely we're getting picked off in those markets).
        if self.adverse_filter is not None and self.adverse_filter.is_blacklisted(token_id):
            log.info("combined_trade_skip_adverse", token_id=token_id[:14])
            return

        # Stop-loss re-entry block: refuse to re-buy a token (or even a
        # condition) that took a stop-loss in the last 24h. The model's
        # "edge" claim was wrong once; it's still wrong.
        if self.broker.was_recently_stopped(token_id) or (
            market.condition_id and self.broker.was_recently_stopped(market.condition_id)
        ):
            log.info("combined_trade_skip_recently_stopped", token_id=token_id[:14])
            return
        # Hard per-token fill cap (broker-level): no strategy may exceed
        # ``broker.max_buys_per_token_window`` BUYs on the same token in
        # the rolling window. Stops the cycling-fill pattern where a
        # single token accumulates 20+ fills as the model emits the same
        # edge claim repeatedly.
        if self.broker.is_token_buy_capped(token_id):
            log.info(
                "combined_trade_skip_token_buy_capped",
                token_id=token_id[:14],
                count=self.broker.buys_in_window(token_id),
            )
            return

        # Averaging-down guard: if we already hold this token AND the
        # current best ask is below our entry by ANY material amount,
        # do NOT add to the position. The model has been wrong on this
        # one and we should stop catching the falling knife. Threshold
        # is 3% relative drop OR 2pp absolute (whichever is smaller) —
        # tightened from 15%/5pp because the previous session showed
        # 4 fills at the same price as a token fell to a stop-loss.
        existing_pos = self.broker.positions.get(token_id)
        if existing_pos and existing_pos.size > 0 and existing_pos.avg_cost > 0:
            book_ask = self.book_store.books.get(token_id)
            ba_now = book_ask.best_ask() if book_ask is not None else None
            if ba_now is not None:
                ask_now = ba_now[0]
                drop_abs = existing_pos.avg_cost - ask_now
                drop_rel = drop_abs / max(existing_pos.avg_cost, 1e-6)
                if drop_abs >= 0.02 or drop_rel >= 0.03:
                    log.info(
                        "combined_trade_skip_averaging_down",
                        token_id=token_id[:14],
                        avg_cost=round(existing_pos.avg_cost, 4),
                        ask_now=round(ask_now, 4),
                        drop_pct=round(drop_rel * 100, 2),
                    )
                    return

        # Wash-trade hygiene (Dubach 2026): skip tokens whose recent trade
        # stream shows >max_wash_share of trades with no concurrent book
        # change.
        wf = getattr(self.book_store, "wash_filter", None)
        if wf is not None and wf.is_blacklisted(token_id):
            log.info("combined_trade_skip_wash", token_id=token_id[:14])
            return

        # Smart-money AS gate (Yang 2026 / Solidus 2026). When the registry
        # has been populated (i.e. we know who the top-PnL maker wallets are)
        # AND we're in a category historically dominated by sophisticated
        # makers (politics_us, geopolitics, ai), tighten theta by
        # ``smart_money_theta_mult`` — equivalent to "post only deeper inside
        # the spread when sharks are likely on the other side". A future
        # real-money build will replace this with an on-chain check that
        # returns True iff a known smart wallet has a resting order on the
        # opposite side of this token at our target price.
        if (
            self.smart_money is not None
            and len(self.smart_money.smart_wallets) >= 50
            and category in ("politics_us", "geopolitics", "ai", "politics_global")
        ):
            theta_sm = theta * self.smart_money_theta_mult
            if edge_eff < theta_sm:
                log.info(
                    "combined_trade_skip_smart_money_as",
                    category=category,
                    edge_eff=round(edge_eff, 3),
                    theta_sm=round(theta_sm, 3),
                    n_smart=len(self.smart_money.smart_wallets),
                )
                return

        book = self.book_store.books.get(token_id)
        if book is None:
            return

        # Stale-quote skip: if the book hasn't moved in N seconds, the quote
        # is suspect (server may have stopped streaming for this asset).
        if book.last_update_ts is not None:
            age = now - book.last_update_ts
            if age > self.quote_freshness_sec:
                return
            # Latency p99 gate: if the book is older than the broker's empirical
            # p99 for WSS book updates, refuse — we'd be trading on a stale
            # snapshot relative to our own pipeline's typical freshness.
            try:
                p99 = self.broker.latency.p99(source="wss_book")
            except Exception:
                p99 = None
            if p99 is not None and age > 5 * p99:
                log.info(
                    "combined_trade_skip_stale_p99",
                    age_sec=round(age, 2),
                    p99=round(p99, 2),
                )
                return

        ask = book.best_ask()
        bid = book.best_bid()
        if ask is None:
            return
        ask_price, ask_size = ask
        if ask_price > self.max_ask or ask_price <= 0 or ask_price >= 1.0:
            return
        # Longshot floor (Della Vedova 2026, doc Lever 1): half-spreads on
        # low-probability deciles are 650-900 bps, swallowing model edge.
        if ask_price < self.min_ask:
            return

        # Spread filter
        if bid is not None:
            spread = ask_price - bid[0]
            if spread > self.max_spread:
                return
        else:
            return  # no bid -> no spread info -> skip

        # Min depth: skip if the ask side has too little money behind it
        if ask_price * ask_size < self.min_ask_depth_usd:
            return

        # Conformal-Kelly (idea #9, Vovk): when the cell has a Venn-Abers
        # calibrator we get a worst-case probability bound. Size on the
        # *lower* bound (worst case for our directional bet) instead of
        # the point estimate — turns "Kelly with a guess" into "Kelly with
        # a finite-sample guarantee". Falls back to the point if no bound.
        if p_combined_low is not None and 0.0 < p_combined_low < 1.0:
            if edge_raw > 0:
                # Buying YES: worst case is lower combined p
                p_for_kelly = p_combined_low
            else:
                # Buying NO: worst case is higher YES → lower NO
                p_for_kelly = 1.0 - max(p_combined, p_combined_low)
                # If interval doesn't include p_combined upper, this is fine
            p_for_kelly = max(0.001, min(0.999, p_for_kelly))
        else:
            p_for_kelly = p

        # Full Kelly: f* = (p_worst - q) / (1 - q)
        f_full = (p_for_kelly - ask_price) / (1.0 - ask_price)
        if f_full <= 0:
            return

        # Throttle from auto-throttler
        throttle = self.throttler.get_mult(self.strategy_name) if self.throttler is not None else 1.0
        if throttle <= 0:
            return

        # Drawdown-aware Kelly: scale down when we're in drawdown
        dd_scale = max(self.drawdown_kelly_floor, 1.0 - 5.0 * dd)

        # Bandit kelly: Thompson-sample a per-category multiplier, learned
        # from realized P&L. New categories use a uniform-ish prior, so
        # early samples explore widely; mature categories converge.
        bandit_scale = self.bandit_sample(category)

        f_used = min(
            self.max_per_trade_kelly,
            self.kelly_mult * f_full * throttle * dd_scale * bandit_scale,
        )
        if f_used <= 0:
            return

        nav = self.broker.nav(mark="bid")  # use honest liquidation NAV for sizing
        if nav <= 0:
            return
        notional = f_used * nav
        notional = min(notional, self.max_per_trade_notional)

        existing = self.broker.positions.get(token_id)
        existing_notional = existing.size * existing.avg_cost if existing else 0.0
        per_market_room = max(0.0, self.max_per_market_notional - existing_notional)
        notional = min(notional, per_market_room)

        # Global concentration cap as a placeholder for per-category cap.
        cat_notional = self._category_notional(category)
        cap = self.max_category_pct_of_nav * nav
        cat_room = max(0.0, cap - cat_notional)
        notional = min(notional, cat_room)

        daily_room = max(0.0, self.max_daily_notional - self.daily_notional_used)
        notional = min(notional, daily_room)

        if notional < 1.0:
            return

        size = min(notional / ask_price, ask_size)
        if size <= 0:
            return

        # Set cooldowns BEFORE submit so concurrent invocations can't both pass.
        self.last_trade_ts[market.condition_id] = now
        self.last_token_trade_ts[token_id] = now

        reason = (
            f"combined cat={category} side={side_label} edge={edge_raw:+.3f} "
            f"p_comb={p_combined:.3f} p_mkt={p_market:.3f} "
            f"theta={theta:.2f} kelly_full={f_full:.3f} kelly_used={f_used:.4f} "
            f"dd_scale={dd_scale:.2f}"
        )

        log.info(
            "combined_trade_attempt",
            condition_id=market.condition_id,
            question=market.question[:90],
            category=category,
            side=side_label,
            ask=ask_price,
            spread=round(ask_price - bid[0], 4),
            depth_usd=round(ask_price * ask_size, 2),
            size=round(size, 2),
            notional=round(size * ask_price, 2),
            edge=round(edge_raw, 3),
            kelly_full=round(f_full, 3),
            dd=round(dd * 100, 2),
        )

        filled = await self.broker.submit(
            strategy=self.strategy_name,
            condition_id=market.condition_id,
            token_id=token_id,
            side="BUY",
            max_size=size,
            max_price=ask_price + 1e-9,
            reason=reason,
        )
        if filled > 0:
            self.daily_notional_used += filled * ask_price
            # Concentration warning: if this single position now exceeds 5% of NAV
            pos_after = self.broker.positions.get(token_id)
            if pos_after and (pos_after.size * ask_price) > 0.05 * nav:
                log.warning(
                    "concentration_warning",
                    condition_id=market.condition_id,
                    token_id=token_id[:14],
                    size=round(pos_after.size, 2),
                    pct_of_nav=round((pos_after.size * ask_price) / nav * 100, 2),
                )
        else:
            # Free up cooldown for retry
            self.last_trade_ts[market.condition_id] = 0.0
            self.last_token_trade_ts[token_id] = 0.0
