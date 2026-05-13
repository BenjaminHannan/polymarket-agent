"""M&O copy-trader: paper-trades on a 10–30 minute lag after a
watch-list wallet adds ≥$500 to a position.

Companion to `polyagent/signals/mitts_ofir_screen.py`. The screen
identifies flagged wallets; this strategy watches the trade tape for
flagged-wallet position increases and fires a same-side paper trade
on a configurable lag.

Why a lag
---------
Mitts & Ofir's 69.9% win rate is *not* a microsecond-execution
phenomenon. The information edge survives at 10–30 minutes because
the underlying signal is informed-trader entry, not market-microstructure
race. The lag also:
  - lets Polygon block finalization complete
  - lets the order book stabilise after the flagged wallet's entry
  - sidesteps any latency arms-race that would void the edge if we
    were trying to front-run

Why a fixed-size copy
---------------------
Per the doc: paper-place a 25%-sized order on the same side. We don't
know what fraction of the flagged wallet's bankroll their bet
represents; copying their notional 1:1 risks concentration. 25% is a
defensible default (the doc's recommendation) and adjustable via env.

Safety
------
- Cert gate: this strategy bypasses the standard cert gate because
  the EV thesis is different (mimicry of informed wallets, not model
  edge). But it has its own per-trade size cap (`max_per_trade_usd`).
- Adverse selection: a watch-list wallet that's been flagged for a
  *long* time is more likely to be over-fitted by copycats already.
  Score decay is handled at the screen level.
- Cooldown: 60s per (wallet, asset) so we don't re-trigger on the
  same flagged event.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class MittsOfirCopyTrader:
    """Long-running supervised task.

    Args:
        broker: PaperBroker.
        markets_by_asset: map of asset_id (token_id) → Market for
            condition_id + category lookup.
        screen_db_conn: sqlite handle that has the `mitts_ofir_watchlist`
            table populated by the screen.
        poll_sec: how often to check the watch-list for recent
            position-size updates.
        lag_sec: how long to wait after a flagged wallet's entry
            before placing our paper trade. Default 900 (15 min);
            doc range is 600–1800 (10–30 min).
        min_position_size_usd: floor on the flagged wallet's added
            position. Below this we don't bother copying.
        copy_fraction: fraction of the flagged wallet's added notional
            we copy in our own paper trade. Default 0.25.
        max_per_trade_usd: hard cap on the size of each copy trade.
        cooldown_sec: per-(wallet, asset) cooldown between copies.
    """
    broker: object
    markets_by_asset: dict
    screen_db_conn: sqlite3.Connection
    poll_sec: float = 60.0
    lag_sec: float = 900.0
    min_position_size_usd: float = 500.0
    copy_fraction: float = 0.25
    max_per_trade_usd: float = 100.0
    cooldown_sec: float = 60.0
    # State
    _pending: deque = field(default_factory=lambda: deque(maxlen=2000))
    _recent_cooldown: dict = field(default_factory=dict)
    _trades_fired: int = 0

    async def run(self) -> None:
        log.info(
            "mitts_ofir_copy_trader_start",
            poll_sec=self.poll_sec,
            lag_sec=self.lag_sec,
            copy_fraction=self.copy_fraction,
            max_per_trade_usd=self.max_per_trade_usd,
        )
        while True:
            try:
                await self._enqueue_recent_signals()
                await self._fire_lag_complete()
            except Exception as e:
                log.warning("mitts_ofir_copy_trader_error", err=str(e))
            await asyncio.sleep(self.poll_sec)

    async def _enqueue_recent_signals(self) -> None:
        """Check the watch-list for entries with recent position updates
        and queue them for delayed firing."""
        from polyagent.signals.mitts_ofir_screen import recent_watchlist_entries
        try:
            entries = recent_watchlist_entries(
                self.screen_db_conn,
                since_ts=time.time() - self.lag_sec - self.poll_sec * 2,
                min_size=self.min_position_size_usd,
            )
        except sqlite3.Error as e:
            log.warning("mo_copy_query_failed", err=str(e))
            return
        for e in entries:
            key = (e["wallet"], e["asset"])
            # Cooldown
            last_used = self._recent_cooldown.get(key, 0.0)
            if time.time() - last_used < self.cooldown_sec:
                continue
            # Dedupe within the queue
            if any(p[1] == key for p in self._pending):
                continue
            self._pending.append((
                e["last_position_ts"] + self.lag_sec,   # fire_at
                key,
                e["last_position_size"],
                e["composite_z"],
            ))

    async def _fire_lag_complete(self) -> None:
        """Trigger any queued copy-trades whose lag has elapsed."""
        now = time.time()
        # Walk the queue: anything whose fire_at <= now should fire.
        new_queue: deque = deque(maxlen=self._pending.maxlen)
        for item in self._pending:
            fire_at, key, size_obs, z = item
            if fire_at > now:
                new_queue.append(item)
                continue
            wallet, asset = key
            market = self.markets_by_asset.get(asset)
            if market is None:
                continue
            # Side inference: we don't have on-chain side here; defer
            # to the most-recent direction observed for the wallet
            # on this asset.
            side = await self._infer_side(wallet, asset)
            if side is None:
                continue
            notional = min(
                self.max_per_trade_usd,
                float(size_obs) * float(self.copy_fraction),
            )
            if notional < 1.0:
                continue
            log.info(
                "mo_copy_trade_fire",
                wallet=wallet[:14], asset=asset[:14],
                side=side, composite_z=round(z, 2),
                notional_usd=round(notional, 2),
                lag_sec=round(self.lag_sec, 0),
            )
            try:
                await self.broker.submit(
                    strategy="mitts_ofir_copy",
                    condition_id=market.condition_id,
                    token_id=asset,
                    side=side,
                    max_size=notional,  # broker walks the book; size in USDC
                    max_price=None,
                    reason=f"mo_copy z={z:.2f} src_wallet={wallet[:14]}",
                )
                self._trades_fired += 1
                self._recent_cooldown[key] = now
            except Exception as e:
                log.warning("mo_copy_submit_failed", err=str(e))
        self._pending = new_queue

    async def _infer_side(self, wallet: str, asset: str) -> str | None:
        """Return the most-recent side observed for (wallet, asset)
        on the trades tape. Used to copy in the same direction."""
        try:
            row = self.screen_db_conn.execute(
                """SELECT side FROM trades
                   WHERE wallet=? AND asset=?
                   ORDER BY timestamp DESC LIMIT 1""",
                (wallet, asset),
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        s = (row[0] or "").upper()
        return s if s in ("BUY", "SELL") else None

    def summary(self) -> dict:
        return {
            "pending_lag_queue": len(self._pending),
            "trades_fired": self._trades_fired,
            "cooldown_entries": len(self._recent_cooldown),
            "lag_sec": self.lag_sec,
            "copy_fraction": self.copy_fraction,
        }
