"""Per-strategy auto-throttling based on realized P&L (v2 §9.2).

Reads attribution over a rolling 30-day window. Sets a per-strategy
multiplier in [0.0, 1.0] applied on top of `kelly_mult`:

    pnl_pct_of_nav  >= 0          -> 1.0  (full size)
    pnl_pct_of_nav  in (-2%, 0%]  -> 1.0
    pnl_pct_of_nav  in (-5%, -2%] -> 0.5  (halve the size)
    pnl_pct_of_nav  <= -5%        -> 0.0  (kill the strategy)
    sharpe < 0.3 (>=5 daily samples) -> at most 0.5

The strategy keeps trading while throttled to 0.5 (so we still gather data),
but a kill (0.0) blocks all new entries until manually reset.

Multipliers default to 1.0 for any strategy without enough data
(< MIN_RESOLVED_FOR_DECISION resolved fills).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

import structlog

from polyagent.config import settings
from polyagent.risk.attribution import StrategyAttribution, attribute_pnl

_PERSIST_PATH = Path(settings.db_path).parent / "throttle.json"

log = structlog.get_logger()


WINDOW_SEC = 30 * 86400
MIN_RESOLVED_FOR_DECISION = 5
MIN_DAYS_FOR_SHARPE = 5
PNL_PCT_HALF = -0.02
PNL_PCT_KILL = -0.05
SHARPE_FLOOR = 0.3
TRADING_DAYS_PER_YEAR = 365


@dataclass
class StrategyMetric:
    strategy: str
    n_resolved: int
    realized_pnl: float
    pnl_pct_of_nav: float
    n_days: int
    daily_sharpe: float | None
    multiplier: float


@dataclass
class StrategyThrottler:
    db_path: str
    nav_reference: float
    multipliers: dict[str, float] = field(default_factory=dict)
    last_metrics: dict[str, StrategyMetric] = field(default_factory=dict)
    last_refresh_ts: float = 0.0

    def get_mult(self, strategy: str) -> float:
        return self.multipliers.get(strategy, 1.0)

    def load_persisted(self) -> None:
        """Restore multipliers from disk so a hard kill stays killed across restart."""
        if not _PERSIST_PATH.exists():
            return
        try:
            data = json.loads(_PERSIST_PATH.read_text())
            self.multipliers = {k: float(v) for k, v in (data.get("multipliers") or {}).items()}
            log.info("throttle_loaded_persisted", multipliers=self.multipliers)
        except Exception as e:
            log.warning("throttle_load_error", err=str(e))

    def _save_persisted(self) -> None:
        try:
            _PERSIST_PATH.write_text(
                json.dumps(
                    {"multipliers": self.multipliers, "ts": self.last_refresh_ts}
                )
            )
        except Exception as e:
            log.warning("throttle_save_error", err=str(e))

    def _sharpe(self, daily_pnl: dict[str, float]) -> float | None:
        if len(daily_pnl) < MIN_DAYS_FOR_SHARPE:
            return None
        values = list(daily_pnl.values())
        m = mean(values)
        s = pstdev(values)
        if s == 0:
            return None
        return float(m / s * math.sqrt(TRADING_DAYS_PER_YEAR))

    def _decide_mult(self, att: StrategyAttribution, sharpe: float | None) -> float:
        if att.n_trades_resolved < MIN_RESOLVED_FOR_DECISION:
            return 1.0
        pnl_pct = att.realized_pnl / self.nav_reference
        if pnl_pct <= PNL_PCT_KILL:
            return 0.0
        if pnl_pct <= PNL_PCT_HALF:
            return 0.5
        if sharpe is not None and sharpe < SHARPE_FLOOR:
            return 0.5
        return 1.0

    async def refresh(self) -> dict[str, StrategyMetric]:
        since = time.time() - WINDOW_SEC
        # Sync sqlite read — push to thread pool to avoid blocking the event loop.
        atts = await asyncio.to_thread(attribute_pnl, self.db_path, since)
        new_metrics: dict[str, StrategyMetric] = {}
        new_mults: dict[str, float] = {}
        for strat, att in atts.items():
            sharpe = self._sharpe(att.daily_pnl)
            mult = self._decide_mult(att, sharpe)
            metric = StrategyMetric(
                strategy=strat,
                n_resolved=att.n_trades_resolved,
                realized_pnl=att.realized_pnl,
                pnl_pct_of_nav=att.realized_pnl / self.nav_reference if self.nav_reference else 0.0,
                n_days=len(att.daily_pnl),
                daily_sharpe=sharpe,
                multiplier=mult,
            )
            new_metrics[strat] = metric
            new_mults[strat] = mult
            log.info(
                "strategy_metric",
                strategy=strat,
                n_resolved=metric.n_resolved,
                realized_pnl=round(metric.realized_pnl, 2),
                pnl_pct=round(metric.pnl_pct_of_nav * 100, 3),
                n_days=metric.n_days,
                sharpe=round(sharpe, 2) if sharpe is not None else None,
                mult=mult,
            )
        self.multipliers = new_mults
        self.last_metrics = new_metrics
        self.last_refresh_ts = time.time()
        self._save_persisted()
        return new_metrics

    async def run(self, interval_sec: float = 300.0) -> None:
        log.info("throttler_start", interval_sec=interval_sec, nav_ref=self.nav_reference)
        # Restore prior multipliers so kills persist across restarts
        self.load_persisted()
        # Initial refresh
        try:
            await self.refresh()
        except Exception as e:
            log.warning("throttler_refresh_error", err=str(e))
        while True:
            await asyncio.sleep(interval_sec)
            try:
                await self.refresh()
            except Exception as e:
                log.warning("throttler_refresh_error", err=str(e))
