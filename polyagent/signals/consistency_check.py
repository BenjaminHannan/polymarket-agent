"""Karkare et al. 2024 NegRisk consistency check (runtime task).

For each NegRisk event with N>=2 mutually-exclusive outcomes, elicit
P(YES) from the LLM independently for each outcome and check whether the
sum is close to 1. Big deviations from 1 are evidence the LLM is
inconsistent on this event — its forecasts should be discounted in the
combiner.

Outputs:
  - persisted signal rows (strategy="consistency_check") with the per-event
    sum / deviation / per-outcome ps. Useful for offline inspection.
  - in-memory `state[event_id] = {"deviation": float, "ts": float}` that
    `combined.py` reads to downweight the llm_forecaster expert when the
    event is inconsistent.

Cost guard: the LLM is expensive. We only run on the M most-trafficked
NegRisk events per cycle, and we cache results for ``ttl_sec`` so we
don't pay the cost again until news activity moves on. Disabled if
ENABLE_LLM_FORECASTER != 1.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from polyagent.gamma import Market
from polyagent.models.llm_forecaster import LLMForecaster
from polyagent.news_store import NewsStore

log = structlog.get_logger()


@dataclass
class ConsistencyCheck:
    markets: list[Market]
    news_store: NewsStore
    llm_forecaster: LLMForecaster
    poll_sec: float = 1800.0  # 30 min — LLM cost is high
    ttl_sec: float = 3600.0   # cache per-event for 1h
    max_events_per_cycle: int = 4
    article_window_days: float = 7.0
    article_limit: int = 6

    # event_id -> {"deviation": float, "sum": float, "ts": float, "n": int}
    state: dict = field(default_factory=dict)

    def _negrisk_groups(self) -> dict[str, list[Market]]:
        groups: dict[str, list[Market]] = {}
        for m in self.markets:
            if m.neg_risk and m.event_id:
                groups.setdefault(m.event_id, []).append(m)
        return {eid: g for eid, g in groups.items() if len(g) >= 2}

    async def _articles_for(self, asof: float) -> list[str]:
        """Pull recent news titles bounded by asof for retrieval. We use the
        same asof-clean discipline as the combined signaler — only news
        published <= asof is fed to the LLM."""
        if self.news_store.db is None:
            return []
        try:
            cutoff = asof - self.article_window_days * 86400
            async with self.news_store.db.execute(
                "SELECT title FROM news WHERE ts <= ? AND ts >= ? "
                "ORDER BY ts DESC LIMIT ?",
                (asof, cutoff, self.article_limit),
            ) as cur:
                return [row[0] async for row in cur if row and row[0]]
        except Exception:
            return []

    def deviation_for(self, event_id: str) -> float | None:
        """Lookup helper for downstream consumers (combined.py). Returns
        None if the event hasn't been checked recently enough."""
        rec = self.state.get(event_id)
        if rec is None:
            return None
        if time.time() - rec["ts"] > self.ttl_sec * 2:
            return None
        return float(rec["deviation"])

    async def _check_one(self, event_id: str, members: list[Market]) -> None:
        # Skip if cached and fresh
        rec = self.state.get(event_id)
        if rec is not None and (time.time() - rec["ts"]) < self.ttl_sec:
            return
        asof = time.time()
        articles = await self._articles_for(asof)

        # Run the (sync, blocking) consistency_score off the event loop.
        questions = [m.question for m in members]
        per_market_articles = [articles for _ in members]
        try:
            res = await asyncio.to_thread(
                self.llm_forecaster.consistency_score,
                questions,
                per_market_articles,
            )
        except Exception as e:
            log.warning("consistency_check_error", event_id=event_id, err=str(e))
            return
        if res is None:
            return

        ts = time.time()
        self.state[event_id] = {
            "deviation": float(res["deviation"]),
            "sum": float(res["sum"]),
            "ts": ts,
            "n": len(members),
        }
        log.info(
            "consistency_check",
            event_id=event_id,
            n_outcomes=len(members),
            sum_yes=round(float(res["sum"]), 3),
            deviation=round(float(res["deviation"]), 3),
        )
        # Persist as a signal row keyed to the first member's condition_id.
        # We log a single row per event with the event_id in detail so
        # offline analysis can reconstruct the group.
        try:
            await self.news_store.insert_signal(
                strategy="consistency_check",
                condition_id=members[0].condition_id,
                direction="info",
                score=float(res["deviation"]),
                news_hash="",
                detail={
                    "event_id": event_id,
                    "sum_yes": round(float(res["sum"]), 4),
                    "deviation": round(float(res["deviation"]), 4),
                    "n_outcomes": len(members),
                    "ps": [round(p, 4) for p in res["ps"]],
                    "questions": [m.question[:120] for m in members],
                },
            )
        except Exception as e:
            log.warning("consistency_check_signal_insert_failed", err=str(e))

    async def run(self) -> None:
        if not self.llm_forecaster.is_enabled():
            log.info("consistency_check_disabled_llm_off")
            await asyncio.Event().wait()
            return
        groups = self._negrisk_groups()
        if not groups:
            log.info("consistency_check_no_negrisk_groups")
            await asyncio.Event().wait()
            return
        log.info(
            "consistency_check_start",
            n_negrisk_events=len(groups),
            poll_sec=self.poll_sec,
            ttl_sec=self.ttl_sec,
        )
        # Order events by total liquidity so the most-trafficked get checked
        # first. Cheap proxy — sum member liquidity.
        ordered = sorted(
            groups.items(),
            key=lambda kv: -sum((m.liquidity or 0.0) for m in kv[1]),
        )
        idx = 0
        while True:
            await asyncio.sleep(self.poll_sec)
            batch = ordered[idx : idx + self.max_events_per_cycle]
            if not batch:
                idx = 0
                continue
            for event_id, members in batch:
                try:
                    await self._check_one(event_id, members)
                except Exception as e:
                    log.warning("consistency_check_loop_error", err=str(e))
            idx += self.max_events_per_cycle
            if idx >= len(ordered):
                idx = 0
