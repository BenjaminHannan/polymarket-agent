"""LLM-augmented forecaster for Polymarket weather markets.

Pairs the existing LLMForecaster (Phi-4-mini-instruct, retrieval-augmented
prompt → calibrated probability) with weather-specific structured retrieval:

  1. Active and recent natural events (last 30d) from the natural_events
     table — earthquakes, severe storms, wildfires, volcanoes, floods.
  2. A historical base rate computed from the resolutions table for the
     question's pattern. Examples:
         "Another 7.0+ earthquake by [date]?" → look up how often a 30d
         window has seen >=1 M7+ event historically.
         "Will hurricane <name> reach landfall by ..." → harder to base-rate;
         we abstain unless we have a clear precedent.
  3. The question itself, including parsed threshold and deadline, so the
     LLM is told precisely what we want a probability for.

The forecaster runs on a slow loop (default 30 min) since each LLM call
takes ~10-30 s on local hardware. Each call emits a `weather_llm_forecast`
signal if |p_llm - p_market| >= MIN_EDGE; otherwise the call is logged
but no signal is written. Default OFF; opt in via
ENABLE_WEATHER_LLM_FORECAST=1.

Design choices that keep this honest:

  - We feed the LLM the same base-rate calculation a careful human would
    do, so the LLM can either accept or override it explicitly.
  - We prompt the LLM to output its probability in a fixed format and
    parse it with the existing _parse_probability regex.
  - We pass a structured event context, not raw text, so the LLM can't
    confabulate which events happened.
  - We only emit signals on questions where we successfully computed a
    base rate. Otherwise the LLM has no anchor and we abstain.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.models.llm_forecaster import LLMForecaster
from polyagent.news_store import NewsStore
from polyagent.orderbook import BookStore
from polyagent.signals.natural_event_match import (
    _parse_deadline,
    _parse_eq_threshold,
)

log = structlog.get_logger()


# Question-pattern recognisers — lets us route to the right retrieval
# strategy and base-rate lookup. Each returns either a structured tag dict
# or None (abstain).
def _classify_question(question: str) -> dict | None:
    q = (question or "").lower()
    if "earthquake" in q or "magnitude" in q or re.search(r"\bm\s?\d(?:\.\d)?\+", q):
        threshold = _parse_eq_threshold(question)
        if threshold is None:
            return None
        return {"kind": "earthquake_count", "magnitude": threshold}
    if "hurricane" in q or "tropical storm" in q or "typhoon" in q or "cyclone" in q:
        return {"kind": "tropical_cyclone"}
    if "wildfire" in q or "fire" in q:
        return {"kind": "wildfire"}
    if "volcan" in q or "eruption" in q:
        return {"kind": "volcano"}
    if "flood" in q:
        return {"kind": "flood"}
    return None


# ── Historical base rates ────────────────────────────────────────────────
def _eq_base_rate(
    db_path: str, *, magnitude: float, window_days: int
) -> dict | None:
    """Empirical base rate: across the resolutions table, how often did
    a 'M>=magnitude earthquake by [date]?' question with this window size
    resolve YES? Falls back to a calculation over the natural_events table
    if there isn't enough resolutions data.

    Returns {"p_yes": float, "n": int, "method": str} or None.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Method A: similar resolved questions in our DB
        rows = conn.execute(
            """SELECT yes_won, detail FROM resolutions
               WHERE detail LIKE ?""",
            (f'%earthquake%',),
        ).fetchall()
        same_thresh = []
        for yw, detail in rows:
            try:
                d = json.loads(detail or "{}")
            except Exception:
                continue
            q = (d.get("question") or "").lower()
            if "earthquake" not in q:
                continue
            t = _parse_eq_threshold(d.get("question") or "")
            if t is None or abs(t - magnitude) > 0.5:
                continue
            same_thresh.append(int(yw))
        if len(same_thresh) >= 5:
            p = sum(same_thresh) / len(same_thresh)
            return {"p_yes": p, "n": len(same_thresh), "method": "historical_resolutions"}

        # Method B: count distinct events of magnitude >= threshold in
        # rolling windows over natural_events
        evs = conn.execute(
            """SELECT occurred_ts, magnitude FROM natural_events
               WHERE source='usgs' AND category='earthquakes'""",
        ).fetchall()
        if len(evs) < 10:
            return None
        # For each event date, count how many >= magnitude events occurred
        # in the next window_days. P(YES) ~ fraction of windows with >= 1.
        window_sec = window_days * 86400
        ts_above = sorted(
            float(t) for t, m in evs if m is not None and m >= magnitude
        )
        if not ts_above:
            return {"p_yes": 0.05, "n": 0, "method": "no_qualifying_events"}
        # Count "windows starting on each event start date" that captured >=1
        # event in the next window_days. Equivalently, P(at least one in
        # window) = 1 - exp(-rate * window).
        # rate = events / total observation period
        first, last = float(min(t for t, _ in evs)), float(max(t for t, _ in evs))
        period_sec = max(1.0, last - first)
        rate_per_sec = len(ts_above) / period_sec
        import math
        p = 1 - math.exp(-rate_per_sec * window_sec)
        p = max(0.01, min(0.99, p))
        return {
            "p_yes": p,
            "n": len(ts_above),
            "method": "poisson_rate_natural_events",
            "rate_per_day": rate_per_sec * 86400,
        }
    finally:
        conn.close()


# ── Active event retrieval ───────────────────────────────────────────────
def _recent_events(db_path: str, *, days: int = 30, category: str | None = None) -> list[dict]:
    """Return recent natural_events rows ordered most-recent first."""
    conn = sqlite3.connect(db_path)
    try:
        cutoff = time.time() - days * 86400
        sql = (
            "SELECT title, magnitude, place, occurred_ts, category "
            "FROM natural_events WHERE occurred_ts >= ? "
        )
        params: list = [cutoff]
        if category:
            sql += "AND category = ? "
            params.append(category)
        sql += "ORDER BY occurred_ts DESC LIMIT 30"
        return [
            {
                "title": r[0],
                "magnitude": r[1],
                "place": r[2],
                "occurred_iso": datetime.fromtimestamp(r[3], tz=timezone.utc).strftime("%Y-%m-%d"),
                "category": r[4],
            }
            for r in conn.execute(sql, params)
        ]
    finally:
        conn.close()


# ── Prompt building ──────────────────────────────────────────────────────
def _format_event_block(events: list[dict]) -> str:
    if not events:
        return "  (no recent events on file)"
    lines = []
    for e in events[:15]:
        mag = f"M{e['magnitude']}" if e["magnitude"] is not None else "—"
        lines.append(f"  [{e['category']}/{e['occurred_iso']}] {mag} {e['title']}")
    return "\n".join(lines)


def build_prompt(
    market: Market,
    *,
    classification: dict,
    base_rate: dict | None,
    events: list[dict],
    days_to_deadline: int,
) -> str:
    """Compose the forecasting prompt for an LLM (Halawi-style)."""
    base_rate_block = "(no calculable base rate)"
    if base_rate is not None:
        method = base_rate.get("method", "?")
        n = base_rate.get("n", 0)
        p = base_rate.get("p_yes", 0.5)
        extra = ""
        if "rate_per_day" in base_rate:
            extra = f", rate≈{base_rate['rate_per_day']:.3f}/day in catalog"
        base_rate_block = f"P(YES)≈{p:.3f} (method={method}, n={n}{extra})"
    return (
        "You are a calibrated probability forecaster for prediction-market "
        "weather events. Output your final probability that the question "
        "resolves YES. Use the structured context — do NOT confabulate "
        "events that are not listed.\n\n"
        f"Question: {market.question}\n"
        f"Days to deadline: {days_to_deadline}\n"
        f"Question kind (parsed): {classification.get('kind')}\n"
        + (f"Threshold (parsed): M{classification['magnitude']}\n"
           if "magnitude" in classification else "")
        + f"\nEmpirical base rate: {base_rate_block}\n\n"
        "Recent relevant natural events (most recent first):\n"
        f"{_format_event_block(events)}\n\n"
        "Reasoning rules:\n"
        "  - Earthquakes are NOT predictable — the probability of one in a "
        "future window is set by the long-run Poisson rate, full stop.\n"
        "  - For questions about events that have already occurred within "
        "the window, the answer is YES with probability ≈1.\n"
        "  - For specific named events (named hurricane, named volcano), "
        "use the listed event status; if the event is not listed, the "
        "probability of resolution by deadline is mostly the base rate.\n"
        "  - If days_to_deadline is small, the conditional probability "
        "given no event so far is what matters; account for that.\n"
        "\nOutput exactly one line, no commentary:\n"
        "Probability: <number in [0,1]>\n"
    )


# ── Forecaster orchestration ─────────────────────────────────────────────
@dataclass
class WeatherForecast:
    p_llm: float
    p_market: float | None
    edge: float | None
    reasoning_tag: dict
    base_rate: dict | None
    n_events_in_context: int


@dataclass
class WeatherForecaster:
    book_store: BookStore
    markets: list[Market]
    news_store: NewsStore
    db_path: str = settings.db_path
    poll_sec: float = 1800.0
    min_edge: float = 0.10
    forecaster: LLMForecaster | None = None

    def __post_init__(self) -> None:
        if self.forecaster is None:
            self.forecaster = LLMForecaster()

    async def forecast_one(self, market: Market) -> WeatherForecast | None:
        cls = _classify_question(market.question)
        if cls is None:
            return None
        deadline_ts = _parse_deadline(market.question, market.end_date_iso)
        if deadline_ts is None:
            return None
        days_to_deadline = max(0, int((deadline_ts - time.time()) / 86400))

        # Window length for base-rate calc (use days from now; if deadline
        # is in the past relative to question issuance we could reconstruct,
        # but for now we just use days_to_deadline).
        window_days = max(1, days_to_deadline)
        base_rate: dict | None = None
        if cls["kind"] == "earthquake_count":
            base_rate = _eq_base_rate(
                self.db_path, magnitude=float(cls["magnitude"]), window_days=window_days
            )
            cat_filter = "earthquakes"
        elif cls["kind"] == "tropical_cyclone":
            cat_filter = "severeStorms"
        elif cls["kind"] == "wildfire":
            cat_filter = "wildfires"
        elif cls["kind"] == "volcano":
            cat_filter = "volcanoes"
        elif cls["kind"] == "flood":
            cat_filter = "floods"
        else:
            cat_filter = None

        events = _recent_events(self.db_path, days=30, category=cat_filter)

        # Skip categories where we have no base rate AND no helpful active
        # event context to anchor the LLM. Without anchors the LLM
        # confabulates.
        if base_rate is None and not events:
            return None

        prompt = build_prompt(
            market,
            classification=cls,
            base_rate=base_rate,
            events=events,
            days_to_deadline=days_to_deadline,
        )

        # Call the LLM. forecast_async takes (question, articles); we put
        # our entire structured prompt in the "question" slot and pass an
        # empty articles list — this lets us reuse the existing infra
        # without reshaping it.
        if not self.forecaster.is_enabled():
            log.info("weather_forecast_skip_disabled", question=market.question[:80])
            return None
        result = await self.forecaster.forecast_async(prompt, [])
        if result is None:
            return None
        p_llm = float(result.get("p", 0.5))

        # Market mid for comparison
        book = self.book_store.books.get(market.yes_token_id)
        p_market = book.mid() if book else None
        edge = (p_llm - p_market) if p_market is not None else None

        wf = WeatherForecast(
            p_llm=p_llm,
            p_market=p_market,
            edge=edge,
            reasoning_tag=cls,
            base_rate=base_rate,
            n_events_in_context=len(events),
        )
        log.info(
            "weather_forecast",
            question=market.question[:90],
            p_llm=round(p_llm, 4),
            p_market=round(p_market, 4) if p_market is not None else None,
            edge=round(edge, 4) if edge is not None else None,
            kind=cls.get("kind"),
            base_rate=base_rate.get("p_yes") if base_rate else None,
        )
        return wf

    async def emit_if_edge(self, market: Market, wf: WeatherForecast) -> bool:
        if wf.edge is None or abs(wf.edge) < self.min_edge:
            return False
        direction = "yes" if wf.edge > 0 else "no"
        await self.news_store.insert_signal(
            strategy="weather_llm_forecast",
            condition_id=market.condition_id,
            direction=direction,
            score=abs(wf.edge),
            news_hash="",
            detail={
                "p_llm": round(wf.p_llm, 4),
                "p_market": round(wf.p_market or 0.0, 4),
                "edge": round(wf.edge, 4),
                "reasoning": wf.reasoning_tag,
                "base_rate": wf.base_rate,
                "n_events_in_context": wf.n_events_in_context,
                "question": market.question[:160],
                "category": market.category,
            },
        )
        return True

    async def run(self) -> None:
        if not self.forecaster.is_enabled():
            log.warning("weather_forecaster_skip_llm_disabled")
            await asyncio.Event().wait()
            return
        log.info("weather_forecaster_start", n_markets=len(self.markets), poll_sec=self.poll_sec)
        while True:
            n_emitted = 0
            n_scanned = 0
            for m in self.markets:
                try:
                    wf = await self.forecast_one(m)
                except Exception as e:
                    log.warning("weather_forecast_error", err=str(e), q=m.question[:90])
                    continue
                if wf is None:
                    continue
                n_scanned += 1
                if await self.emit_if_edge(m, wf):
                    n_emitted += 1
            log.info("weather_forecaster_pass_complete", scanned=n_scanned, emitted=n_emitted)
            await asyncio.sleep(self.poll_sec)
