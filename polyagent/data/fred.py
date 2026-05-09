"""FRED ingest — pulls latest observations for high-impact macro series."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

# Series ID -> friendly name. Markets-relevant macro releases.
SERIES: dict[str, str] = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI All Urban (NSA)",
    "CPILFESL": "Core CPI (NSA)",
    "PAYEMS": "Total Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    "GDP": "Real GDP",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS10": "10-Year Treasury Yield",
    "DTB3": "3-Month Treasury Bill",
    "T10Y2Y": "10Y-2Y Yield Spread",
    "VIXCLS": "VIX",
    "DCOILWTICO": "WTI Crude",
}

API = "https://api.stlouisfed.org/fred/series/observations"


async def _fetch_latest(session: aiohttp.ClientSession, series_id: str) -> dict | None:
    if not settings.fred_api_key:
        return None
    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "2",
    }
    try:
        async with session.get(API, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                log.warning("fred_http", series=series_id, status=r.status)
                return None
            data = await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("fred_error", series=series_id, err=str(e))
        return None

    obs = data.get("observations") or []
    if not obs:
        return None
    latest = obs[0]
    prior = obs[1] if len(obs) > 1 else None
    if latest.get("value") in (".", None, ""):
        return None
    return {"latest": latest, "prior": prior}


async def run(queue: asyncio.Queue) -> None:
    if not settings.fred_api_key:
        log.warning("fred_disabled_no_key")
        await asyncio.Event().wait()
        return
    log.info("fred_start", n_series=len(SERIES), poll_sec=settings.fred_poll_sec)

    last_seen: dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        while True:
            for series_id, name in SERIES.items():
                obs = await _fetch_latest(session, series_id)
                if not obs:
                    continue
                latest = obs["latest"]
                prior = obs.get("prior")
                date = latest.get("date", "")
                if last_seen.get(series_id) == date:
                    continue
                first_run = series_id not in last_seen
                last_seen[series_id] = date

                if first_run:
                    # Don't flood the queue on cold start with months-old prints.
                    continue

                try:
                    val = float(latest["value"])
                    pval = float(prior["value"]) if prior and prior.get("value") not in (".", None, "") else None
                except (TypeError, ValueError):
                    continue
                delta = (val - pval) if pval is not None else None

                title = f"FRED {series_id} ({name}): {val} on {date}"
                body_parts = [title]
                if delta is not None:
                    body_parts.append(f"prior: {pval} ({prior['date']}); delta: {delta:+.4f}")
                body = ". ".join(body_parts)

                evt = NewsEvent(
                    source="fred",
                    title=title,
                    body=body,
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    ts=time.time(),
                    extra={
                        "series_id": series_id,
                        "name": name,
                        "date": date,
                        "value": val,
                        "prior_value": pval,
                        "delta": delta,
                    },
                )
                await queue.put(evt)

            await asyncio.sleep(settings.fred_poll_sec)
