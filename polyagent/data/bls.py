"""BLS Public Data API v2 ingest — labor and inflation series.

Free tier with key: 500 queries/day, up to 50 series per request.
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

# BLS series IDs of market interest.
SERIES: dict[str, str] = {
    "LNS14000000": "Unemployment Rate (Seasonally Adjusted)",
    "CES0000000001": "Total Nonfarm Employment (NFP)",
    "CUUR0000SA0": "CPI-U All Items (NSA)",
    "CUSR0000SA0L1E": "Core CPI (Seasonally Adjusted)",
    "WPSFD49207": "PPI Final Demand",
    "LNS11300000": "Labor Force Participation Rate",
}

API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


async def run(queue: asyncio.Queue) -> None:
    if not settings.bls_api_key:
        log.warning("bls_disabled_no_key")
        await asyncio.Event().wait()
        return
    log.info("bls_start", n_series=len(SERIES), poll_sec=settings.bls_poll_sec)

    last_seen: dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        while True:
            payload = {
                "seriesid": list(SERIES.keys()),
                "startyear": str(time.localtime().tm_year - 1),
                "endyear": str(time.localtime().tm_year),
                "registrationkey": settings.bls_api_key,
            }
            try:
                async with session.post(
                    API,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status != 200:
                        log.warning("bls_http", status=r.status)
                        await asyncio.sleep(settings.bls_poll_sec)
                        continue
                    data = await r.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("bls_error", err=str(e))
                await asyncio.sleep(settings.bls_poll_sec)
                continue

            if data.get("status") != "REQUEST_SUCCEEDED":
                log.warning("bls_request_failed", msgs=data.get("message"))
                await asyncio.sleep(settings.bls_poll_sec)
                continue

            for series in (data.get("Results") or {}).get("series") or []:
                series_id = series.get("seriesID", "")
                name = SERIES.get(series_id, series_id)
                obs = series.get("data") or []
                if not obs:
                    continue
                latest = obs[0]
                period_key = f"{latest.get('year','')}-{latest.get('period','')}"
                if last_seen.get(series_id) == period_key:
                    continue
                first_run = series_id not in last_seen
                last_seen[series_id] = period_key
                if first_run:
                    continue
                try:
                    val = float(latest.get("value", "nan"))
                except ValueError:
                    continue
                title = f"BLS {series_id} ({name}): {val} for {latest.get('periodName','?')} {latest.get('year','')}"
                evt = NewsEvent(
                    source="bls",
                    title=title,
                    body=title,
                    url=f"https://data.bls.gov/timeseries/{series_id}",
                    ts=time.time(),
                    extra={
                        "series_id": series_id,
                        "name": name,
                        "period": period_key,
                        "value": val,
                    },
                )
                await queue.put(evt)

            await asyncio.sleep(settings.bls_poll_sec)
