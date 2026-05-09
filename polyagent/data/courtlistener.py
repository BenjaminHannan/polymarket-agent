"""CourtListener ingest — recent federal court opinions and dockets.

API docs: https://www.courtlistener.com/help/api/rest/
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

API_BASE = "https://www.courtlistener.com/api/rest/v4"


async def _fetch(session: aiohttp.ClientSession, path: str, params: dict | None = None) -> dict | None:
    if not settings.courtlistener_api_key:
        return None
    headers = {"Authorization": f"Token {settings.courtlistener_api_key}"}
    try:
        async with session.get(
            f"{API_BASE}/{path.lstrip('/')}",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            if r.status != 200:
                log.warning("courtlistener_http", path=path, status=r.status)
                return None
            return await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("courtlistener_error", path=path, err=str(e))
        return None


async def run(queue: asyncio.Queue) -> None:
    if not settings.courtlistener_api_key:
        log.warning("courtlistener_disabled_no_key")
        await asyncio.Event().wait()
        return
    log.info("courtlistener_start", poll_sec=settings.courtlistener_poll_sec)

    async with aiohttp.ClientSession() as session:
        while True:
            data = await _fetch(
                session,
                "opinions/",
                {"order_by": "-date_created", "page_size": 25},
            )
            if data:
                for op in (data.get("results") or []):
                    case_name = op.get("case_name") or op.get("caseName") or "Unknown case"
                    court = op.get("court_id") or op.get("court") or ""
                    date_filed = op.get("date_filed") or ""
                    abs_url = op.get("absolute_url") or ""
                    full_url = (
                        f"https://www.courtlistener.com{abs_url}"
                        if abs_url and abs_url.startswith("/")
                        else abs_url
                    )
                    title = f"{case_name} ({court}, {date_filed})"
                    snippet = (op.get("plain_text") or op.get("html") or "")[:1500]
                    evt = NewsEvent(
                        source="courtlistener",
                        title=title[:280],
                        body=(title + "\n" + snippet)[:4000],
                        url=full_url,
                        ts=time.time(),
                        extra={"court": court, "date_filed": date_filed},
                    )
                    await queue.put(evt)
            await asyncio.sleep(settings.courtlistener_poll_sec)
