"""Congress.gov API ingest — recent bill activity for the current Congress."""

from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

# 119th Congress runs 2025-01-03 to 2027-01-03; switch to 120 after.
CURRENT_CONGRESS = 119

API_BASE = "https://api.congress.gov/v3"


async def _fetch(session: aiohttp.ClientSession, path: str, params: dict | None = None) -> dict | None:
    if not settings.congress_api_key:
        return None
    p = dict(params or {})
    p["api_key"] = settings.congress_api_key
    p.setdefault("format", "json")
    try:
        async with session.get(
            f"{API_BASE}/{path.lstrip('/')}",
            params=p,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            if r.status != 200:
                log.warning("congress_http", path=path, status=r.status)
                return None
            return await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("congress_error", path=path, err=str(e))
        return None


async def run(queue: asyncio.Queue) -> None:
    if not settings.congress_api_key:
        log.warning("congress_disabled_no_key")
        await asyncio.Event().wait()
        return
    log.info("congress_start", congress=CURRENT_CONGRESS, poll_sec=settings.congress_poll_sec)

    async with aiohttp.ClientSession() as session:
        while True:
            data = await _fetch(
                session,
                f"bill/{CURRENT_CONGRESS}",
                {"sort": "updateDate desc", "limit": 50},
            )
            if data:
                for bill in (data.get("bills") or []):
                    number = bill.get("number") or ""
                    bill_type = (bill.get("type") or "").lower()
                    title = bill.get("title") or ""
                    latest = bill.get("latestAction") or {}
                    action_text = latest.get("text") or ""
                    action_date = latest.get("actionDate") or ""
                    url = bill.get("url") or ""
                    full_title = f"{bill_type.upper()} {number}: {title}"
                    body = f"{full_title}. Latest action ({action_date}): {action_text}"
                    evt = NewsEvent(
                        source="congress",
                        title=full_title[:280],
                        body=body[:4000],
                        url=url,
                        ts=time.time(),
                        extra={
                            "congress": CURRENT_CONGRESS,
                            "type": bill_type,
                            "number": number,
                            "action_date": action_date,
                        },
                    )
                    await queue.put(evt)
            await asyncio.sleep(settings.congress_poll_sec)
