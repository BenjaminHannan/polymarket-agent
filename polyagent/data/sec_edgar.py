"""SEC EDGAR ingest — pulls the latest filings RSS.

No API key, but the SEC requires every request to identify itself with a
User-Agent header containing a real email. See
https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import feedparser
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

# Recent 8-K filings, all companies. Focused on "current report" form because
# 8-Ks are where M&A, executive changes, material agreements land.
EDGAR_FEED = (
    "https://www.sec.gov/cgi-bin/browse-edgar?"
    "action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom"
)


async def run(queue: asyncio.Queue) -> None:
    log.info("sec_edgar_start", poll_sec=settings.sec_edgar_poll_sec)
    headers = {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept": "application/atom+xml, application/xml;q=0.9",
    }
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    EDGAR_FEED, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        log.warning("sec_edgar_http", status=r.status)
                        await asyncio.sleep(settings.sec_edgar_poll_sec)
                        continue
                    body = await r.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("sec_edgar_error", err=str(e))
                await asyncio.sleep(settings.sec_edgar_poll_sec)
                continue

            feed = feedparser.parse(body)
            for entry in feed.entries[:40]:
                title = (entry.get("title") or "").strip()
                summary = (entry.get("summary") or "").strip()
                link = entry.get("link") or ""
                published_parsed = entry.get("updated_parsed") or entry.get("published_parsed")
                ts = time.mktime(published_parsed) if published_parsed else time.time()
                evt = NewsEvent(
                    source="sec_edgar",
                    title=title[:280],
                    body=summary[:4000],
                    url=link,
                    ts=ts,
                    extra={"form": "8-K"},
                )
                await queue.put(evt)

            await asyncio.sleep(settings.sec_edgar_poll_sec)
