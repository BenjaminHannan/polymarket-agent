"""RSS news aggregator. Polls a curated set of free wire feeds."""

from __future__ import annotations

import asyncio
import time

import aiohttp
import feedparser
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

# Curated free feeds from §3.2 of v2 blueprint.
FEEDS: list[tuple[str, str]] = [
    ("reuters_world", "https://feeds.reuters.com/reuters/worldNews"),
    ("ap_top", "https://apnews.com/index.rss"),
    ("ap_politics", "https://apnews.com/hub/politics?utm_source=apnews&output=rss"),
    ("bbc_news", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("aljazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("npr_news", "https://feeds.npr.org/1001/rss.xml"),
    ("npr_politics", "https://feeds.npr.org/1014/rss.xml"),
    ("cnbc_top", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("cnbc_business", "https://www.cnbc.com/id/10001147/device/rss/rss.html"),
    ("politico", "https://rss.politico.com/politics-news.xml"),
    ("thehill", "https://thehill.com/feed/"),
    ("guardian_world", "https://www.theguardian.com/world/rss"),
    ("fed_press", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("fed_monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("treasury", "https://home.treasury.gov/rss/press.xml"),
    ("federal_register", "https://www.federalregister.gov/api/v1/documents.rss?per_page=100&order=newest"),
    ("sec_press", "https://www.sec.gov/news/pressreleases.rss"),
    ("scotusblog", "https://www.scotusblog.com/feed/"),
    ("ecb_press", "https://www.ecb.europa.eu/rss/press.xml"),
    ("imf_news", "https://www.imf.org/en/News/RSS?Language=ENG&series=News+Articles"),
    ("whitehouse_briefings", "https://www.whitehouse.gov/briefing-room/feed/"),
]


def _entry_to_event(name: str, entry) -> NewsEvent:
    title = (entry.get("title") or "").strip()
    summary = (entry.get("summary") or "").strip()
    link = entry.get("link") or entry.get("id") or ""
    published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if published_parsed:
        ts = time.mktime(published_parsed)
    else:
        ts = time.time()
    return NewsEvent(
        source=f"rss:{name}",
        title=title,
        body=summary[:4000],
        url=link,
        ts=ts,
        extra={"feed": name},
    )


async def _poll_one(session: aiohttp.ClientSession, name: str, url: str, queue: asyncio.Queue) -> None:
    headers = {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.5",
    }
    while True:
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as r:
                body = await r.text()
            feed = feedparser.parse(body)
            n_pushed = 0
            for entry in feed.entries[:50]:
                evt = _entry_to_event(name, entry)
                if not evt.title:
                    continue
                await queue.put(evt)
                n_pushed += 1
            if n_pushed:
                log.debug("rss_poll", feed=name, n=n_pushed)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("rss_error", feed=name, err=str(e))
        except Exception as e:
            log.warning("rss_unexpected", feed=name, err=str(e))
        await asyncio.sleep(settings.rss_poll_sec)


async def run(queue: asyncio.Queue) -> None:
    log.info("rss_start", n_feeds=len(FEEDS), poll_sec=settings.rss_poll_sec)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(_poll_one(session, n, u, queue) for n, u in FEEDS))
