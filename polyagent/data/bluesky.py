"""Bluesky Jetstream public firehose consumer (no auth).

Captures posts from a watchlist of accounts (politicians, journalists, beat
reporters). The DID list is loaded from data/bluesky_watchlist.txt — one DID
per line, comments with #. If empty, the firehose is still consumed but only
posts from listed authors are emitted; no DIDs means no events.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from polyagent.config import ROOT
from polyagent.news_store import NewsEvent

log = structlog.get_logger()

JETSTREAM_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
)

WATCHLIST_PATH = ROOT / "data" / "bluesky_watchlist.txt"


def _load_watchlist() -> set[str]:
    if not WATCHLIST_PATH.exists():
        return set()
    out: set[str] = set()
    for line in WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


async def run(queue: asyncio.Queue) -> None:
    watchlist = _load_watchlist()
    if not watchlist:
        log.info(
            "bluesky_watchlist_empty",
            note="add DIDs to data/bluesky_watchlist.txt to enable; ingest paused",
        )
        # Sleep forever rather than return, so the supervisor doesn't shut down.
        await asyncio.Event().wait()
        return

    log.info("bluesky_start", watchlist_size=len(watchlist))
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(JETSTREAM_URL, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                async for raw in ws:
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("kind") != "commit":
                        continue
                    commit = evt.get("commit") or {}
                    if commit.get("operation") != "create":
                        continue
                    did = evt.get("did", "")
                    if did not in watchlist:
                        continue
                    record = commit.get("record") or {}
                    text = (record.get("text") or "").strip()
                    if not text:
                        continue
                    rkey = commit.get("rkey", "")
                    url = f"https://bsky.app/profile/{did}/post/{rkey}" if rkey else ""
                    ts_us = evt.get("time_us")
                    ts = float(ts_us) / 1e6 if ts_us else None
                    news_evt = NewsEvent(
                        source="bluesky",
                        title=text[:280],
                        body=text,
                        url=url,
                        ts=ts or 0.0,
                        extra={"did": did, "rkey": rkey},
                    )
                    if news_evt.ts == 0.0:
                        import time as _time
                        news_evt.ts = _time.time()
                    await queue.put(news_evt)
        except (ConnectionClosed, WebSocketException, OSError, asyncio.TimeoutError) as e:
            log.warning("bluesky_disconnect", err=str(e), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
