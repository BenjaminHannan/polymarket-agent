"""CLOB /prices-history client for historical price retrieval.

Endpoint:
    GET https://clob.polymarket.com/prices-history?market={token_id}&interval=...&fidelity=...

Common interval/fidelity combos:
    interval=1d  fidelity=60   -> 1 day of 1-min candles
    interval=1w  fidelity=300  -> 1 week of 5-min candles
    interval=max fidelity=3600 -> entire market lifetime, hourly
"""

from __future__ import annotations

import asyncio
from typing import Iterable

import aiohttp
import structlog

from polyagent.config import settings

log = structlog.get_logger()


async def fetch_history(
    session: aiohttp.ClientSession,
    token_id: str,
    interval: str = "max",
    fidelity: int = 3600,
) -> list[dict]:
    """Returns a list of {t: timestamp, p: price} dicts. Empty list on error."""
    url = f"{settings.clob_url}/prices-history"
    params = {"market": token_id, "interval": interval, "fidelity": str(fidelity)}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                return []
            data = await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return []
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    if isinstance(data, list):
        return data
    return []


async def fetch_history_batch(
    token_ids: Iterable[str],
    interval: str = "max",
    fidelity: int = 3600,
    concurrency: int = 8,
) -> dict[str, list[dict]]:
    """Batched fetcher with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, list[dict]] = {}

    async def _one(session, tid):
        async with sem:
            out[tid] = await fetch_history(session, tid, interval=interval, fidelity=fidelity)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(_one(session, tid) for tid in token_ids))
    return out
