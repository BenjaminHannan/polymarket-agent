"""Backfill the resolutions table with historical Polymarket markets from Gamma.

The Polymarket Graph hosted-service subgraph was deprecated in 2024; Gamma's
own /markets endpoint with closed=true serves the same purpose for our needs:
we just need (condition_id, question, yes_won, end_date, outcome_prices).

Usage:
    python -m scripts.backfill_resolutions [--max 5000] [--page 500]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import aiohttp
import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.gamma import _parse_json_field
from polyagent.news_store import NewsStore  # reuses sqlite connection helper layout
import aiosqlite
from pathlib import Path

log = logging_setup.configure()


def _is_clean_resolution(prices: list[float]) -> bool:
    if len(prices) != 2:
        return False
    a, b = prices[0], prices[1]
    return (a >= 0.99 and b <= 0.01) or (b >= 0.99 and a <= 0.01)


def _yes_won(outcomes: list[str], prices: list[float]) -> bool | None:
    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() == "yes"), None)
    if yes_idx is None or yes_idx >= len(prices):
        return None
    return prices[yes_idx] >= 0.99


async def _fetch_page(session: aiohttp.ClientSession, offset: int, limit: int) -> list[dict]:
    url = f"{settings.gamma_url}/markets"
    params = {
        "closed": "true",
        "archived": "false",
        "limit": str(limit),
        "offset": str(offset),
        "order": "endDate",
        "ascending": "false",
    }
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        data = await r.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    return []


async def backfill(max_markets: int = 5000, page_size: int = 500) -> dict:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(settings.db_path)
    # Make sure the resolutions table exists.
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS resolutions (
            condition_id TEXT PRIMARY KEY,
            resolved_ts REAL,
            yes_won INTEGER,
            yes_token_id TEXT,
            no_token_id TEXT,
            yes_size REAL,
            no_size REAL,
            yes_avg_cost REAL,
            no_avg_cost REAL,
            yes_payout REAL,
            no_payout REAL,
            pnl REAL,
            detail TEXT
        );
        CREATE INDEX IF NOT EXISTS resolutions_ts ON resolutions(resolved_ts);
        """
    )
    await db.commit()

    inserted = 0
    skipped_dirty = 0
    skipped_no_tokens = 0
    skipped_dupes = 0
    pages = 0

    async with aiohttp.ClientSession() as session:
        offset = 0
        while offset < max_markets:
            try:
                rows = await _fetch_page(session, offset, page_size)
            except Exception as e:
                log.warning("backfill_fetch_error", offset=offset, err=str(e))
                await asyncio.sleep(2)
                continue
            pages += 1
            if not rows:
                break
            for m in rows:
                cid = m.get("conditionId") or m.get("condition_id") or ""
                if not cid:
                    continue
                outcomes = _parse_json_field(m.get("outcomes")) or []
                prices_raw = _parse_json_field(m.get("outcomePrices")) or []
                try:
                    prices = [float(x) for x in prices_raw]
                except (TypeError, ValueError):
                    prices = []
                tokens = _parse_json_field(m.get("clobTokenIds")) or []
                if len(tokens) != 2:
                    skipped_no_tokens += 1
                    continue
                if not _is_clean_resolution(prices):
                    skipped_dirty += 1
                    continue
                yw = _yes_won(outcomes, prices)
                if yw is None:
                    skipped_dirty += 1
                    continue
                yes_idx = next(
                    (i for i, o in enumerate(outcomes) if str(o).strip().lower() == "yes"), 0
                )
                no_idx = 1 - yes_idx
                yes_token = str(tokens[yes_idx])
                no_token = str(tokens[no_idx])

                end_date = m.get("endDate") or m.get("closedTime") or m.get("end_date_iso")
                resolved_ts = 0.0
                if end_date:
                    try:
                        from datetime import datetime
                        resolved_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        resolved_ts = 0.0
                if resolved_ts == 0.0:
                    resolved_ts = time.time()

                # Capture NegRisk + event_id (used for combinatorial-arb
                # detection and lambdarank query grouping at training time).
                events_arr = m.get("events") or []
                ev_id: str | None = None
                ev_slug: str | None = None
                if isinstance(events_arr, list) and events_arr:
                    first = events_arr[0]
                    if isinstance(first, dict):
                        ev_id = str(first.get("id") or first.get("event_id") or "") or None
                        ev_slug = first.get("slug") or first.get("ticker")
                elif m.get("eventId") or m.get("event_id"):
                    ev_id = str(m.get("eventId") or m.get("event_id"))
                detail = {
                    "question": m.get("question", ""),
                    "category": m.get("category"),
                    "yes_price": prices[yes_idx],
                    "no_price": prices[no_idx],
                    "liquidity": m.get("liquidityNum") or m.get("liquidity"),
                    "volume": m.get("volumeNum") or m.get("volume"),
                    "end_date": end_date,
                    "neg_risk": bool(m.get("negRisk") or False),
                    "event_id": ev_id,
                    "event_slug": ev_slug,
                    "source": "gamma_backfill",
                }

                cur = await db.execute(
                    """INSERT OR IGNORE INTO resolutions(
                        condition_id, resolved_ts, yes_won,
                        yes_token_id, no_token_id,
                        yes_size, no_size, yes_avg_cost, no_avg_cost,
                        yes_payout, no_payout, pnl, detail
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        cid,
                        resolved_ts,
                        1 if yw else 0,
                        yes_token,
                        no_token,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0 if yw else 0.0,
                        0.0 if yw else 1.0,
                        0.0,
                        json.dumps(detail),
                    ),
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped_dupes += 1
            await db.commit()
            log.info(
                "backfill_progress",
                page=pages,
                offset=offset,
                inserted=inserted,
                dirty=skipped_dirty,
                no_tokens=skipped_no_tokens,
                dupes=skipped_dupes,
            )
            offset += len(rows)
            if len(rows) < page_size:
                break
            # Be polite to the public API.
            await asyncio.sleep(0.5)

    await db.close()
    return {
        "inserted": inserted,
        "skipped_dirty": skipped_dirty,
        "skipped_no_tokens": skipped_no_tokens,
        "skipped_dupes": skipped_dupes,
        "pages": pages,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=5000)
    p.add_argument("--page", type=int, default=500)
    args = p.parse_args()
    summary = asyncio.run(backfill(max_markets=args.max, page_size=args.page))
    log.info("backfill_done", **summary)


if __name__ == "__main__":
    main()
