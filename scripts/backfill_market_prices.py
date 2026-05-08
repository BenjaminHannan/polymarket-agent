"""Backfill multi-horizon pre-resolution YES prices.

Pulls CLOB /prices-history once per market and extracts the YES mid at four
realistic trade-time horizons before close: 1h, 6h, 24h, 7d. Each lands in a
separate column on signal_outcomes so we can compare which horizon's market
price predicts best (i.e. which horizon's combiner actually generalizes).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

import aiohttp
import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.data.clob_history import fetch_history
from polyagent.models.outcomes import _ensure_table

log = logging_setup.configure()


HORIZONS = {
    "p_market_1h": 1 * 3600,
    "p_market_6h": 6 * 3600,
    "p_market_24h": 24 * 3600,
    "p_market_7d": 7 * 86400,
}


def _price_at_offset(history: list[dict], close_ts: float, offset_sec: int) -> float | None:
    if not history:
        return None
    target = close_ts - offset_sec
    candidate = None
    for pt in history:
        try:
            t = float(pt.get("t", 0))
            p = float(pt.get("p", 0))
        except (TypeError, ValueError):
            continue
        if t <= target:
            candidate = p
        else:
            break
    return candidate  # may be None if market shorter than offset


def _ensure_columns(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)
    conn.close()


async def backfill(db_path: str, max_rows: int | None, concurrency: int, force: bool) -> dict:
    _ensure_columns(db_path)
    conn = sqlite3.connect(db_path)
    where = "" if force else " WHERE p_market_24h IS NULL"
    rows = list(
        conn.execute(
            f"""SELECT s.condition_id, s.resolved_ts, r.yes_token_id
                FROM signal_outcomes s
                JOIN resolutions r ON r.condition_id = s.condition_id
                {where}
                ORDER BY s.resolved_ts DESC"""
        )
    )
    conn.close()
    if max_rows:
        rows = rows[:max_rows]
    log.info("backfill_horizons_start", n=len(rows), concurrency=concurrency, horizons=list(HORIZONS))

    sem = asyncio.Semaphore(concurrency)
    updated = 0
    no_history = 0
    errors = 0

    async def _one(session: aiohttp.ClientSession, cid: str, close_ts: float, token_id: str) -> None:
        nonlocal updated, no_history, errors
        async with sem:
            try:
                history = await fetch_history(session, token_id, interval="max", fidelity=3600)
            except Exception as e:
                errors += 1
                log.warning("price_history_error", cid=cid, err=str(e))
                return
            if not history:
                no_history += 1
                return
            prices = {col: _price_at_offset(history, close_ts, sec) for col, sec in HORIZONS.items()}
            if all(v is None for v in prices.values()):
                no_history += 1
                return
            sets = ", ".join(f"{c}=?" for c in prices)
            params = list(prices.values()) + [cid]
            conn2 = sqlite3.connect(db_path)
            conn2.execute(f"UPDATE signal_outcomes SET {sets} WHERE condition_id = ?", params)
            conn2.commit()
            conn2.close()
            updated += 1

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (cid, close_ts, token_id) in enumerate(rows):
            tasks.append(asyncio.create_task(_one(session, cid, close_ts, token_id)))
            if (i + 1) % 200 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                log.info(
                    "backfill_horizons_progress",
                    done=i + 1,
                    of=len(rows),
                    updated=updated,
                    no_history=no_history,
                    errors=errors,
                )
        if tasks:
            await asyncio.gather(*tasks)

    return {"total": len(rows), "updated": updated, "no_history": no_history, "errors": errors}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--force", action="store_true", help="Re-fetch even rows already populated")
    args = p.parse_args()
    summary = asyncio.run(backfill(settings.db_path, args.max, args.concurrency, args.force))
    log.info("backfill_horizons_done", **summary)


if __name__ == "__main__":
    main()
