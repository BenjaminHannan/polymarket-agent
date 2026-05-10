"""Backfill historical_trades for every market matching the certified-
category allowlist (currently `sports_global`).

The data-api global stream caps at offset=3500 (~minutes of history).
Per-market pagination caps at offset=3000 but per-market that's usually
~5-10 days for an active market. Iterating across the certified-slice
markets gives us the best available retroactive coverage from the
public API.

Usage:
  python -m scripts.backfill_certified_markets
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.data.polymarket_trades import (
    backfill_market,
    top_volume_wallets,
)
from polyagent.gamma import fetch_markets_by_category

log = logging_setup.configure()


async def _run() -> None:
    # Build the certified-category allowlist from strategy_certificates.
    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        "SELECT detail FROM strategy_certificates WHERE enabled = 1"
    ).fetchall()
    import json
    allowed = set()
    for (detail,) in rows:
        try:
            d = json.loads(detail or "{}")
        except Exception:
            continue
        cat = d.get("category")
        if isinstance(cat, str) and cat:
            allowed.add(cat)
    conn.close()
    if not allowed:
        log.warning("no_certified_categories")
        return

    # Fetch live markets in those categories
    log.info("fetching_certified_markets", categories=sorted(allowed))
    all_markets = []
    for cat in allowed:
        ms = await fetch_markets_by_category(cat, limit=500, min_liquidity=100, pages=10)
        all_markets.extend(ms)
    # Dedupe by condition_id
    seen = set()
    markets = []
    for m in all_markets:
        if m.condition_id in seen:
            continue
        seen.add(m.condition_id)
        markets.append(m)
    log.info("certified_markets_loaded", n=len(markets))

    # Backfill each market (cap at offset=3000 per-market via the API)
    started = time.time()
    totals = {"fetched": 0, "inserted": 0, "duplicates": 0, "errors": 0}
    for i, m in enumerate(markets):
        try:
            s = await backfill_market(
                settings.db_path, m.condition_id, max_pages=10,
            )
            totals["fetched"] += s.fetched
            totals["inserted"] += s.inserted
            totals["duplicates"] += s.duplicates
            log.info(
                "market_backfill_done",
                idx=i + 1, of=len(markets),
                cid=m.condition_id[:14],
                fetched=s.fetched, inserted=s.inserted,
                question=(m.question or "")[:60],
            )
        except Exception as e:
            totals["errors"] += 1
            log.warning("market_backfill_error", cid=m.condition_id[:14], err=str(e))

    elapsed = time.time() - started
    log.info("certified_backfill_summary", elapsed_sec=round(elapsed, 1), **totals)

    # Print top-20 wallets after the backfill
    wallets = top_volume_wallets(
        settings.db_path, days=14, top_k=20, min_usdc_volume=5000.0,
    )
    log.info("top_volume_wallets_after_backfill", n=len(wallets))
    for w in wallets:
        log.info("wallet_volume", **w)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
