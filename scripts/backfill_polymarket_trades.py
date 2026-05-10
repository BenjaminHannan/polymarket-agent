"""Backfill historical trades from data-api.polymarket.com.

Two modes:
  --global         walks the global recent-trades stream (no market filter)
  --market <cid>   walks one specific market's history

Both write to the historical_trades table and are idempotent (PRIMARY KEY
on tx_hash + wallet + asset + side + size + price drops dupes).

Usage:
  python -m scripts.backfill_polymarket_trades --global --max-pages 200
  python -m scripts.backfill_polymarket_trades --market 0x33a87... --max-pages 50
  python -m scripts.backfill_polymarket_trades --since-days 30 --global
"""
from __future__ import annotations

import argparse
import asyncio
import time

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.data.polymarket_trades import (
    backfill_global,
    backfill_market,
    top_volume_wallets,
)

log = logging_setup.configure()


async def _run(args) -> None:
    earliest = None
    if args.since_days:
        earliest = time.time() - args.since_days * 86400

    if args.market:
        summary = await backfill_market(
            settings.db_path, args.market,
            max_pages=args.max_pages, earliest_ts=earliest,
        )
    else:
        summary = await backfill_global(
            settings.db_path,
            max_pages=args.max_pages, earliest_ts=earliest,
        )
    log.info(
        "polymarket_trades_backfill_done",
        fetched=summary.fetched,
        inserted=summary.inserted,
        duplicates=summary.duplicates,
        pages=summary.pages,
        earliest_ts=summary.earliest_ts,
        latest_ts=summary.latest_ts,
    )

    if args.show_top_wallets:
        wallets = top_volume_wallets(
            settings.db_path, days=args.since_days or 30, top_k=20,
        )
        log.info("top_volume_wallets_sample", n=len(wallets))
        for w in wallets[:20]:
            log.info("wallet_volume", **w)


def main() -> None:
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--global", dest="globalmode", action="store_true",
                     help="walk the global recent-trades stream")
    grp.add_argument("--market", help="condition_id (0x…) to backfill one market")
    p.add_argument("--max-pages", type=int, default=200,
                   help="max pages to walk (each page = 500 trades)")
    p.add_argument("--since-days", type=int, default=None,
                   help="stop walking once a page's oldest trade is older than this")
    p.add_argument("--show-top-wallets", action="store_true",
                   help="after backfill, print top-20 wallets by USDC volume")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
