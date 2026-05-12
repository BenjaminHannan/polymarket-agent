"""Polymarket native-endpoint features: comments, leaderboard, uniques.

Three signals from public unauthenticated Polymarket REST endpoints
that the existing pipeline doesn't yet use:

  - `comment_count_delta_6h`  — change in comment count over last 6h;
                                proxy for retail attention spikes
  - `top_trader_inflow_24h`   — net YES-side inflow to a market from
                                wallets in the top-100 monthly leaderboard
  - `unique_traders_1h`       — distinct wallet count trading in last 1h

The doc framing: these are *retail-attention* and *informed-flow*
proxies that Heng-Soh's LR gate and the smart-money tighten module
don't currently see at this surface. Mitts & Ofir's wallet features
operate on a different time scale; these are minute-to-hour.

This module is a polled cache populated on a 5-minute cadence, written
to `polymarket_native_features` keyed by `condition_id`. The features
in `features.py` read from this table at decision time so we don't
hit the API per signal evaluation.

Endpoints
---------
  - GET /comments/{market}     (Gamma + CLOB)
  - GET /trades?market=...     (data-api, last 1h)
  - GET /leaderboard/top       (data-api monthly)

All free / unauthenticated; same conventions as
`polyagent/data/polymarket_trades.py`.

Disabled when ENABLE_POLYMARKET_NATIVE=0.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS polymarket_native_features (
            condition_id TEXT PRIMARY KEY,
            comment_count_6h INTEGER,
            comment_count_delta_6h INTEGER,
            top_trader_inflow_24h REAL,
            unique_traders_1h INTEGER,
            last_updated REAL NOT NULL
        )"""
    )
    conn.commit()


@dataclass
class NativeFeatures:
    condition_id: str
    comment_count_6h: int
    comment_count_delta_6h: int
    top_trader_inflow_24h: float
    unique_traders_1h: int


async def _http_get_json(url: str, *, timeout: float = 10.0) -> object | None:
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return await r.json()


async def fetch_comment_count(market_id: str) -> int:
    """Count of comments on the market. Polymarket caps to a few pages
    so for "delta" purposes we just return the page-1 length."""
    try:
        data = await _http_get_json(f"{GAMMA_API_BASE}/comments/{market_id}")
    except Exception:
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "comments" in data:
        return len(data.get("comments") or [])
    return 0


async def fetch_unique_traders_1h(market_id: str) -> int:
    """Distinct wallets trading on this market in the last hour."""
    since = int(time.time()) - 3600
    try:
        data = await _http_get_json(
            f"{DATA_API_BASE}/trades?market={market_id}&limit=500&offset=0"
        )
    except Exception:
        return 0
    if not isinstance(data, list):
        return 0
    seen = set()
    for row in data:
        ts = row.get("timestamp", 0) or 0
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            continue
        if ts < since:
            continue
        w = row.get("proxyWallet") or row.get("wallet")
        if w:
            seen.add(w)
    return len(seen)


async def fetch_top_trader_inflow_24h(
    market_id: str, top_wallets: set[str],
) -> float:
    """Net YES-side inflow over the last 24h from wallets in the top-100
    monthly leaderboard. Positive = buyers, negative = sellers."""
    since = int(time.time()) - 86400
    try:
        data = await _http_get_json(
            f"{DATA_API_BASE}/trades?market={market_id}&limit=2000&offset=0"
        )
    except Exception:
        return 0.0
    if not isinstance(data, list):
        return 0.0
    net = 0.0
    for row in data:
        ts = row.get("timestamp", 0) or 0
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            continue
        if ts < since:
            continue
        wallet = row.get("proxyWallet") or row.get("wallet")
        if wallet not in top_wallets:
            continue
        side = (row.get("side") or "").upper()
        try:
            size = float(row.get("size") or 0)
            price = float(row.get("price") or 0)
        except (TypeError, ValueError):
            continue
        notional = size * price
        if side == "BUY":
            net += notional
        elif side == "SELL":
            net -= notional
    return float(net)


async def fetch_top_wallets(limit: int = 100) -> set[str]:
    """Top-N wallets by monthly volume per the public leaderboard."""
    try:
        data = await _http_get_json(
            f"{DATA_API_BASE}/leaderboard/top?period=monthly&limit={limit}"
        )
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    out = set()
    for row in data:
        w = row.get("proxyWallet") or row.get("wallet") or row.get("address")
        if w:
            out.add(w)
    return out


async def refresh_market(
    conn: sqlite3.Connection,
    market_id: str,
    condition_id: str,
    top_wallets: set[str],
) -> NativeFeatures | None:
    """Refresh native features for a single market and persist."""
    ensure_table(conn)
    cc_now, ut1h, inflow24h = await asyncio.gather(
        fetch_comment_count(market_id),
        fetch_unique_traders_1h(market_id),
        fetch_top_trader_inflow_24h(market_id, top_wallets),
    )
    # delta vs previous snapshot
    prev = conn.execute(
        "SELECT comment_count_6h FROM polymarket_native_features WHERE condition_id=?",
        (condition_id,),
    ).fetchone()
    delta = int(cc_now) - int(prev[0]) if prev else 0
    now = time.time()
    conn.execute(
        """INSERT INTO polymarket_native_features
           (condition_id, comment_count_6h, comment_count_delta_6h,
            top_trader_inflow_24h, unique_traders_1h, last_updated)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(condition_id) DO UPDATE SET
              comment_count_6h=excluded.comment_count_6h,
              comment_count_delta_6h=excluded.comment_count_delta_6h,
              top_trader_inflow_24h=excluded.top_trader_inflow_24h,
              unique_traders_1h=excluded.unique_traders_1h,
              last_updated=excluded.last_updated""",
        (condition_id, int(cc_now), int(delta), float(inflow24h),
         int(ut1h), now),
    )
    conn.commit()
    return NativeFeatures(
        condition_id=condition_id,
        comment_count_6h=int(cc_now),
        comment_count_delta_6h=int(delta),
        top_trader_inflow_24h=float(inflow24h),
        unique_traders_1h=int(ut1h),
    )


def lookup_features(
    conn: sqlite3.Connection, condition_id: str,
) -> NativeFeatures | None:
    """Read native features for a single market (used at signal-eval time
    from features.py). Returns None if no snapshot yet."""
    row = conn.execute(
        """SELECT comment_count_6h, comment_count_delta_6h,
                  top_trader_inflow_24h, unique_traders_1h
           FROM polymarket_native_features
           WHERE condition_id=?""",
        (condition_id,),
    ).fetchone()
    if not row:
        return None
    return NativeFeatures(
        condition_id=condition_id,
        comment_count_6h=int(row[0] or 0),
        comment_count_delta_6h=int(row[1] or 0),
        top_trader_inflow_24h=float(row[2] or 0.0),
        unique_traders_1h=int(row[3] or 0),
    )


async def run_native_poller(
    db_path: str,
    markets: list,
    *,
    poll_sec: float = 300.0,
    leaderboard_refresh_sec: float = 3600.0,
    max_concurrent: int = 5,
) -> None:
    """Long-running supervised task: poll each market's native features
    on `poll_sec` cadence (5 min default)."""
    log.info(
        "polymarket_native_poller_start",
        n_markets=len(markets),
        poll_sec=poll_sec,
    )
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_table(conn)
    last_leaderboard = 0.0
    top_wallets: set[str] = set()
    sem = asyncio.Semaphore(max_concurrent)

    async def _refresh_one(m):
        async with sem:
            try:
                # Polymarket market_id is the condition_id in many cases;
                # some endpoints want the slug. Try condition_id first.
                await refresh_market(
                    conn,
                    market_id=getattr(m, "condition_id", "") or "",
                    condition_id=getattr(m, "condition_id", "") or "",
                    top_wallets=top_wallets,
                )
            except Exception as e:
                log.warning("native_poller_market_failed",
                            cid=getattr(m, "condition_id", "")[:14], err=str(e))

    while True:
        try:
            now = time.time()
            if now - last_leaderboard >= leaderboard_refresh_sec or not top_wallets:
                top_wallets = await fetch_top_wallets()
                last_leaderboard = now
                log.info("native_poller_leaderboard_refresh", n=len(top_wallets))
            if markets:
                await asyncio.gather(*[_refresh_one(m) for m in markets])
        except Exception as e:
            log.warning("native_poller_loop_error", err=str(e))
        await asyncio.sleep(poll_sec)
