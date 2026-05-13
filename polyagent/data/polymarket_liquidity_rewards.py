"""Polymarket Liquidity Rewards API poller (CLOB v2, April 2026).

Polymarket launched the CLOB v2 upgrade on April 28, 2026 alongside a
**$1M Liquidity Rewards Program**. The program scores resting maker
quotes via a quadratic-spread formula and distributes USDC rewards
daily. Eligible markets are flagged in the official endpoints with
the `rewards` object containing `pool_size_usd`, `max_spread_bps`,
`min_size_at_quote`, and category-specific parameters.

This module is the *data poller* for that program. We don't collect
the rebates (KYC + real money required), but we can:

  1. Pull per-market reward-pool size and eligibility flags
  2. Combine with `polyagent/risk/maker_rewards.py`'s local quadratic-
     spread score to estimate our paper book's daily USD entitlement
  3. Expose `is_market_eligible(condition_id)` so `passive_poster_v2`
     can prioritize quoting on rewards-eligible markets

What's stored
-------------
Per-market state in `polymarket_liquidity_rewards` (sqlite):

  condition_id TEXT PRIMARY KEY
  is_eligible INTEGER
  pool_size_usd REAL
  max_spread_bps REAL
  min_size_at_quote REAL
  last_updated REAL

Default disabled when ENABLE_POLYMARKET_REWARDS=0 or no Polymarket-
api endpoint is configured.

Reference
---------
- Polymarket Help Center: Liquidity Rewards
  https://help.polymarket.com/en/articles/13364466-liquidity-rewards
- Polymarket docs: https://docs.polymarket.com/market-makers/liquidity-rewards
- CLOB v2 launch coverage:
  https://coinalertnews.com/news/2026/04/28/polymarket-clob-v2-liquidity-rewards
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"


@dataclass
class RewardsState:
    condition_id: str
    is_eligible: bool
    pool_size_usd: float
    max_spread_bps: float
    min_size_at_quote: float


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS polymarket_liquidity_rewards (
            condition_id TEXT PRIMARY KEY,
            is_eligible INTEGER NOT NULL,
            pool_size_usd REAL,
            max_spread_bps REAL,
            min_size_at_quote REAL,
            last_updated REAL NOT NULL
        )"""
    )
    conn.commit()


async def _http_get_json(url: str, *, timeout: float = 10.0) -> object | None:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return await r.json()


async def fetch_rewards_for_market(market_id: str) -> RewardsState | None:
    """Pull rewards eligibility + parameters for one market.

    Verified against the live Gamma API (May 2026): rewards parameters
    are exposed as **flat scalar fields** on the market object, not
    nested in a `rewards` block. The relevant keys are:

      - `rewardsMinSize`   — minimum quote size (shares) for eligibility
      - `rewardsMaxSpread` — maximum half-spread (cents) for eligibility

    A market is enrolled in the Liquidity Rewards Program iff either
    field is non-zero. Gamma does NOT expose the per-market USDC pool
    size on this endpoint — only the structural parameters. The pool
    is shared across all eligible markets and distributed pro-rata to
    each maker's quadratic-spread score. We populate `pool_size_usd`
    with 0.0 here; the strategy uses `max_spread_bps` + `min_size_at_quote`
    as the operational signal (quote tighter than max_spread_bps on
    eligible markets to qualify for the daily share).

    Reference: live response shape includes
      {
        "rewardsMinSize": 20,
        "rewardsMaxSpread": 3.5,
        ... (no nested 'rewards' block)
      }
    """
    # Use the list form because /markets/{id} sometimes returns a
    # wrapped object that varies in shape; the list form is reliable.
    try:
        data = await _http_get_json(
            f"{GAMMA_API_BASE}/markets?condition_ids={market_id}"
        )
    except Exception as e:
        log.warning("rewards_fetch_failed", mid=market_id[:14], err=str(e))
        return None
    if isinstance(data, list):
        if not data:
            return None
        market = data[0]
    elif isinstance(data, dict):
        market = data
    else:
        return None
    rewards_min_size = float(market.get("rewardsMinSize") or 0.0)
    rewards_max_spread_cents = float(market.get("rewardsMaxSpread") or 0.0)
    is_eligible = rewards_min_size > 0 or rewards_max_spread_cents > 0
    # rewardsMaxSpread is in *cents* (3.5 = 3.5¢ max half-spread).
    # Convert to bps: 1 cent = 100 bps of probability space.
    max_spread_bps = rewards_max_spread_cents * 100.0
    return RewardsState(
        condition_id=market_id,
        is_eligible=is_eligible,
        pool_size_usd=0.0,   # not exposed on this endpoint; placeholder
        max_spread_bps=max_spread_bps,
        min_size_at_quote=rewards_min_size,
    )


def persist_state(conn: sqlite3.Connection, state: RewardsState) -> None:
    """Persist one market's rewards eligibility snapshot."""
    ensure_table(conn)
    conn.execute(
        """INSERT INTO polymarket_liquidity_rewards
           (condition_id, is_eligible, pool_size_usd, max_spread_bps,
            min_size_at_quote, last_updated)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(condition_id) DO UPDATE SET
              is_eligible=excluded.is_eligible,
              pool_size_usd=excluded.pool_size_usd,
              max_spread_bps=excluded.max_spread_bps,
              min_size_at_quote=excluded.min_size_at_quote,
              last_updated=excluded.last_updated""",
        (state.condition_id, int(state.is_eligible), state.pool_size_usd,
         state.max_spread_bps, state.min_size_at_quote, time.time()),
    )
    conn.commit()


def lookup(conn: sqlite3.Connection, condition_id: str) -> RewardsState | None:
    """O(1) lookup of cached rewards state for one market."""
    row = conn.execute(
        """SELECT condition_id, is_eligible, pool_size_usd,
                  max_spread_bps, min_size_at_quote
           FROM polymarket_liquidity_rewards WHERE condition_id=?""",
        (condition_id,),
    ).fetchone()
    if not row:
        return None
    return RewardsState(
        condition_id=row[0],
        is_eligible=bool(row[1]),
        pool_size_usd=float(row[2] or 0.0),
        max_spread_bps=float(row[3] or 0.0),
        min_size_at_quote=float(row[4] or 0.0),
    )


def is_market_eligible(conn: sqlite3.Connection, condition_id: str) -> bool:
    """Cheap O(1) check at quote-decision time."""
    s = lookup(conn, condition_id)
    return bool(s and s.is_eligible)


def eligible_pool_for_market(conn: sqlite3.Connection, condition_id: str) -> float:
    """Daily USDC pool size for `condition_id`, or 0.0 if not eligible."""
    s = lookup(conn, condition_id)
    return float(s.pool_size_usd) if (s and s.is_eligible) else 0.0


def total_eligible_pool_usd(conn: sqlite3.Connection) -> float:
    """Sum of all eligible pool sizes across markets we've seen."""
    ensure_table(conn)
    row = conn.execute(
        "SELECT COALESCE(SUM(pool_size_usd), 0) FROM polymarket_liquidity_rewards "
        "WHERE is_eligible = 1"
    ).fetchone()
    return float(row[0] or 0.0)


async def run_rewards_poller(
    db_path: str,
    markets: list,
    *,
    poll_sec: float = 1800.0,   # 30 min — rewards eligibility doesn't change often
    max_concurrent: int = 5,
) -> None:
    """Long-running supervised task: poll rewards eligibility for every
    known market on a 30-min cadence."""
    log.info(
        "polymarket_liquidity_rewards_poller_start",
        n_markets=len(markets), poll_sec=poll_sec,
    )
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_table(conn)
    sem = asyncio.Semaphore(max_concurrent)

    async def _refresh_one(m):
        async with sem:
            try:
                state = await fetch_rewards_for_market(m.condition_id)
                if state is not None:
                    persist_state(conn, state)
            except Exception as e:
                log.warning("rewards_poller_one_failed",
                            cid=(m.condition_id or "")[:14], err=str(e))

    while True:
        try:
            if markets:
                await asyncio.gather(*[_refresh_one(m) for m in markets])
                eligible = total_eligible_pool_usd(conn)
                log.info(
                    "polymarket_liquidity_rewards_refresh",
                    n_markets=len(markets),
                    total_eligible_pool_usd=round(eligible, 2),
                )
        except Exception as e:
            log.warning("rewards_poller_loop_error", err=str(e))
        await asyncio.sleep(poll_sec)
