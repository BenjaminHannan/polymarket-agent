"""Della Vedova wallet-orthogonality classifier.

Direct implementation of the doc's Problem-10 fix #1 — Della Vedova
2026 (SSRN 6191618, dellavedova.com) wallet-level orthogonality test.
On 222M Polymarket trades, 6,292 wallets out of 483K were flagged with
realized accuracy too high to be luck at p<0.01, concentrated in Action
and Vote markets, lowest in Stochastic.

Statistical test
----------------
For wallet w with N resolved trades and realized YES-win count K:

  H0: trades are independent coin-flips (p_win = 0.5)
  one-sided p = P(Binom(N, 0.5) ≥ K)

A wallet is flagged "informed" iff this p < `p_threshold` (default 0.01)
AND N ≥ `min_trades` (default 50, below which the binomial test is
underpowered against the realistic null of any sensible market price).

Closing-window caveat
---------------------
Polymarket + Chainalysis went live Apr 2026 to actively suppress these
wallets. The edge is closing, but a confirmed-skill wallet is still a
positive prior. Use as a *negative-suppressant* in markets dominated by
unflagged wallets and a *positive boost* when a flagged wallet has
recently entered the same market.

Storage
-------
Persisted to `wallet_orthogonality` table:
  wallet TEXT PRIMARY KEY
  n_trades INTEGER
  n_wins INTEGER
  win_rate REAL
  binom_p REAL
  is_informed INTEGER     -- 1 if p < p_threshold and N ≥ min_trades
  last_updated REAL
  category_dist TEXT      -- JSON {category: trade_count}
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


def _binom_sf(n: int, k: int, p: float = 0.5) -> float:
    """One-sided binomial survival function P(X >= k) for X ~ Binom(n, p).

    Pure-Python implementation to avoid scipy dependency. Uses log-space
    to avoid overflow on n up to ~10K.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # P(X >= k) = sum_{i=k}^{n} C(n, i) p^i (1-p)^(n-i)
    # log domain to avoid overflow:
    log_p = math.log(p) if p > 0 else float("-inf")
    log_q = math.log(1 - p) if p < 1 else float("-inf")
    # log C(n, i) via lgamma
    def log_binom(n_, i_):
        return (math.lgamma(n_ + 1) - math.lgamma(i_ + 1) - math.lgamma(n_ - i_ + 1))
    # Stream the sum in log-space using log-sum-exp.
    log_terms = []
    for i in range(k, n + 1):
        log_terms.append(log_binom(n, i) + i * log_p + (n - i) * log_q)
    if not log_terms:
        return 0.0
    m = max(log_terms)
    s = m + math.log(sum(math.exp(lt - m) for lt in log_terms))
    return float(math.exp(s))


@dataclass
class WalletStats:
    wallet: str
    n_trades: int
    n_wins: int
    win_rate: float
    binom_p: float
    is_informed: bool
    category_dist: dict


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS wallet_orthogonality (
            wallet TEXT PRIMARY KEY,
            n_trades INTEGER NOT NULL,
            n_wins INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            binom_p REAL NOT NULL,
            is_informed INTEGER NOT NULL,
            last_updated REAL NOT NULL,
            category_dist TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS wo_informed ON wallet_orthogonality(is_informed, n_trades)"
    )
    conn.commit()


def compute_wallet_stats(
    conn: sqlite3.Connection,
    *,
    p_threshold: float = 0.01,
    min_trades: int = 50,
    category_field: str = "category",
) -> list[WalletStats]:
    """Compute Della Vedova orthogonality stats for every wallet with
    resolved trades in the `trades` table.

    Schema assumption (matches `polyagent/data/polymarket_trades.py`):
        trades(
            tx_hash, wallet, asset, side, size, price,
            timestamp, market_id, outcome_resolved INTEGER,  -- 1 YES wins
            category TEXT
        )

    Only counts trades where `outcome_resolved IS NOT NULL`. A trade
    is a "win" iff (side='BUY' AND outcome_resolved=1) OR
    (side='SELL' AND outcome_resolved=0) — i.e. they ended up on the
    winning side.
    """
    ensure_table(conn)
    rows = conn.execute(
        f"""SELECT wallet,
                  SUM(CASE
                        WHEN (side='BUY'  AND outcome_resolved=1)
                          OR (side='SELL' AND outcome_resolved=0)
                        THEN 1 ELSE 0
                      END) AS n_wins,
                  COUNT(*) AS n_trades,
                  GROUP_CONCAT(COALESCE({category_field}, 'unknown'), ',') AS cats
           FROM trades
           WHERE outcome_resolved IS NOT NULL
             AND wallet IS NOT NULL
           GROUP BY wallet
           HAVING COUNT(*) >= ?""",
        (min_trades,),
    ).fetchall()
    now = time.time()
    out: list[WalletStats] = []
    for wallet, n_wins, n_trades, cats_str in rows:
        if n_trades < min_trades:
            continue
        p = _binom_sf(n_trades, n_wins)
        wr = n_wins / n_trades if n_trades else 0.0
        is_informed = (p < p_threshold) and (n_trades >= min_trades) and (wr > 0.5)
        cat_dist: dict[str, int] = {}
        if cats_str:
            for c in cats_str.split(","):
                cat_dist[c] = cat_dist.get(c, 0) + 1
        stats = WalletStats(
            wallet=wallet,
            n_trades=n_trades,
            n_wins=n_wins,
            win_rate=wr,
            binom_p=p,
            is_informed=is_informed,
            category_dist=cat_dist,
        )
        conn.execute(
            """INSERT INTO wallet_orthogonality
               (wallet, n_trades, n_wins, win_rate, binom_p, is_informed,
                last_updated, category_dist)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(wallet) DO UPDATE SET
                  n_trades=excluded.n_trades,
                  n_wins=excluded.n_wins,
                  win_rate=excluded.win_rate,
                  binom_p=excluded.binom_p,
                  is_informed=excluded.is_informed,
                  last_updated=excluded.last_updated,
                  category_dist=excluded.category_dist""",
            (wallet, n_trades, n_wins, wr, p, int(is_informed), now,
             json.dumps(cat_dist)),
        )
        out.append(stats)
    conn.commit()
    n_informed = sum(1 for s in out if s.is_informed)
    log.info(
        "wallet_orthogonality_refresh",
        n_wallets=len(out),
        n_informed=n_informed,
        threshold=p_threshold,
        min_trades=min_trades,
    )
    return out


def is_wallet_informed(conn: sqlite3.Connection, wallet: str) -> bool:
    """Cheap O(1) check used at trading decision time."""
    row = conn.execute(
        "SELECT is_informed FROM wallet_orthogonality WHERE wallet=?",
        (wallet,),
    ).fetchone()
    return bool(row and row[0])


def informed_wallets_in_market(
    conn: sqlite3.Connection,
    asset: str,
    *,
    lookback_sec: float = 86400.0,
) -> list[str]:
    """Return wallets currently flagged informed that have traded
    `asset` in the last `lookback_sec` seconds. Use as a positive-prior
    signal — if an informed wallet has recently entered the same
    market, we have a probabilistic reason to think the consensus has
    private information."""
    cutoff = time.time() - lookback_sec
    rows = conn.execute(
        """SELECT DISTINCT t.wallet
           FROM trades t
           INNER JOIN wallet_orthogonality wo ON wo.wallet = t.wallet
           WHERE t.asset = ?
             AND t.timestamp >= ?
             AND wo.is_informed = 1""",
        (asset, cutoff),
    ).fetchall()
    return [r[0] for r in rows]
