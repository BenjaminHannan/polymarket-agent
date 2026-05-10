"""Wash-trade graph clustering (Sirolly et al. Nov 2025, SSRN 5714122).

Companion to the runtime `WashFilter`. This module operates on the
*on-chain trades* table populated by `polyagent/data/polymarket_trades.py`
and detects approximately-closed counterparty subgraphs — i.e. clusters
of wallets that trade predominantly with each other and rarely with the
broader market. Such clusters are the published structural signature of
wash trading: $X cycled between wallets in the cluster shows up as
volume but provides no informational content.

Empirical context (Sirolly Nov 2025):
  - Wash-share averaged ~25% across all of Polymarket.
  - Peaked at **60% in December 2024** (when bot reward-farming
    incentives were highest).
  - Sat at ~20% in October 2025 — still material.
  - **Sports is the worst-affected category.** Our certified
    `sports_global` slice is in the worst-contaminated zone.

Algorithm (lightweight version)
-------------------------------
The full Sirolly paper iteratively partitions a weighted directed
counterparty graph using community-detection (Louvain / Leiden). We
implement a simpler signature that is both cheaper and surprisingly
discriminative:

  For each wallet w, compute:
    - n_counterparties(w)       — distinct other wallets traded with
    - top_share(w)              — fraction of w's volume with its
                                  single most-frequent counterparty
    - reciprocal_share(w)       — fraction of w's BUY volume against
                                  its top SELL counterparty (and vice
                                  versa) — wash-traders cycle in pairs

A wallet is **suspect** when:
    n_counterparties ≤ 5  AND
    top_share ≥ 0.70      AND
    reciprocal_share ≥ 0.50

A market is then assigned a `wash_share` = (volume from suspect
wallets) / (total volume), and we expose:

    market_wash_share(asset) -> float
    is_high_wash(asset, threshold=0.30) -> bool
    suppression_factor(asset) -> float in [0.0, 1.0]   # 1 − wash_share

so signal-layer code can multiply alpha by `suppression_factor` to
attenuate trades in contaminated markets.

Storage
-------
Persists to:
  wash_suspect_wallets(
    wallet PRIMARY KEY,
    n_counterparties INT, top_share REAL, reciprocal_share REAL,
    suspect INT, last_updated REAL
  )

  market_wash_score(
    asset PRIMARY KEY,
    wash_share REAL, n_trades INT, last_updated REAL
  )
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class WalletSignature:
    wallet: str
    n_counterparties: int
    top_share: float
    reciprocal_share: float
    is_suspect: bool


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS wash_suspect_wallets (
            wallet TEXT PRIMARY KEY,
            n_counterparties INTEGER NOT NULL,
            top_share REAL NOT NULL,
            reciprocal_share REAL NOT NULL,
            is_suspect INTEGER NOT NULL,
            last_updated REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS market_wash_score (
            asset TEXT PRIMARY KEY,
            wash_share REAL NOT NULL,
            n_trades INTEGER NOT NULL,
            last_updated REAL NOT NULL
        )"""
    )
    conn.commit()


def compute_wallet_signatures(
    conn: sqlite3.Connection,
    *,
    min_trades: int = 20,
    max_n_counterparties: int = 5,
    min_top_share: float = 0.70,
    min_reciprocal_share: float = 0.50,
) -> list[WalletSignature]:
    """Scan the `trades` table and compute per-wallet wash signatures.

    Requires the trades table to have a `counterparty_wallet` column —
    Polymarket's CTF emits both maker and taker on each fill. If only
    `wallet` is recorded (one side), the heuristic degrades to
    counterparty-count via market_id × close-timestamp matching,
    which is left as a TODO. Assumes the `trades` table has been
    augmented; if missing, returns []."""
    ensure_tables(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if "counterparty_wallet" not in cols or "wallet" not in cols:
        log.info(
            "wash_graph_skip_no_counterparty_col",
            available_cols=sorted(cols),
        )
        return []

    rows = conn.execute(
        """SELECT wallet, counterparty_wallet, side, size
           FROM trades
           WHERE wallet IS NOT NULL AND counterparty_wallet IS NOT NULL"""
    ).fetchall()
    if not rows:
        return []

    # Aggregate: wallet -> counterparty -> {buy_vol, sell_vol}
    agg: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"buy_vol": 0.0, "sell_vol": 0.0})
    )
    wallet_trades: dict[str, int] = defaultdict(int)
    for wallet, cpty, side, size in rows:
        s = float(size or 0.0)
        if s <= 0:
            continue
        side_n = (side or "").upper().strip()
        bucket = agg[wallet][cpty]
        if side_n == "BUY":
            bucket["buy_vol"] += s
        elif side_n == "SELL":
            bucket["sell_vol"] += s
        wallet_trades[wallet] += 1

    now = time.time()
    out: list[WalletSignature] = []
    for wallet, cpty_map in agg.items():
        n_trades = wallet_trades.get(wallet, 0)
        if n_trades < min_trades:
            continue
        n_cpty = len(cpty_map)
        cpty_totals = [
            (c, d["buy_vol"] + d["sell_vol"])
            for c, d in cpty_map.items()
        ]
        cpty_totals.sort(key=lambda x: x[1], reverse=True)
        total_vol = sum(v for _, v in cpty_totals)
        if total_vol <= 0:
            continue
        top_cpty, top_vol = cpty_totals[0]
        top_share = top_vol / total_vol
        top_record = cpty_map[top_cpty]
        # Reciprocal share: how much of my BUY volume was to the top
        # cpty, paired with my SELL volume to that same cpty? On wash
        # cycles these are nearly equal.
        buy = top_record["buy_vol"]
        sell = top_record["sell_vol"]
        reciprocal = (min(buy, sell) * 2.0) / max(top_vol, 1e-9)

        is_suspect = (
            n_cpty <= max_n_counterparties
            and top_share >= min_top_share
            and reciprocal >= min_reciprocal_share
        )
        sig = WalletSignature(
            wallet=wallet,
            n_counterparties=n_cpty,
            top_share=top_share,
            reciprocal_share=reciprocal,
            is_suspect=is_suspect,
        )
        conn.execute(
            """INSERT INTO wash_suspect_wallets
               (wallet, n_counterparties, top_share, reciprocal_share,
                is_suspect, last_updated)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(wallet) DO UPDATE SET
                   n_counterparties=excluded.n_counterparties,
                   top_share=excluded.top_share,
                   reciprocal_share=excluded.reciprocal_share,
                   is_suspect=excluded.is_suspect,
                   last_updated=excluded.last_updated""",
            (wallet, n_cpty, top_share, reciprocal, int(is_suspect), now),
        )
        out.append(sig)
    conn.commit()
    n_suspect = sum(1 for s in out if s.is_suspect)
    log.info("wash_graph_refresh", n_wallets=len(out), n_suspect=n_suspect)
    return out


def compute_market_wash_scores(conn: sqlite3.Connection) -> None:
    """For every asset traded, compute wash_share = volume from
    suspect wallets / total volume."""
    ensure_tables(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if "wallet" not in cols or "asset" not in cols:
        return
    rows = conn.execute(
        """SELECT t.asset,
                  SUM(CASE WHEN ws.is_suspect=1 THEN t.size ELSE 0 END) AS susp_vol,
                  SUM(t.size) AS total_vol,
                  COUNT(*) AS n_trades
           FROM trades t
           LEFT JOIN wash_suspect_wallets ws ON ws.wallet = t.wallet
           WHERE t.asset IS NOT NULL
           GROUP BY t.asset"""
    ).fetchall()
    now = time.time()
    for asset, susp_vol, total_vol, n_trades in rows:
        if total_vol is None or total_vol <= 0:
            continue
        ws = float(susp_vol or 0) / float(total_vol)
        conn.execute(
            """INSERT INTO market_wash_score (asset, wash_share, n_trades, last_updated)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(asset) DO UPDATE SET
                  wash_share=excluded.wash_share,
                  n_trades=excluded.n_trades,
                  last_updated=excluded.last_updated""",
            (asset, ws, int(n_trades or 0), now),
        )
    conn.commit()


def market_wash_share(conn: sqlite3.Connection, asset: str) -> float:
    """O(1) lookup of an asset's wash share. Defaults to 0 if not yet
    scored."""
    row = conn.execute(
        "SELECT wash_share FROM market_wash_score WHERE asset=?",
        (asset,),
    ).fetchone()
    return float(row[0]) if row else 0.0


def suppression_factor(conn: sqlite3.Connection, asset: str) -> float:
    """1 − wash_share, clamped to [0, 1]. Multiply alpha or sizing
    by this to attenuate signal in contaminated markets."""
    ws = market_wash_share(conn, asset)
    return float(max(0.0, min(1.0, 1.0 - ws)))


def is_high_wash(
    conn: sqlite3.Connection, asset: str, threshold: float = 0.30
) -> bool:
    """Convenience: True iff wash_share ≥ threshold."""
    return market_wash_share(conn, asset) >= float(threshold)
