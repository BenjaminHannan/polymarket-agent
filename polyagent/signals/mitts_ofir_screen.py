"""Mitts & Ofir (2026) 5-signal informed-wallet screen.

Reference: Mitts & Ofir, "From Iran to Taylor Swift: Informed Trading
in Prediction Markets," SSRN 6426778, March 2026.

The paper identifies 210,718 suspicious (wallet, market) pairs with a
**69.9% realized win rate** — over 60σ above the chance null. The
methodology composes five wallet-level features into a per-(wallet,
market) suspicion z-score, then flags the top 1%:

  f1  cross-sectional bet size  — z-score vs other traders in this market
  f2  within-trader bet size    — z-score vs this wallet's own history
  f3  realized profitability    — 60-day win rate on resolved positions
  f4  pre-event timing          — fraction of position established >24h
                                  before resolution
  f5  directional concentration — Herfindahl over YES vs NO across all
                                  open positions in the same category

Composite z-score = sum of standardised features. Top 1% by z goes on
a watch list. When any watch-list wallet adds ≥$500 to a position,
*copy-trade the same side on a 10–30 min lag* (avoids latency races;
captures the underlying information, not the speed).

How this differs from `wallet_orthogonality.py`
-----------------------------------------------
- That module is a binomial-tail test on realized win rate.
- This module is a behaviour-signature test: five features that
  *jointly* predict win rate without observing all resolutions.
- They are complementary, not redundant. A wallet flagged by both is
  the strongest signal; one-flag wallets are still useful.

Honest caveats
--------------
- The Mitts & Ofir paper is March 2026; the wallets flagged are likely
  already being copy-traded by other bots, so the 30-min lag will see
  a degraded version of the 69.9% headline. Bake in 15-20pp decay.
- Paper mode P&L is bounded by the same paper-fill optimism as the
  rest of the bot.

Storage
-------
- `mitts_ofir_features` — per-(wallet, market) feature snapshot
- `mitts_ofir_watchlist` — top-1% pairs with composite z and last seen
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class WalletMarketFeatures:
    wallet: str
    asset: str
    f1_cross_sectional_z: float
    f2_within_trader_z: float
    f3_profit_60d: float
    f4_pre_event_timing: float
    f5_directional_concentration: float
    composite_z: float
    n_trades_in_market: int


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS mitts_ofir_features (
            wallet TEXT NOT NULL,
            asset TEXT NOT NULL,
            f1_cross_sectional_z REAL NOT NULL,
            f2_within_trader_z REAL NOT NULL,
            f3_profit_60d REAL NOT NULL,
            f4_pre_event_timing REAL NOT NULL,
            f5_directional_concentration REAL NOT NULL,
            composite_z REAL NOT NULL,
            n_trades_in_market INTEGER NOT NULL,
            last_updated REAL NOT NULL,
            PRIMARY KEY (wallet, asset)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS mitts_ofir_watchlist (
            wallet TEXT NOT NULL,
            asset TEXT NOT NULL,
            composite_z REAL NOT NULL,
            last_position_size REAL,
            last_position_ts REAL,
            added_ts REAL NOT NULL,
            PRIMARY KEY (wallet, asset)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS mo_watch_recent ON mitts_ofir_watchlist(last_position_ts)"
    )
    conn.commit()


# ── Feature computations ───────────────────────────────────────────────
def _z_score(x: float, mean: float, std: float) -> float:
    """Standard z-score with std floor."""
    if std <= 1e-9:
        return 0.0
    return float((x - mean) / std)


def compute_features(
    conn: sqlite3.Connection,
    *,
    min_trades_per_wallet: int = 5,
    min_trades_per_market: int = 5,
    lookback_60d_sec: float = 60 * 86400.0,
) -> list[WalletMarketFeatures]:
    """Compute the 5-feature snapshot for every (wallet, market) pair
    in the `trades` table that meets minimum thresholds.

    Requires `trades` to have: wallet, asset, side, size, price, timestamp,
    and (for f3 profitability) `outcome_resolved` (only present on the
    historical_trades schema; otherwise f3 is left at 0).
    """
    ensure_tables(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if not {"wallet", "asset", "side", "size", "timestamp"}.issubset(cols):
        log.info("mo_screen_skip_no_required_cols", available=sorted(cols))
        return []
    has_resolved = "outcome_resolved" in cols
    has_category = "category" in cols
    has_market_id = "market_id" in cols

    now = time.time()
    cutoff_60d = now - lookback_60d_sec

    # ── Feature 1: cross-sectional bet size z (per-market normalisation) ──
    market_stats = {}
    for r in conn.execute(
        "SELECT asset, AVG(size), AVG(size*size), COUNT(*) "
        "FROM trades WHERE asset IS NOT NULL GROUP BY asset"
    ):
        asset, mu, mu_sq, n = r
        if (n or 0) < min_trades_per_market:
            continue
        var = max(0.0, float(mu_sq or 0) - float(mu or 0) ** 2)
        market_stats[asset] = {"mean": float(mu or 0), "std": math.sqrt(var), "n": int(n)}

    # ── Feature 2: within-trader bet size z (per-wallet normalisation) ──
    wallet_stats = {}
    for r in conn.execute(
        "SELECT wallet, AVG(size), AVG(size*size), COUNT(*) "
        "FROM trades WHERE wallet IS NOT NULL GROUP BY wallet"
    ):
        wallet, mu, mu_sq, n = r
        if (n or 0) < min_trades_per_wallet:
            continue
        var = max(0.0, float(mu_sq or 0) - float(mu or 0) ** 2)
        wallet_stats[wallet] = {"mean": float(mu or 0), "std": math.sqrt(var), "n": int(n)}

    # ── Feature 3: realised 60-day profitability per wallet ──
    profit_60d = {}
    if has_resolved:
        for r in conn.execute(
            """SELECT wallet,
                      SUM(CASE
                            WHEN (side='BUY'  AND outcome_resolved=1)
                              OR (side='SELL' AND outcome_resolved=0)
                            THEN 1 ELSE 0
                          END) AS wins,
                      COUNT(*) AS n
               FROM trades
               WHERE outcome_resolved IS NOT NULL
                 AND wallet IS NOT NULL
                 AND timestamp >= ?
               GROUP BY wallet
               HAVING n >= ?""",
            (cutoff_60d, min_trades_per_wallet),
        ):
            wallet, wins, n = r
            profit_60d[wallet] = float(wins or 0) / max(1, int(n or 0))

    # ── Aggregate per (wallet, market) ──
    rows = conn.execute(
        """SELECT wallet, asset, AVG(size) AS avg_size, COUNT(*) AS n,
                  MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts
           FROM trades
           WHERE wallet IS NOT NULL AND asset IS NOT NULL
           GROUP BY wallet, asset"""
    ).fetchall()
    out: list[WalletMarketFeatures] = []
    for wallet, asset, avg_size, n, first_ts, last_ts in rows:
        if (n or 0) < 1:
            continue
        ms = market_stats.get(asset)
        ws = wallet_stats.get(wallet)
        if ms is None or ws is None:
            continue
        f1 = _z_score(float(avg_size or 0), ms["mean"], ms["std"])
        f2 = _z_score(float(avg_size or 0), ws["mean"], ws["std"])
        f3 = profit_60d.get(wallet, 0.0)
        # f4 pre-event timing approximation: we don't have explicit
        # `resolved_ts` join here so we use last_ts of this market's
        # trade activity as a proxy for resolution. A position
        # established >24h before the last activity in the market gets
        # credit. Without per-position cost-basis snapshots this is
        # bounded by 0 (entire position is at first_ts).
        if first_ts is not None and last_ts is not None and last_ts > first_ts:
            f4 = 1.0 if (last_ts - first_ts) > 86400.0 else 0.0
        else:
            f4 = 0.0
        # f5 directional concentration: Herfindahl over BUY vs SELL
        # at the wallet level within this asset.
        side_rows = conn.execute(
            """SELECT side, SUM(size)
               FROM trades
               WHERE wallet=? AND asset=?
               GROUP BY side""",
            (wallet, asset),
        ).fetchall()
        total = sum(float(s or 0) for _, s in side_rows)
        if total > 0:
            shares = [(float(s or 0) / total) for _, s in side_rows]
            f5 = sum(p * p for p in shares)
        else:
            f5 = 0.0
        composite = f1 + f2 + f3 * 2.0 + f4 + f5
        out.append(WalletMarketFeatures(
            wallet=wallet, asset=asset,
            f1_cross_sectional_z=f1, f2_within_trader_z=f2,
            f3_profit_60d=f3, f4_pre_event_timing=f4,
            f5_directional_concentration=f5,
            composite_z=composite, n_trades_in_market=int(n or 0),
        ))
    return out


def compute_and_persist(
    conn: sqlite3.Connection, *, top_pct: float = 0.01,
) -> int:
    """Compute features, persist all of them, and update the watchlist
    with the top-`top_pct` fraction by composite z. Returns watchlist
    size."""
    ensure_tables(conn)
    feats = compute_features(conn)
    if not feats:
        log.info("mo_screen_no_features")
        return 0
    now = time.time()
    for f in feats:
        conn.execute(
            """INSERT INTO mitts_ofir_features
               (wallet, asset, f1_cross_sectional_z, f2_within_trader_z,
                f3_profit_60d, f4_pre_event_timing,
                f5_directional_concentration, composite_z,
                n_trades_in_market, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(wallet, asset) DO UPDATE SET
                   f1_cross_sectional_z=excluded.f1_cross_sectional_z,
                   f2_within_trader_z=excluded.f2_within_trader_z,
                   f3_profit_60d=excluded.f3_profit_60d,
                   f4_pre_event_timing=excluded.f4_pre_event_timing,
                   f5_directional_concentration=excluded.f5_directional_concentration,
                   composite_z=excluded.composite_z,
                   n_trades_in_market=excluded.n_trades_in_market,
                   last_updated=excluded.last_updated""",
            (f.wallet, f.asset, f.f1_cross_sectional_z, f.f2_within_trader_z,
             f.f3_profit_60d, f.f4_pre_event_timing,
             f.f5_directional_concentration, f.composite_z,
             f.n_trades_in_market, now),
        )
    # Take top fraction by composite_z
    sorted_feats = sorted(feats, key=lambda f: -f.composite_z)
    cutoff_idx = max(1, int(len(sorted_feats) * top_pct))
    watchlist = sorted_feats[:cutoff_idx]
    # Clear and re-insert the watchlist
    conn.execute("DELETE FROM mitts_ofir_watchlist")
    for f in watchlist:
        conn.execute(
            """INSERT INTO mitts_ofir_watchlist
               (wallet, asset, composite_z, last_position_size,
                last_position_ts, added_ts)
               VALUES (?, ?, ?, NULL, NULL, ?)""",
            (f.wallet, f.asset, f.composite_z, now),
        )
    conn.commit()
    log.info(
        "mo_screen_persist",
        n_features=len(feats),
        n_watchlist=len(watchlist),
        top_z=round(max(f.composite_z for f in feats), 2) if feats else None,
    )
    return len(watchlist)


def is_on_watchlist(conn: sqlite3.Connection, wallet: str, asset: str) -> bool:
    """O(1) lookup at decision time."""
    row = conn.execute(
        "SELECT 1 FROM mitts_ofir_watchlist WHERE wallet=? AND asset=?",
        (wallet, asset),
    ).fetchone()
    return row is not None


def recent_watchlist_entries(
    conn: sqlite3.Connection,
    *,
    since_ts: float | None = None,
    min_size: float = 500.0,
) -> list[dict]:
    """Return watch-list (wallet, asset) pairs whose latest position
    update is more recent than since_ts AND whose position is ≥ min_size.

    Used by the copy-trade trigger: "any flagged wallet adds ≥$500 to
    a position, paper-trade the same side 10–30 minutes later."
    """
    where = ["last_position_size IS NOT NULL", "last_position_size >= ?"]
    params: list = [min_size]
    if since_ts is not None:
        where.append("last_position_ts >= ?")
        params.append(since_ts)
    rows = conn.execute(
        f"""SELECT wallet, asset, composite_z, last_position_size, last_position_ts
            FROM mitts_ofir_watchlist
            WHERE {" AND ".join(where)}
            ORDER BY last_position_ts DESC""",
        params,
    ).fetchall()
    return [
        {"wallet": r[0], "asset": r[1], "composite_z": r[2],
         "last_position_size": r[3], "last_position_ts": r[4]}
        for r in rows
    ]


def record_watchlist_position(
    conn: sqlite3.Connection, wallet: str, asset: str,
    size: float, ts: float,
) -> None:
    """Update a watch-list pair's most-recent position. Called from
    the trade-tape watcher when an on-chain `OrderFilled` shows a
    flagged wallet adding to a position."""
    conn.execute(
        """UPDATE mitts_ofir_watchlist
           SET last_position_size=?, last_position_ts=?
           WHERE wallet=? AND asset=?""",
        (float(size), float(ts), wallet, asset),
    )
    conn.commit()
