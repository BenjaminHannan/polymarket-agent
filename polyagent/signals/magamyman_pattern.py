"""Magamyman pattern detector — the highest-conviction insider signal.

Named after the canonical wallet in [[Iran Strike Polymarket Insider
Case]]: 71 minutes before the Feb 28, 2026 U.S.-Israeli strike on Iran,
a fresh Polymarket wallet ("Magamyman") bought "Yes" shares in the
"US strikes Iran by February 28?" contract at $0.10 when the implied
market probability was 17%. Realized profit ~$553,000.

This is a *specialization* of the Mitts-Ofir 5-signal screen. M-O
catches it in aggregate (it's one of the ~210K flagged pairs at 69.9%
win rate), but the Magamyman pattern is the highest-confidence subset:

  - Wallet is "fresh" — created within the last 30 days
  - First (or near-first) trade is on this single market
  - Trade is at an extreme-low price (≤15 cents) on a longshot outcome
  - Position size is large in absolute terms (≥$1000 notional)
  - Position is built before the resolution event

When *all* of these conditions co-occur, the post-hoc base rate per
Mitts-Ofir is much higher than the population 69.9%. The 60σ
permutation test on the M-O screen is dominated by a small number of
Magamyman-style cases; this detector tries to find them at decision
time so they get higher copy-fraction in the copy-trader.

When a Magamyman-pattern entry is detected, we mark it in the
`mitts_ofir_watchlist` table with `super_signal=1` so the copy-trader
applies a 50% copy fraction instead of the default 25%.

Honest caveat
-------------
This is a high-precision low-recall detector. Most days it fires zero
times. The Iran case is rare. But when it fires, the prior is much
stronger than the M-O baseline.
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class MagamymanCandidate:
    """A wallet × market pair matching the canonical insider pattern."""
    wallet: str
    asset: str
    first_trade_ts: float
    wallet_age_at_first_trade_sec: float
    entry_price: float
    notional_usd: float
    side: str       # "BUY" expected for the canonical pattern (cheap longshot YES)
    n_prior_trades: int


def detect_candidates(
    conn: sqlite3.Connection,
    *,
    max_wallet_age_sec: float = 30 * 86400.0,   # 30 days
    max_entry_price: float = 0.15,              # ≤15c
    min_notional_usd: float = 1000.0,
    max_prior_trades: int = 5,                  # near-first trade
    lookback_sec: float = 7 * 86400.0,          # only scan recent week
) -> list[MagamymanCandidate]:
    """Scan the trades table for wallet × market pairs matching the
    Magamyman pattern.

    Returns an empty list when the required columns aren't present —
    we need `wallet`, `asset`, `side`, `size`, `price`, `timestamp`.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    needed = {"wallet", "asset", "side", "size", "price", "timestamp"}
    if not needed.issubset(cols):
        log.info("magamyman_skip_no_cols", missing=sorted(needed - cols))
        return []
    now = time.time()
    cutoff = now - lookback_sec

    # For each (wallet, asset) pair in the lookback, find:
    #   - first_trade_ts (earliest trade on this asset)
    #   - wallet's first_ever_trade_ts (across all assets)
    #   - n_prior_trades (count of trades by this wallet before first
    #     trade on this asset, on OTHER assets)
    #   - entry_price + notional at first trade
    rows = conn.execute(
        """SELECT t.wallet, t.asset, t.side, t.size, t.price, t.timestamp
           FROM trades t
           WHERE t.timestamp >= ?
             AND t.wallet IS NOT NULL AND t.asset IS NOT NULL
             AND (t.side IS NULL OR upper(t.side) = 'BUY')
             AND t.price <= ?
           ORDER BY t.timestamp ASC""",
        (cutoff, float(max_entry_price)),
    ).fetchall()
    if not rows:
        return []

    # Index by wallet so we can compute wallet age and prior-trade count
    # without N+1 queries.
    wallet_trade_history: dict[str, list] = {}
    for w, a, side, size, price, ts in rows:
        wallet_trade_history.setdefault(w, []).append({
            "asset": a, "side": (side or "").upper(),
            "size": float(size or 0), "price": float(price or 0),
            "ts": float(ts or 0),
        })

    # Bulk fetch first_ever_trade_ts per wallet (across all time, not
    # just lookback).
    wallets = list(wallet_trade_history.keys())
    first_seen: dict[str, float] = {}
    if wallets:
        chunk = 500
        for i in range(0, len(wallets), chunk):
            sub = wallets[i:i + chunk]
            placeholders = ",".join(["?"] * len(sub))
            for w, mn in conn.execute(
                f"SELECT wallet, MIN(timestamp) FROM trades "
                f"WHERE wallet IN ({placeholders}) GROUP BY wallet",
                sub,
            ):
                first_seen[w] = float(mn or 0)

    out: list[MagamymanCandidate] = []
    for wallet, trades in wallet_trade_history.items():
        trades.sort(key=lambda t: t["ts"])
        # Group by (wallet, asset) — first trade on each new asset.
        seen_assets: set[str] = set()
        for tr in trades:
            asset = tr["asset"]
            if asset in seen_assets:
                continue
            seen_assets.add(asset)
            first_trade_ts = tr["ts"]
            wallet_first_seen = first_seen.get(wallet, first_trade_ts)
            wallet_age_at_first = first_trade_ts - wallet_first_seen
            if wallet_age_at_first > max_wallet_age_sec:
                continue
            # Count prior trades by this wallet (on any asset)
            n_prior = sum(1 for t in trades if t["ts"] < first_trade_ts)
            if n_prior > max_prior_trades:
                continue
            notional = tr["price"] * tr["size"]
            if notional < min_notional_usd:
                continue
            out.append(MagamymanCandidate(
                wallet=wallet, asset=asset,
                first_trade_ts=first_trade_ts,
                wallet_age_at_first_trade_sec=wallet_age_at_first,
                entry_price=tr["price"],
                notional_usd=notional,
                side=tr["side"],
                n_prior_trades=n_prior,
            ))
    if out:
        log.info(
            "magamyman_candidates",
            n=len(out),
            sample_wallet=out[0].wallet[:14] if out else None,
            top_notional=round(max(c.notional_usd for c in out), 2),
        )
    return out


def flag_in_watchlist(conn: sqlite3.Connection, candidates: list[MagamymanCandidate]) -> int:
    """Mark each Magamyman candidate in the `mitts_ofir_watchlist` with
    super_signal=1. Returns count actually flagged (must already be on
    the M-O watchlist; Magamyman is a *specialization*, not a parallel
    list)."""
    # Add column if missing
    cols = {r[1] for r in conn.execute("PRAGMA table_info(mitts_ofir_watchlist)")}
    if "super_signal" not in cols:
        conn.execute(
            "ALTER TABLE mitts_ofir_watchlist ADD COLUMN super_signal INTEGER DEFAULT 0"
        )
    n = 0
    for c in candidates:
        # Only flag if the (wallet, asset) is already on the M-O watchlist.
        # The two screens are complementary — Magamyman tightens, doesn't
        # broaden.
        row = conn.execute(
            "SELECT 1 FROM mitts_ofir_watchlist WHERE wallet=? AND asset=?",
            (c.wallet, c.asset),
        ).fetchone()
        if not row:
            # Not on M-O watchlist; record-only via mitts_ofir_features
            # if we want, but don't elevate.
            continue
        conn.execute(
            "UPDATE mitts_ofir_watchlist SET super_signal=1 "
            "WHERE wallet=? AND asset=?",
            (c.wallet, c.asset),
        )
        n += 1
    conn.commit()
    if n:
        log.info("magamyman_flagged", n=n)
    return n


def is_super_signal(conn: sqlite3.Connection, wallet: str, asset: str) -> bool:
    """Lookup at copy-trade decision time."""
    try:
        row = conn.execute(
            "SELECT COALESCE(super_signal, 0) FROM mitts_ofir_watchlist "
            "WHERE wallet=? AND asset=?",
            (wallet, asset),
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row and row[0])
