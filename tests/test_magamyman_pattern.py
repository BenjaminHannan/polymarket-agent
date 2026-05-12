"""Tests for the Magamyman pattern detector."""
from __future__ import annotations

import sqlite3
import time

from polyagent.signals.magamyman_pattern import (
    detect_candidates, flag_in_watchlist, is_super_signal,
    MagamymanCandidate,
)


def _seed_trades(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE trades (
            tx_hash TEXT, wallet TEXT, asset TEXT, side TEXT,
            size REAL, price REAL, timestamp REAL,
            outcome_resolved INTEGER, category TEXT
        );
        CREATE TABLE mitts_ofir_watchlist (
            wallet TEXT NOT NULL, asset TEXT NOT NULL,
            composite_z REAL NOT NULL, last_position_size REAL,
            last_position_ts REAL, added_ts REAL NOT NULL,
            PRIMARY KEY (wallet, asset)
        );
        """
    )


def test_detect_no_required_cols_returns_empty(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute("CREATE TABLE trades (foo TEXT)")
    assert detect_candidates(conn) == []


def test_detect_classic_magamyman_pattern(tmp_path):
    """A fresh wallet (created within 30 days) making a single BUY at
    $0.10 for $10K notional should be flagged."""
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    # Fresh wallet — first trade is also wallet's first trade
    conn.execute(
        "INSERT INTO trades VALUES ('tx1', 'magamyman', 'asset_iran', 'BUY', "
        "10000, 0.10, ?, NULL, 'geo')",
        (now - 86400,),  # 1 day ago
    )
    conn.commit()
    cands = detect_candidates(
        conn, max_wallet_age_sec=30 * 86400, max_entry_price=0.15,
        min_notional_usd=1000, max_prior_trades=5,
    )
    assert len(cands) == 1
    c = cands[0]
    assert c.wallet == "magamyman"
    assert c.asset == "asset_iran"
    assert c.entry_price == 0.10
    assert c.notional_usd == 1000.0  # 10000 × 0.10
    assert c.n_prior_trades == 0


def test_detect_skips_old_wallet(tmp_path):
    """Wallet whose first trade is >30 days before this trade is not fresh."""
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    # Wallet's first trade was 60 days ago
    conn.execute(
        "INSERT INTO trades VALUES ('tx_old', 'veteran', 'some_other', 'BUY', "
        "100, 0.05, ?, NULL, 'other')",
        (now - 60 * 86400,),
    )
    # Their "new market" entry — within lookback
    conn.execute(
        "INSERT INTO trades VALUES ('tx_new', 'veteran', 'asset_iran', 'BUY', "
        "20000, 0.10, ?, NULL, 'geo')",
        (now - 86400,),
    )
    conn.commit()
    cands = detect_candidates(conn, max_wallet_age_sec=30 * 86400)
    # Wallet is 60 days old → not fresh → not flagged
    assert cands == []


def test_detect_skips_high_price(tmp_path):
    """Entry at $0.20 should NOT match the canonical pattern (need ≤$0.15)."""
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    conn.execute(
        "INSERT INTO trades VALUES ('tx1', 'w1', 'a1', 'BUY', 10000, 0.20, ?, NULL, 'sports')",
        (now - 86400,),
    )
    conn.commit()
    cands = detect_candidates(conn, max_entry_price=0.15)
    assert cands == []


def test_detect_skips_small_notional(tmp_path):
    """Below $1K notional doesn't match."""
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    conn.execute(
        "INSERT INTO trades VALUES ('tx1', 'w1', 'a1', 'BUY', 100, 0.10, ?, NULL, 'sports')",
        (now - 3600,),
    )
    conn.commit()
    # 100 × 0.10 = $10 < $1000 floor
    cands = detect_candidates(conn, min_notional_usd=1000)
    assert cands == []


def test_flag_in_watchlist_requires_existing_mo_entry(tmp_path):
    """Magamyman doesn't broaden the watchlist; it elevates existing entries."""
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    cands = [MagamymanCandidate(
        wallet="not_on_watchlist", asset="some_asset",
        first_trade_ts=time.time(), wallet_age_at_first_trade_sec=86400,
        entry_price=0.10, notional_usd=5000.0, side="BUY",
        n_prior_trades=0,
    )]
    # Watchlist is empty → no candidates can be flagged
    assert flag_in_watchlist(conn, cands) == 0


def test_flag_in_watchlist_elevates_existing_entry(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    conn.execute(
        "INSERT INTO mitts_ofir_watchlist VALUES ('mo_w', 'mo_a', 4.5, NULL, NULL, ?)",
        (now,),
    )
    conn.commit()
    cands = [MagamymanCandidate(
        wallet="mo_w", asset="mo_a",
        first_trade_ts=now, wallet_age_at_first_trade_sec=86400,
        entry_price=0.10, notional_usd=5000.0, side="BUY",
        n_prior_trades=0,
    )]
    n = flag_in_watchlist(conn, cands)
    assert n == 1
    # Now check the super_signal flag is on
    assert is_super_signal(conn, "mo_w", "mo_a") is True
    # And a different (wallet, asset) is NOT super signal
    assert is_super_signal(conn, "other", "other") is False


def test_is_super_signal_missing_returns_false(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    assert is_super_signal(conn, "no_wallet", "no_asset") is False
