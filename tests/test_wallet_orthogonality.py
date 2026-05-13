"""Tests for Della Vedova wallet-orthogonality classifier."""
from __future__ import annotations

import sqlite3

from polyagent.signals.wallet_orthogonality import (
    _binom_sf,
    compute_wallet_stats,
    ensure_table,
    is_wallet_informed,
)


def test_binom_sf_extremes():
    # All wins: extreme tail.
    assert _binom_sf(100, 100) < 1e-20
    # Half wins: exactly fair.
    p = _binom_sf(100, 50)
    assert 0.4 < p < 0.6  # near 0.5 since P(X >= 50) under Binom(100, 0.5)
    # Zero wins required: always 1.0.
    assert _binom_sf(100, 0) == 1.0
    # Impossible: k > n.
    assert _binom_sf(50, 100) == 0.0


def test_compute_stats_skips_below_min_trades(tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE trades (
            tx_hash TEXT, wallet TEXT, asset TEXT, side TEXT,
            size REAL, price REAL, timestamp REAL,
            market_id TEXT, outcome_resolved INTEGER, category TEXT
        );
        """
    )
    # Wallet with only 10 trades — below min_trades default 50.
    for i in range(10):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w1', 'a1', 'BUY', 100, 0.5, 0, 'm1', 1, 'sports')",
            (str(i),),
        )
    conn.commit()
    stats = compute_wallet_stats(conn, min_trades=50)
    assert stats == []


def test_compute_stats_flags_informed_wallet(tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE trades (
            tx_hash TEXT, wallet TEXT, asset TEXT, side TEXT,
            size REAL, price REAL, timestamp REAL,
            market_id TEXT, outcome_resolved INTEGER, category TEXT
        );
        """
    )
    # Wallet w_informed: 100 trades, 80 wins. P(X >= 80) under Binom(100, 0.5)
    # = 5.6e-10 ⇒ deeply informed at p < 0.01.
    for i in range(80):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w_informed', 'a1', 'BUY', 100, 0.5, 0, ?, 1, 'sports')",
            (f"tx_winner_{i}", f"m_{i}"),
        )
    for i in range(20):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w_informed', 'a1', 'BUY', 100, 0.5, 0, ?, 0, 'sports')",
            (f"tx_loser_{i}", f"m_{80+i}"),
        )
    conn.commit()
    stats = compute_wallet_stats(conn, p_threshold=0.01, min_trades=50)
    assert len(stats) == 1
    s = stats[0]
    assert s.wallet == "w_informed"
    assert s.n_trades == 100
    assert s.n_wins == 80
    assert s.is_informed is True
    assert s.binom_p < 0.01
    # is_wallet_informed lookup
    assert is_wallet_informed(conn, "w_informed") is True
    assert is_wallet_informed(conn, "unknown") is False


def test_ensure_table_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    ensure_table(conn)  # should not raise
