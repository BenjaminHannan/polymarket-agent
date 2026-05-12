"""Tests for the Mitts & Ofir 5-signal screen."""
from __future__ import annotations

import sqlite3
import time

from polyagent.signals.mitts_ofir_screen import (
    ensure_tables, compute_features, compute_and_persist,
    is_on_watchlist, recent_watchlist_entries, record_watchlist_position,
    _z_score,
)


def _seed_trades(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE trades (
            tx_hash TEXT, wallet TEXT, asset TEXT, side TEXT,
            size REAL, price REAL, timestamp REAL,
            outcome_resolved INTEGER, category TEXT
        );
        """
    )


def test_z_score_zero_std_returns_zero():
    assert _z_score(5.0, 5.0, 0.0) == 0.0


def test_z_score_basic():
    assert abs(_z_score(10, 5, 2) - 2.5) < 1e-9


def test_ensure_tables_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    ensure_tables(conn)


def test_compute_features_skips_no_required_cols(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute("CREATE TABLE trades (foo TEXT)")
    out = compute_features(conn)
    assert out == []


def test_compute_features_basic(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    now = time.time()
    # Wallet A: 10 trades on asset M1 at large size (suspicious bet size)
    # Wallet B: 10 trades on asset M1 at small size (control)
    for i in range(10):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'A', 'M1', 'BUY', 100, 0.5, ?, 1, 'sports')",
            (f"tx_a_{i}", now - 3600 * (i + 1)),
        )
    for i in range(10):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'B', 'M1', 'BUY', 10, 0.5, ?, 0, 'sports')",
            (f"tx_b_{i}", now - 3600 * (i + 1)),
        )
    conn.commit()
    feats = compute_features(conn, min_trades_per_market=5, min_trades_per_wallet=5)
    assert len(feats) == 2
    feat_a = next(f for f in feats if f.wallet == "A")
    feat_b = next(f for f in feats if f.wallet == "B")
    # A's cross-sectional bet size should be higher than B's
    assert feat_a.f1_cross_sectional_z > feat_b.f1_cross_sectional_z
    # A's 60-day win rate should be 1.0; B's 0.0
    assert feat_a.f3_profit_60d == 1.0
    assert feat_b.f3_profit_60d == 0.0
    # Composite should be higher for A
    assert feat_a.composite_z > feat_b.composite_z


def test_compute_and_persist_top_pct(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    # 100 (wallet, asset) pairs with varying bet sizes
    for i in range(100):
        for j in range(5):
            conn.execute(
                "INSERT INTO trades VALUES (?, ?, ?, 'BUY', ?, 0.5, ?, 1, 'sports')",
                (f"tx_{i}_{j}", f"w{i:03d}", f"a{i:03d}", float(10 + i),
                 float(i * 5 + j)),
            )
    conn.commit()
    n = compute_and_persist(conn, top_pct=0.05)
    # 5% of 100 = 5 pairs (compute_features may return fewer if some are
    # filtered, but watchlist should be ≥1)
    assert n >= 1
    # Watchlist size matches the top 5% by composite_z
    rows = conn.execute("SELECT COUNT(*) FROM mitts_ofir_watchlist").fetchone()
    assert rows[0] == n


def test_watchlist_lookup(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    conn.execute(
        "INSERT INTO mitts_ofir_watchlist (wallet, asset, composite_z, "
        "last_position_size, last_position_ts, added_ts) VALUES "
        "('flag_wallet', 'asset1', 5.5, NULL, NULL, ?)",
        (time.time(),),
    )
    conn.commit()
    assert is_on_watchlist(conn, "flag_wallet", "asset1") is True
    assert is_on_watchlist(conn, "missing", "asset1") is False


def test_record_watchlist_position(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    conn.execute(
        "INSERT INTO mitts_ofir_watchlist (wallet, asset, composite_z, "
        "added_ts) VALUES ('flag', 'a1', 4.0, ?)",
        (time.time(),),
    )
    conn.commit()
    record_watchlist_position(conn, "flag", "a1", size=750, ts=time.time())
    rows = recent_watchlist_entries(conn, since_ts=time.time() - 60, min_size=500)
    assert len(rows) == 1
    assert rows[0]["wallet"] == "flag"
    assert rows[0]["last_position_size"] == 750.0


def test_recent_watchlist_filters_below_min_size(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    now = time.time()
    conn.execute(
        "INSERT INTO mitts_ofir_watchlist (wallet, asset, composite_z, "
        "last_position_size, last_position_ts, added_ts) VALUES "
        "('w1', 'a1', 3, 300, ?, ?), "
        "('w2', 'a2', 4, 600, ?, ?)",
        (now, now, now, now),
    )
    conn.commit()
    rows = recent_watchlist_entries(conn, since_ts=now - 60, min_size=500)
    assert len(rows) == 1
    assert rows[0]["wallet"] == "w2"
