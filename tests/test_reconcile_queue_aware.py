"""Tests for queue-aware reconciliation script."""
from __future__ import annotations

import sqlite3

from scripts.reconcile_queue_aware import reconcile


def _seed_fills(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE fills (
            ts REAL, strategy TEXT, condition_id TEXT, token_id TEXT,
            side TEXT, size REAL, price REAL, fee REAL, reason TEXT
        );
        CREATE TABLE fills_shadow_queue (
            ts REAL, token_id TEXT, side TEXT,
            walked_vwap_price REAL, queue_aware_price REAL
        );
        """
    )


def test_no_shadow_table_zero_haircut(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        "CREATE TABLE fills (ts REAL, strategy TEXT, condition_id TEXT, "
        "token_id TEXT, side TEXT, size REAL, price REAL, fee REAL, reason TEXT)"
    )
    conn.execute(
        "INSERT INTO fills VALUES (1.0, 'combined_trader', 'c1', 'tok1', "
        "'BUY', 100, 0.50, 0.5, 'test')"
    )
    conn.commit()
    res = reconcile(str(tmp_path / "t.db"))
    assert res["n_fills"] == 1
    assert res["has_shadow_table"] is False
    # Without shadow, both haircuts should be 0 (we treat absent shadow as walked=naive).
    b = res["by_strategy"]["combined_trader"]
    assert b["walked_pnl_usd"] == 0.0


def test_shadow_haircut_present(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_fills(conn)
    # Fill BUY 100 shares at 0.50; walked VWAP would have been 0.52 (worse for us).
    conn.execute(
        "INSERT INTO fills VALUES (1.0, 'combined_trader', 'c1', 'tok1', "
        "'BUY', 100, 0.50, 0.5, 'test')"
    )
    conn.execute(
        "INSERT INTO fills_shadow_queue VALUES (1.0, 'tok1', 'BUY', 0.52, 0.55)"
    )
    conn.commit()
    res = reconcile(str(tmp_path / "t.db"))
    assert res["n_fills"] == 1
    assert res["has_shadow_table"] is True
    b = res["by_strategy"]["combined_trader"]
    # walked − naive = 0.02 worse × 100 = $2 haircut
    assert abs(b["walked_pnl_usd"] - (-2.0)) < 1e-6
    # queue_aware haircut bigger
    assert abs(b["queue_aware_pnl_usd"] - (-5.0)) < 1e-6


def test_strategy_filter(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_fills(conn)
    conn.execute(
        "INSERT INTO fills VALUES (1.0, 'A', 'c1', 'tok1', 'BUY', 100, 0.50, 0.5, '')"
    )
    conn.execute(
        "INSERT INTO fills VALUES (2.0, 'B', 'c1', 'tok1', 'BUY', 100, 0.50, 0.5, '')"
    )
    conn.commit()
    res = reconcile(str(tmp_path / "t.db"), strategy="A")
    assert res["n_fills"] == 1
    assert "A" in res["by_strategy"]
    assert "B" not in res["by_strategy"]
