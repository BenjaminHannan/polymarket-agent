"""Tests for the Polymarket native-features poller."""
from __future__ import annotations

import sqlite3
import time

from polyagent.data.polymarket_native import (
    ensure_table, lookup_features, NativeFeatures,
)


def test_ensure_table_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    ensure_table(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(polymarket_native_features)")}
    assert "comment_count_6h" in cols
    assert "comment_count_delta_6h" in cols
    assert "top_trader_inflow_24h" in cols
    assert "unique_traders_1h" in cols


def test_lookup_returns_none_on_missing(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    assert lookup_features(conn, "no-such-cid") is None


def test_lookup_returns_features_on_hit(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    now = time.time()
    conn.execute(
        """INSERT INTO polymarket_native_features
           (condition_id, comment_count_6h, comment_count_delta_6h,
            top_trader_inflow_24h, unique_traders_1h, last_updated)
           VALUES ('c1', 42, 5, 1234.56, 17, ?)""",
        (now,),
    )
    conn.commit()
    f = lookup_features(conn, "c1")
    assert f is not None
    assert f.comment_count_6h == 42
    assert f.comment_count_delta_6h == 5
    assert abs(f.top_trader_inflow_24h - 1234.56) < 1e-6
    assert f.unique_traders_1h == 17
