"""Tests for the Polymarket Liquidity Rewards API poller."""
from __future__ import annotations

import sqlite3
import time

from polyagent.data.polymarket_liquidity_rewards import (
    ensure_table, persist_state, lookup, is_market_eligible,
    eligible_pool_for_market, total_eligible_pool_usd,
    RewardsState,
)


def test_ensure_table_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    ensure_table(conn)
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(polymarket_liquidity_rewards)"
    )}
    assert "condition_id" in cols
    assert "is_eligible" in cols
    assert "pool_size_usd" in cols


def test_persist_then_lookup(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    s = RewardsState(
        condition_id="c1", is_eligible=True,
        pool_size_usd=500.0, max_spread_bps=300.0,
        min_size_at_quote=50.0,
    )
    persist_state(conn, s)
    got = lookup(conn, "c1")
    assert got is not None
    assert got.is_eligible is True
    assert got.pool_size_usd == 500.0


def test_lookup_missing_returns_none(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    assert lookup(conn, "no-such") is None


def test_is_eligible_helper(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    persist_state(conn, RewardsState(
        condition_id="c1", is_eligible=True, pool_size_usd=100.0,
        max_spread_bps=300.0, min_size_at_quote=10.0,
    ))
    persist_state(conn, RewardsState(
        condition_id="c2", is_eligible=False, pool_size_usd=0.0,
        max_spread_bps=0.0, min_size_at_quote=0.0,
    ))
    assert is_market_eligible(conn, "c1") is True
    assert is_market_eligible(conn, "c2") is False
    assert is_market_eligible(conn, "c3") is False


def test_eligible_pool_for_market(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    persist_state(conn, RewardsState(
        condition_id="c1", is_eligible=True, pool_size_usd=750.0,
        max_spread_bps=300.0, min_size_at_quote=50.0,
    ))
    assert eligible_pool_for_market(conn, "c1") == 750.0
    persist_state(conn, RewardsState(
        condition_id="c2", is_eligible=False, pool_size_usd=999.0,
        max_spread_bps=300.0, min_size_at_quote=50.0,
    ))
    # Not eligible — pool reported as 0.0
    assert eligible_pool_for_market(conn, "c2") == 0.0


def test_total_eligible_pool_sums_only_eligible(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    for i, eligible in enumerate([True, True, False, True]):
        persist_state(conn, RewardsState(
            condition_id=f"c{i}", is_eligible=eligible,
            pool_size_usd=100.0, max_spread_bps=300.0,
            min_size_at_quote=50.0,
        ))
    # 3 eligible × 100 = 300
    assert total_eligible_pool_usd(conn) == 300.0
