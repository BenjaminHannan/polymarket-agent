"""Tests for round_trip_legs FIFO matching ledger."""
from __future__ import annotations

import sqlite3

from polyagent.risk.round_trips import (
    FillContext,
    ensure_table,
    record_fill,
    strategy_summary,
)


def _conn():
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    return c


def _buy(conn, fid, size, price, ts=0, fees=0.0, rebate=0.0, strategy="m", token="t1"):
    return record_fill(conn, FillContext(
        fill_id=fid, strategy=strategy, condition_id="0xc",
        token_id=token, side="BUY", price=price, size=size, ts=ts,
        fees_paid=fees, rebate_credited=rebate,
    ))


def _sell(conn, fid, size, price, ts=0, fees=0.0, rebate=0.0, strategy="m", token="t1"):
    return record_fill(conn, FillContext(
        fill_id=fid, strategy=strategy, condition_id="0xc",
        token_id=token, side="SELL", price=price, size=size, ts=ts,
        fees_paid=fees, rebate_credited=rebate,
    ))


def test_buy_creates_open_leg():
    c = _conn()
    s = _buy(c, fid=1, size=10, price=0.40, ts=100)
    assert s["opened"] == 1
    rows = c.execute("SELECT size, open_price, close_ts FROM round_trip_legs").fetchall()
    assert len(rows) == 1
    assert rows[0] == (10, 0.40, None)


def test_simple_full_close():
    c = _conn()
    _buy(c, 1, 10, 0.40, ts=100)
    s = _sell(c, 2, 10, 0.50, ts=200)
    assert len(s["closed_legs"]) == 1
    assert abs(s["closed_legs"][0]["gross_pnl"] - 1.0) < 1e-9  # 10 * 0.10
    rows = c.execute("SELECT close_price, gross_pnl FROM round_trip_legs WHERE close_ts IS NOT NULL").fetchall()
    assert rows[0][0] == 0.50
    assert abs(rows[0][1] - 1.0) < 1e-9


def test_fifo_partial_close_splits_leg():
    c = _conn()
    _buy(c, 1, 30, 0.40, ts=100)
    _buy(c, 2, 20, 0.45, ts=200)
    # Sell 35 at 0.50: should fully close the first leg (30) + partially close the second (5 of 20)
    s = _sell(c, 3, 35, 0.50, ts=300)
    assert len(s["closed_legs"]) == 2
    # Total closed = 30 + 5 = 35 shares
    closed_sizes = sorted(l["size"] for l in s["closed_legs"])
    assert closed_sizes == [5, 30]
    # Remaining open: 15 shares of leg-2 at 0.45
    open_rows = c.execute(
        "SELECT size, open_price FROM round_trip_legs WHERE close_ts IS NULL"
    ).fetchall()
    assert len(open_rows) == 1
    assert open_rows[0][0] == 15
    assert open_rows[0][1] == 0.45


def test_pnl_includes_fees_and_rebates():
    c = _conn()
    # Maker BUY (no fee, rebate $0.10) at 0.40
    _buy(c, 1, 10, 0.40, ts=100, fees=0.0, rebate=0.10)
    # Taker SELL (fee $0.05) at 0.50
    s = _sell(c, 2, 10, 0.50, ts=200, fees=0.05, rebate=0.0)
    assert len(s["closed_legs"]) == 1
    leg = s["closed_legs"][0]
    # gross = (0.50 - 0.40) × 10 = 1.0
    # net = 1.0 - (-0.10) - 0.05 = 1.0 + 0.10 - 0.05 = 1.05
    assert abs(leg["gross_pnl"] - 1.0) < 1e-9
    assert abs(leg["net_pnl"] - 1.05) < 1e-9


def test_oversell_creates_short_leg():
    c = _conn()
    _buy(c, 1, 10, 0.40, ts=100)
    s = _sell(c, 2, 25, 0.50, ts=200)  # 15 over the inventory
    assert s["remaining_to_close"] == 15
    # Long 10 closed; short 15 opened
    open_short = c.execute(
        "SELECT size, open_price FROM round_trip_legs WHERE close_ts IS NULL"
    ).fetchall()
    assert len(open_short) == 1
    assert open_short[0][0] == -15  # negative size denotes short
    assert open_short[0][1] == 0.50


def test_strategy_summary_aggregates():
    c = _conn()
    _buy(c, 1, 10, 0.40, ts=100, rebate=0.05, strategy="X")
    _sell(c, 2, 10, 0.50, ts=200, rebate=0.05, strategy="X")
    _buy(c, 3, 5, 0.30, ts=300, strategy="X")  # left open
    s = strategy_summary(c, "X")
    assert s["closed_round_trips"] == 1
    assert s["open_legs"] == 1
    assert s["open_size_total"] == 5
    assert abs(s["gross_pnl_realized"] - 1.0) < 1e-9
    # net_pnl: gross 1.0 + rebate on open 0.05 + rebate on close 0.05 = 1.10
    assert abs(s["net_pnl_realized"] - 1.10) < 1e-9


def test_separate_tokens_dont_match():
    c = _conn()
    _buy(c, 1, 10, 0.40, token="t_yes")
    s = _sell(c, 2, 10, 0.50, token="t_no")  # different token!
    # Should NOT match — t_no SELL becomes its own short leg, t_yes BUY stays open
    assert s["closed_legs"] == []
    assert s["remaining_to_close"] == 10
    open_legs = c.execute(
        "SELECT token_id, size FROM round_trip_legs WHERE close_ts IS NULL"
    ).fetchall()
    # Two open legs: long t_yes 10, short t_no 10
    assert sorted(open_legs) == [("t_no", -10), ("t_yes", 10)]
