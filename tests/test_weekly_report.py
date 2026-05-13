"""Tests for polyagent.eval.weekly_report.

Build a tiny synthetic SQLite that exercises:
  - per-strategy realized Sharpe with one resolved win + one resolved loss
  - hit rate by category aggregation
  - combined-signal calibration buckets
"""

import json
import os
import sqlite3
import sys
import tempfile

import pytest

# weekly_report imports config which reads env. Force a known db path before import.
@pytest.fixture
def synthetic_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE fills (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, strategy TEXT, condition_id TEXT, token_id TEXT,
            side TEXT, price REAL, size REAL, notional REAL, reason TEXT);
        CREATE TABLE resolutions (condition_id TEXT PRIMARY KEY,
            resolved_ts REAL, yes_won INTEGER,
            yes_token_id TEXT, no_token_id TEXT,
            yes_size REAL, no_size REAL,
            yes_avg_cost REAL, no_avg_cost REAL,
            yes_payout REAL, no_payout REAL,
            pnl REAL, detail TEXT);
        CREATE TABLE nav_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, cash REAL, position_value REAL, nav REAL);
        CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, strategy TEXT, condition_id TEXT,
            direction TEXT, score REAL, news_hash TEXT, detail TEXT);
        CREATE TABLE signal_outcomes (
            condition_id TEXT PRIMARY KEY, resolved_ts REAL, yes_won INTEGER,
            question TEXT, p_stat_lgbm REAL, p_news_match REAL,
            p_market_pre REAL, n_news_signals INTEGER, detail TEXT,
            p_market_1h REAL, p_market_6h REAL, p_market_24h REAL);
        """
    )
    # 2 markets: one win (combined_trader BUY YES, YES wins),
    # one loss (passive_poster BUY YES, YES loses).
    # Use entry ts = 100, 200; resolution ts = 100000 (day 1+).
    con.execute(
        "INSERT INTO fills(ts, strategy, condition_id, token_id, side, price, size, notional, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (100.0, "combined_trader", "C1", "T1_YES", "BUY", 0.40, 10.0, 4.0, "x"),
    )
    con.execute(
        "INSERT INTO fills(ts, strategy, condition_id, token_id, side, price, size, notional, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (200.0, "passive_poster", "C2", "T2_YES", "BUY", 0.60, 5.0, 3.0, "x"),
    )
    con.execute(
        "INSERT INTO resolutions(condition_id, resolved_ts, yes_won, yes_token_id, no_token_id, "
        "yes_size, no_size, yes_avg_cost, no_avg_cost, yes_payout, no_payout, pnl, detail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("C1", 100000.0, 1, "T1_YES", "T1_NO", 10.0, 0.0, 0.40, 0.0, 1.0, 0.0, 6.0,
         json.dumps({"question": "Will Bitcoin pass 100k?"})),
    )
    con.execute(
        "INSERT INTO resolutions(condition_id, resolved_ts, yes_won, yes_token_id, no_token_id, "
        "yes_size, no_size, yes_avg_cost, no_avg_cost, yes_payout, no_payout, pnl, detail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("C2", 200000.0, 0, "T2_YES", "T2_NO", 5.0, 0.0, 0.60, 0.0, 0.0, 1.0, -3.0,
         json.dumps({"question": "Will the Lakers win?"})),
    )
    # Two combined signal rows for calibration: one well-calibrated, one wildly off.
    con.execute(
        "INSERT INTO signals(ts, strategy, condition_id, direction, score, news_hash, detail) "
        "VALUES (?,?,?,?,?,?,?)",
        (50.0, "combined", "C1", "yes", 0.2,
         "", json.dumps({"p_combined": 0.85, "p_market": 0.55})),
    )
    con.execute(
        "INSERT INTO signals(ts, strategy, condition_id, direction, score, news_hash, detail) "
        "VALUES (?,?,?,?,?,?,?)",
        (50.0, "combined", "C2", "yes", 0.3,
         "", json.dumps({"p_combined": 0.85, "p_market": 0.50})),
    )
    con.commit()
    con.close()
    yield path
    os.unlink(path)


def test_per_strategy_sharpe_win_and_loss(synthetic_db):
    from polyagent.eval.weekly_report import per_strategy_sharpe
    con = sqlite3.connect(synthetic_db)
    try:
        out = per_strategy_sharpe(con)
    finally:
        con.close()
    assert "combined_trader" in out
    assert "passive_poster" in out
    # combined_trader bought 10@0.40, resolved at $1 -> pnl = 6.0
    assert out["combined_trader"]["total_pnl"] == pytest.approx(6.0)
    # passive_poster bought 5@0.60, resolved at $0 -> pnl = -3.0
    assert out["passive_poster"]["total_pnl"] == pytest.approx(-3.0)
    assert out["combined_trader"]["trades"] == 1
    assert out["passive_poster"]["trades"] == 1


def test_hit_rate_by_category(synthetic_db):
    from polyagent.eval.weekly_report import hit_rate_by_category
    con = sqlite3.connect(synthetic_db)
    try:
        out = hit_rate_by_category(con)
    finally:
        con.close()
    # 2 fills, 1 win, 1 loss across whatever categories the categorizer assigns.
    total_n = sum(b["n"] for b in out.values())
    total_wins = sum(b["wins"] for b in out.values())
    assert total_n == 2
    assert total_wins == 1


def test_combined_calibration_brier(synthetic_db):
    from polyagent.eval.weekly_report import combined_calibration
    con = sqlite3.connect(synthetic_db)
    try:
        out = combined_calibration(con, bins=10)
    finally:
        con.close()
    # Two signal rows, one for each resolved market. Bucket [0.8, 0.9) gets both.
    bucket = out["buckets"][8]
    assert bucket["n"] == 2
    # p_combined was 0.85 for both; yes_won = [1, 0] -> realized 0.5
    assert bucket["realized_yes_rate"] == pytest.approx(0.5)
    # Brier for the model: 0.5 * ((1-0.85)^2 + (0-0.85)^2) = 0.5 * (0.0225 + 0.7225) = 0.3725
    assert bucket["model_brier"] == pytest.approx(0.3725, abs=1e-4)
    # Market gives 0.55 and 0.50: 0.5 * ((1-0.55)^2 + (0-0.50)^2) = 0.5 * (0.2025 + 0.25) = 0.22625
    assert bucket["market_brier"] == pytest.approx(0.22625, abs=1e-4)
    assert out["overall"]["n"] == 2
