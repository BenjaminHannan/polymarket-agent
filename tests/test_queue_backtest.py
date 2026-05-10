"""Tests for the queue-aware backtest replay framework."""
from __future__ import annotations

import sqlite3

from polyagent.eval.queue_backtest import (
    FillReplay,
    aggregate_haircut,
    replay_fills_for_strategy,
)


def _setup_db(tmp_path):
    """Build an in-fixture DB with the minimum schema needed by the
    backtest: fills, fills_shadow_queue, book_snapshots."""
    db = tmp_path / "test.db"
    c = sqlite3.connect(str(db))
    c.executescript("""
        CREATE TABLE fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, strategy TEXT, condition_id TEXT, token_id TEXT,
            side TEXT, price REAL, size REAL, notional REAL, reason TEXT
        );
        CREATE TABLE fills_shadow_queue (
            fill_id INTEGER PRIMARY KEY,
            top_of_book_price REAL, walked_vwap_price REAL,
            pessimistic_price REAL, size REAL, levels_walked INTEGER,
            partial INTEGER, slippage_bps_walked REAL, slippage_bps_pess REAL,
            is_maker INTEGER, taker_fee_paid REAL, maker_rebate_credited REAL,
            cancel_latency_penalty REAL, effective_fill_price REAL,
            rebate_density_adjusted REAL DEFAULT 0
        );
        CREATE TABLE book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id TEXT, ts REAL, trigger TEXT, mid REAL,
            best_bid REAL, best_ask REAL, spread REAL,
            n_bid_levels INTEGER, n_ask_levels INTEGER,
            bid_total_size REAL, ask_total_size REAL,
            last_update_ts REAL, book_blob BLOB
        );
    """)
    c.commit()
    return c, str(db)


def test_replay_empty_strategy_returns_empty():
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        c, db_path = _setup_db(Path(td))
        c.close()
        out = replay_fills_for_strategy(db_path, "nothing")
        assert out == []


def test_replay_taker_buy_with_walked_vwap():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        c, db_path = _setup_db(Path(td))
        # 1 taker BUY at 0.50; walked VWAP was 0.51 (paid more)
        c.execute(
            """INSERT INTO fills(ts,strategy,condition_id,token_id,side,price,size,notional,reason)
               VALUES(1000, 't', '0xc', 'tok', 'BUY', 0.50, 100, 50.0, '')"""
        )
        c.execute(
            """INSERT INTO fills_shadow_queue
               (fill_id, top_of_book_price, walked_vwap_price, pessimistic_price,
                size, levels_walked, partial, slippage_bps_walked, slippage_bps_pess,
                is_maker, taker_fee_paid, maker_rebate_credited,
                cancel_latency_penalty, effective_fill_price)
               VALUES (1, 0.50, 0.51, 0.515, 100, 2, 0, 200, 300, 0, 0.375, 0, 0, 0.51)"""
        )
        c.commit()
        c.close()
        fills = replay_fills_for_strategy(db_path, "t")
        assert len(fills) == 1
        f = fills[0]
        assert f.recorded_price == 0.50
        assert f.walked_vwap == 0.51
        assert abs(f.pnl_delta_per_share - 0.01) < 1e-9  # 0.51 - 0.50 paid extra
        assert f.is_maker is False


def test_aggregate_haircut_taker_buy_pays_more():
    f = FillReplay(
        fill_id=1, ts=1000, strategy="t", side="BUY", token_id="tok",
        recorded_price=0.50, recorded_size=100,
        walked_vwap=0.51, pessimistic_price=0.515,
        cancel_latency_penalty=0.0, effective_fill_price=0.51,
        is_maker=False,
        notional_recorded=50.0, notional_queue_aware=51.0,
        pnl_delta_per_share=0.01, snapshot_age_sec=5.0,
    )
    h = aggregate_haircut([f])
    assert h["n_fills"] == 1
    assert h["n_buys"] == 1
    # BUY paid 0.01 more per share × 100 shares = -$1 P&L haircut
    assert abs(h["pnl_haircut_from_queue_aware"] + 1.0) < 1e-9


def test_aggregate_haircut_maker_uses_effective_price():
    """For a maker fill the effective_fill_price reflects cancel-latency."""
    f = FillReplay(
        fill_id=2, ts=2000, strategy="m", side="BUY", token_id="tok",
        recorded_price=0.40, recorded_size=50,
        walked_vwap=None,  # makers don't walk
        pessimistic_price=0.405,
        cancel_latency_penalty=0.005 * 50,  # $0.25
        effective_fill_price=0.405,         # paid 0.005 more (drift)
        is_maker=True,
        notional_recorded=20.0, notional_queue_aware=20.25,
        pnl_delta_per_share=0.005,
        snapshot_age_sec=15.0,
    )
    h = aggregate_haircut([f])
    assert h["n_maker_fills"] == 1
    # BUY paid 0.005 more per share × 50 shares = -$0.25 P&L haircut
    assert abs(h["pnl_haircut_from_queue_aware"] + 0.25) < 1e-9


def test_aggregate_haircut_sell_receives_less():
    f = FillReplay(
        fill_id=3, ts=3000, strategy="t", side="SELL", token_id="tok",
        recorded_price=0.60, recorded_size=100,
        walked_vwap=0.59, pessimistic_price=0.585,
        cancel_latency_penalty=0.0, effective_fill_price=0.59,
        is_maker=False,
        notional_recorded=60.0, notional_queue_aware=59.0,
        pnl_delta_per_share=-0.01, snapshot_age_sec=2.0,
    )
    h = aggregate_haircut([f])
    # SELL received 0.01 less per share × 100 shares = -$1 haircut
    assert abs(h["pnl_haircut_from_queue_aware"] + 1.0) < 1e-9
