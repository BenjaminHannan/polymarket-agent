"""Tests for the maker-thesis certification pipeline."""
from __future__ import annotations

import sqlite3
import time

from scripts.certify_maker_thesis import (
    RoundTrip, cpcv_market_folds, per_fold_edge, sharpe_per_market_pnl,
    sign_test_p, _reconstruct_round_trips_from_fills, persist_cert,
    ensure_cert_table,
)


def _make_rt(cond, tok, sz, op_px, cl_px, pnl):
    return RoundTrip(
        condition_id=cond, token_id=tok, open_ts=1.0, close_ts=2.0,
        size=sz, open_price=op_px, close_price=cl_px,
        open_fees=0.0, close_fees=0.0, net_pnl=pnl,
    )


def test_cpcv_market_folds_no_market_leak():
    """All round-trips with the same condition_id must end up in the same fold."""
    rts = [
        _make_rt("c1", "t1", 10, 0.5, 0.55, 0.5),
        _make_rt("c1", "t1", 10, 0.5, 0.55, 0.5),    # same market
        _make_rt("c2", "t2", 10, 0.5, 0.55, 0.5),
        _make_rt("c3", "t3", 10, 0.5, 0.55, 0.5),
        _make_rt("c4", "t4", 10, 0.5, 0.55, 0.5),
        _make_rt("c5", "t5", 10, 0.5, 0.55, 0.5),
        _make_rt("c6", "t6", 10, 0.5, 0.55, 0.5),
        _make_rt("c7", "t7", 10, 0.5, 0.55, 0.5),
        _make_rt("c8", "t8", 10, 0.5, 0.55, 0.5),
    ]
    folds = cpcv_market_folds(rts, n_folds=4)
    # Inspect: indices 0 and 1 are same market — must be in same fold across all 4 splits.
    for train_idx, test_idx in folds:
        rt0_in_test = 0 in test_idx
        rt1_in_test = 1 in test_idx
        assert rt0_in_test == rt1_in_test, "market-id leak across folds"


def test_per_fold_edge_basic():
    rts = [
        _make_rt("c1", "t1", 100, 0.50, 0.55, 5.0),
        _make_rt("c2", "t2", 100, 0.50, 0.45, -5.0),
    ]
    edge_pos = per_fold_edge(rts, [0])
    edge_neg = per_fold_edge(rts, [1])
    assert edge_pos > 0
    assert edge_neg < 0


def test_per_fold_edge_empty():
    rts = [_make_rt("c1", "t1", 10, 0.5, 0.5, 0.0)]
    assert per_fold_edge(rts, []) == 0.0


def test_sharpe_per_market_pnl_zero_when_constant():
    rts = [
        _make_rt(f"c{i}", "t1", 10, 0.5, 0.5, 1.0)
        for i in range(20)
    ]
    s = sharpe_per_market_pnl(rts)
    # All P&L equal ⇒ std=0 ⇒ Sharpe=0
    assert s == 0.0


def test_sharpe_per_market_pnl_positive():
    """20 markets all with +1 P&L is the constant case → 0 Sharpe;
    20 markets with random positive P&L should be > 0."""
    import random as _r
    rng = _r.Random(0)
    rts = [
        _make_rt(f"c{i}", "t1", 10, 0.5, 0.5,
                 rng.uniform(0.5, 1.5))
        for i in range(30)
    ]
    s = sharpe_per_market_pnl(rts)
    assert s > 0


def test_sign_test_all_positive():
    """All 8 folds positive → very small p-value."""
    p = sign_test_p([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert p < 0.01


def test_sign_test_balanced():
    p = sign_test_p([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4])
    # 4 positive of 8 ⇒ P(X >= 4 | Binom(8, 0.5)) = 0.637
    assert 0.5 < p < 0.8


def test_reconstruct_returns_empty_on_no_fills(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        """CREATE TABLE fills (
            ts REAL, strategy TEXT, condition_id TEXT, token_id TEXT,
            side TEXT, size REAL, price REAL,
            taker_fee_paid REAL, maker_rebate_credited REAL
        )"""
    )
    rts = _reconstruct_round_trips_from_fills(conn, "anything")
    assert rts == []


def test_reconstruct_basic_fifo(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        """CREATE TABLE fills (
            ts REAL, strategy TEXT, condition_id TEXT, token_id TEXT,
            side TEXT, size REAL, price REAL,
            taker_fee_paid REAL, maker_rebate_credited REAL
        )"""
    )
    conn.execute(
        "INSERT INTO fills VALUES (1.0, 'pp', 'c1', 't1', 'BUY', 100, 0.40, 0, 0)"
    )
    conn.execute(
        "INSERT INTO fills VALUES (2.0, 'pp', 'c1', 't1', 'SELL', 100, 0.50, 0, 0)"
    )
    conn.commit()
    rts = _reconstruct_round_trips_from_fills(conn, "pp")
    assert len(rts) == 1
    rt = rts[0]
    assert rt.size == 100
    assert rt.open_price == 0.40
    assert rt.close_price == 0.50
    # PnL = (0.50 − 0.40) × 100 = 10.00
    assert abs(rt.net_pnl - 10.0) < 1e-6


def test_persist_cert_round_trip(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_cert_table(conn)
    persist_cert(
        conn, "test_cert_name",
        enabled=True, dsr=0.99, n=300,
        detail={"foo": "bar"}, category="sports_global",
    )
    row = conn.execute(
        "SELECT enabled, dsr_holdout, n_holdout, detail FROM strategy_certificates WHERE name=?",
        ("test_cert_name",),
    ).fetchone()
    assert row[0] == 1
    assert abs(row[1] - 0.99) < 1e-9
    assert row[2] == 300
    assert '"foo"' in row[3]
