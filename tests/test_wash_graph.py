"""Tests for Sirolly wash-graph clustering."""
from __future__ import annotations

import sqlite3

from polyagent.risk.wash_graph import (
    compute_wallet_signatures,
    compute_market_wash_scores,
    market_wash_share,
    suppression_factor,
    is_high_wash,
    ensure_tables,
)


def _seed_trades(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE trades (
            tx_hash TEXT, wallet TEXT, counterparty_wallet TEXT,
            asset TEXT, side TEXT, size REAL, price REAL, timestamp REAL
        );
        """
    )


def test_no_counterparty_column_returns_empty(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    # Trades table without counterparty_wallet column.
    conn.executescript(
        """
        CREATE TABLE trades (tx_hash TEXT, wallet TEXT, asset TEXT,
                             side TEXT, size REAL);
        INSERT INTO trades VALUES ('a', 'w1', 'tok1', 'BUY', 100);
        """
    )
    sigs = compute_wallet_signatures(conn)
    assert sigs == []


def test_wash_pair_flagged(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    # w1 and w2 cycle exclusively: 100 trades each, all between each other.
    for i in range(60):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w1', 'w2', 'tok1', 'BUY', 100, 0.5, ?)",
            (f"tx_buy_{i}", i),
        )
    for i in range(60):
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w1', 'w2', 'tok1', 'SELL', 100, 0.5, ?)",
            (f"tx_sell_{i}", 100 + i),
        )
    conn.commit()
    sigs = compute_wallet_signatures(conn)
    assert len(sigs) >= 1
    w1 = next((s for s in sigs if s.wallet == "w1"), None)
    assert w1 is not None
    assert w1.n_counterparties == 1
    assert w1.top_share == 1.0
    assert w1.is_suspect is True


def test_diverse_wallet_not_flagged(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    _seed_trades(conn)
    # 30 trades against 15 different counterparties.
    for i in range(30):
        cpty = f"w_other_{i % 15}"
        conn.execute(
            "INSERT INTO trades VALUES (?, 'w_diverse', ?, 'tok1', 'BUY', 100, 0.5, ?)",
            (f"tx_{i}", cpty, i),
        )
    conn.commit()
    sigs = compute_wallet_signatures(conn, min_trades=10)
    diverse = next((s for s in sigs if s.wallet == "w_diverse"), None)
    assert diverse is not None
    # 15 unique counterparties > 5 threshold ⇒ not suspect.
    assert diverse.is_suspect is False


def test_market_wash_share_with_no_data(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    assert market_wash_share(conn, "nonexistent") == 0.0
    assert suppression_factor(conn, "nonexistent") == 1.0
    assert is_high_wash(conn, "nonexistent") is False
