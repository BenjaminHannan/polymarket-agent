"""Tests for the L2 book-snapshot archive."""
from __future__ import annotations

import sqlite3
import time

from polyagent.orderbook import OrderBook
from polyagent.risk.book_archive import (
    archive_stats,
    decode_book,
    encode_book,
    ensure_table,
    replay_book_at,
    snapshot,
)


def _book(bids: dict[float, float], asks: dict[float, float]) -> OrderBook:
    b = OrderBook(token_id="t")
    b.bids = {float(p): float(s) for p, s in bids.items()}
    b.asks = {float(p): float(s) for p, s in asks.items()}
    b.last_update_ts = time.time()
    return b


def test_encode_decode_roundtrip():
    b = _book({0.49: 100, 0.48: 200}, {0.51: 150, 0.52: 80})
    blob, summary = encode_book(b)
    assert summary["best_bid"] == 0.49
    assert summary["best_ask"] == 0.51
    assert summary["mid"] == 0.50
    assert summary["spread"] == pytest_approx(0.02)
    assert summary["n_bid_levels"] == 2
    assert summary["n_ask_levels"] == 2
    decoded = decode_book(blob)
    # JSON round-trip yields lists not tuples
    assert decoded["bids"] == [[0.49, 100.0], [0.48, 200.0]]
    assert decoded["asks"] == [[0.51, 150.0], [0.52, 80.0]]


def test_encode_compression_actually_shrinks_payload():
    # 50 levels each side; check zlib compression ratio
    bids = {0.50 - 0.001 * i: 100.0 + i for i in range(50)}
    asks = {0.51 + 0.001 * i: 100.0 + i for i in range(50)}
    b = _book(bids, asks)
    blob, _ = encode_book(b)
    import json
    raw_size = len(json.dumps({"bids": list(bids.items()), "asks": list(asks.items())}).encode())
    assert len(blob) < raw_size  # zlib actually shrunk it
    # Tight books on prediction markets should compress to ~30% or better
    assert len(blob) < raw_size * 0.5


def test_snapshot_persists_and_replay_returns_nearest():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    b1 = _book({0.49: 100}, {0.51: 100})
    snapshot(conn, "tok1", b1, trigger="periodic", ts=1000.0)
    b2 = _book({0.50: 200}, {0.52: 200})
    snapshot(conn, "tok1", b2, trigger="fill", ts=2000.0)
    # Replay at ts=1500 → should return the ts=1000 snapshot
    out = replay_book_at(conn, "tok1", 1500.0)
    assert out is not None
    assert out["snapshot_ts"] == 1000.0
    assert out["bids"] == [[0.49, 100.0]]
    # Replay at ts=2500 → should return the ts=2000 snapshot
    out2 = replay_book_at(conn, "tok1", 2500.0)
    assert out2 is not None
    assert out2["snapshot_ts"] == 2000.0


def test_replay_returns_none_when_no_history():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    assert replay_book_at(conn, "missing", 1000.0) is None


def test_replay_returns_none_when_ts_before_first_snapshot():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    b = _book({0.49: 100}, {0.51: 100})
    snapshot(conn, "tok1", b, ts=2000.0)
    # Asking for the book at ts=1000 — earlier than first snapshot
    assert replay_book_at(conn, "tok1", 1000.0) is None


def test_archive_stats_aggregates_correctly():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    b1 = _book({0.49: 100}, {0.51: 100})
    b2 = _book({0.50: 200}, {0.52: 200})
    snapshot(conn, "tok_a", b1, trigger="periodic", ts=1000.0)
    snapshot(conn, "tok_a", b2, trigger="fill", ts=2000.0)
    snapshot(conn, "tok_b", b1, trigger="periodic", ts=1500.0)
    s = archive_stats(conn)
    assert s["total_snapshots"] == 3
    assert s["distinct_tokens"] == 2
    assert s["earliest_ts"] == 1000.0
    assert s["latest_ts"] == 2000.0
    assert s["by_trigger"] == {"periodic": 2, "fill": 1}
    assert s["blob_bytes_total"] > 0


def test_snapshot_skipped_on_empty_book():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    empty = OrderBook(token_id="t")  # no bids / asks
    out = snapshot(conn, "tok1", empty, trigger="periodic", ts=1000.0)
    assert out is None  # no-op
    assert conn.execute("SELECT COUNT(*) FROM book_snapshots").fetchone()[0] == 0


# Helper for approximate floating-point comparison without depending on
# pytest.approx in case of any import quirks
def pytest_approx(value: float, tol: float = 1e-9):
    class _A:
        def __eq__(self, other): return abs(other - value) < tol
        def __repr__(self): return f"approx({value})"
    return _A()
