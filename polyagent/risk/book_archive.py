"""L2 order-book snapshot archive — self-recorded historical book state.

Path 1 from the queue-aware-fill conversation: Polymarket doesn't expose
historical L2 via public API, but the WSS stream gives current book
state in real time. The bot already reconstructs books in BookStore;
this module persists snapshots so downstream analysis (queue-position
backtests, cert re-validation under realistic fills) has a real
historical L2 archive to work from.

Two snapshot triggers:

  1. **fill** — every time PaperBroker actually fills, snapshot the
     book at that exact moment. THIS is the data that matters most
     for cert validation: "would my quote have filled given the queue
     ahead at the post-time?"
  2. **periodic** — every snapshot_interval_sec (default 300) we
     snapshot every certified-category token's current book. Provides
     baseline coverage between fills so we can reconstruct intra-fill
     book state and run AS calibration backtests.

Storage: SQLite `book_snapshots` table with zlib-compressed JSON of
the bids and asks dicts. Typical compression ratio is ~5–8× on
sparse books (most price levels empty); ~70 sports_global tokens at
5-min cadence = 1,008 rows/day × ~500 B avg = ~500 KB/day. Plus
fill-triggered snapshots (sparse) ~10 KB/day. Total: <1 MB/day for
the certified slice.

Default OFF behind ENABLE_BOOK_ARCHIVE=1.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import zlib
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# ── Schema ──────────────────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS book_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id        TEXT NOT NULL,
            ts              REAL NOT NULL,
            trigger         TEXT NOT NULL,    -- "fill" | "periodic" | "stale" | "manual"
            mid             REAL,
            best_bid        REAL,
            best_ask        REAL,
            spread          REAL,
            n_bid_levels    INTEGER,
            n_ask_levels    INTEGER,
            bid_total_size  REAL,
            ask_total_size  REAL,
            last_update_ts  REAL,
            book_blob       BLOB             -- zlib-compressed JSON
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS book_snap_token_ts ON book_snapshots(token_id, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS book_snap_trigger_ts ON book_snapshots(trigger, ts)")
    conn.commit()


# ── Encode / decode ─────────────────────────────────────────────────────
def encode_book(book) -> tuple[bytes, dict]:
    """Compress book bids+asks to zlib-blob; return (blob, summary).

    Summary is a small dict of denormalized fields that go into indexable
    columns so we can do range queries without uncompressing every row.
    """
    bids = sorted([(float(p), float(s)) for p, s in book.bids.items()], reverse=True)
    asks = sorted([(float(p), float(s)) for p, s in book.asks.items()])
    payload = {"bids": bids, "asks": asks, "last_update_ts": getattr(book, "last_update_ts", None)}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    blob = zlib.compress(raw, level=6)
    bb = bids[0] if bids else None
    ba = asks[0] if asks else None
    summary = {
        "best_bid": bb[0] if bb else None,
        "best_ask": ba[0] if ba else None,
        "mid": (bb[0] + ba[0]) / 2.0 if (bb and ba) else None,
        "spread": (ba[0] - bb[0]) if (bb and ba) else None,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
        "bid_total_size": sum(s for _, s in bids),
        "ask_total_size": sum(s for _, s in asks),
        "last_update_ts": getattr(book, "last_update_ts", None),
    }
    return blob, summary


def decode_book(blob: bytes) -> dict:
    """Reconstruct a {bids: [(price, size), ...], asks: [...], last_update_ts}
    dict from the compressed blob."""
    raw = zlib.decompress(blob)
    return json.loads(raw.decode("utf-8"))


# ── Persistence ─────────────────────────────────────────────────────────
def snapshot(
    conn: sqlite3.Connection,
    token_id: str,
    book,
    *,
    trigger: str = "periodic",
    ts: float | None = None,
) -> int | None:
    """Persist one book snapshot. Returns the row id, or None on no-op
    (empty book, etc.)."""
    if not getattr(book, "bids", None) and not getattr(book, "asks", None):
        return None
    ensure_table(conn)
    blob, summary = encode_book(book)
    cur = conn.execute(
        """INSERT INTO book_snapshots
           (token_id, ts, trigger, mid, best_bid, best_ask, spread,
            n_bid_levels, n_ask_levels, bid_total_size, ask_total_size,
            last_update_ts, book_blob)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            token_id, float(ts if ts is not None else time.time()), trigger,
            summary["mid"], summary["best_bid"], summary["best_ask"],
            summary["spread"], summary["n_bid_levels"], summary["n_ask_levels"],
            summary["bid_total_size"], summary["ask_total_size"],
            summary["last_update_ts"], blob,
        ),
    )
    conn.commit()
    return cur.lastrowid


def replay_book_at(conn: sqlite3.Connection, token_id: str, ts: float) -> dict | None:
    """Return the most recent snapshot at or before `ts` for `token_id`,
    decoded into a {bids, asks, last_update_ts} dict. Returns None if
    no snapshot exists."""
    row = conn.execute(
        """SELECT ts, book_blob FROM book_snapshots
           WHERE token_id = ? AND ts <= ?
           ORDER BY ts DESC LIMIT 1""",
        (token_id, float(ts)),
    ).fetchone()
    if row is None:
        return None
    decoded = decode_book(row[1])
    decoded["snapshot_ts"] = float(row[0])
    return decoded


def archive_stats(conn: sqlite3.Connection) -> dict:
    """Aggregate stats for dashboard + observability."""
    ensure_table(conn)
    total = conn.execute("SELECT COUNT(*) FROM book_snapshots").fetchone()[0]
    by_trigger = {
        r[0]: r[1] for r in conn.execute(
            "SELECT trigger, COUNT(*) FROM book_snapshots GROUP BY trigger"
        )
    }
    range_row = conn.execute(
        "SELECT MIN(ts), MAX(ts), COUNT(DISTINCT token_id) FROM book_snapshots"
    ).fetchone()
    blob_size = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(book_blob)), 0) FROM book_snapshots"
    ).fetchone()[0]
    return {
        "total_snapshots": int(total),
        "by_trigger": by_trigger,
        "earliest_ts": float(range_row[0]) if range_row[0] is not None else None,
        "latest_ts": float(range_row[1]) if range_row[1] is not None else None,
        "distinct_tokens": int(range_row[2] or 0),
        "blob_bytes_total": int(blob_size),
    }


# ── Periodic snapshot loop ──────────────────────────────────────────────
async def periodic_snapshot_loop(
    book_store,
    target_tokens: list[str],
    db_path: str,
    *,
    interval_sec: float = 300.0,
) -> None:
    """Background task: every interval_sec, snapshot every target token's
    current book to the archive with trigger='periodic'.
    """
    import asyncio
    log.info(
        "book_archive_start",
        n_tokens=len(target_tokens),
        interval_sec=interval_sec,
    )
    while True:
        now = time.time()
        n = 0
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                ensure_table(conn)
                for tok in target_tokens:
                    book = book_store.books.get(tok)
                    if book is None:
                        continue
                    if snapshot(conn, tok, book, trigger="periodic", ts=now):
                        n += 1
            finally:
                conn.close()
            if n:
                log.info("book_archive_periodic", n=n)
        except Exception as e:
            log.warning("book_archive_loop_error", err=str(e))
        await asyncio.sleep(interval_sec)
