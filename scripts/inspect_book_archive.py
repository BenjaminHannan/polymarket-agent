"""Inspect the L2 book_snapshots archive.

Usage:
  python -m scripts.inspect_book_archive                 # summary stats
  python -m scripts.inspect_book_archive --token <id>    # last 10 snapshots for a token
  python -m scripts.inspect_book_archive --token <id> --replay <ts>
                                                          # rebuild book at ts
"""
from __future__ import annotations

import argparse
import json
import sqlite3

from polyagent.config import settings
from polyagent.risk.book_archive import (
    archive_db_path, archive_stats, decode_book, replay_book_at,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=None)
    p.add_argument("--replay", type=float, default=None,
                   help="UTC unix timestamp to replay book at; requires --token")
    p.add_argument("--last", type=int, default=10)
    p.add_argument("--db", default=None,
                   help="path to book_archive db (default: from BOOK_ARCHIVE_DB_PATH or sibling of paper.db)")
    args = p.parse_args()

    db_path = args.db or archive_db_path(settings.db_path)
    conn = sqlite3.connect(db_path)

    if args.token and args.replay:
        book = replay_book_at(conn, args.token, args.replay)
        if book is None:
            print(f"no snapshot at or before ts={args.replay} for token={args.token}")
            return
        print(f"snapshot_ts={book['snapshot_ts']:.1f}")
        print(f"  best_bid={book['bids'][0][0] if book['bids'] else None}")
        print(f"  best_ask={book['asks'][0][0] if book['asks'] else None}")
        print(f"  bid_levels={len(book['bids'])}  ask_levels={len(book['asks'])}")
        if book["bids"]:
            print("  top 3 bids:", book["bids"][:3])
        if book["asks"]:
            print("  top 3 asks:", book["asks"][:3])
        return

    if args.token:
        rows = conn.execute(
            """SELECT ts, trigger, mid, best_bid, best_ask, n_bid_levels,
                      n_ask_levels, LENGTH(book_blob) AS blob_size
               FROM book_snapshots
               WHERE token_id = ?
               ORDER BY ts DESC LIMIT ?""",
            (args.token, args.last),
        ).fetchall()
        if not rows:
            print(f"no snapshots for token {args.token}")
            return
        print(f"last {len(rows)} snapshots for {args.token}:")
        print(f"  {'ts':>12s}  {'trigger':10s}  {'mid':>7s}  {'spread':>7s}  {'levels':>10s}  {'bytes':>6s}")
        for r in rows:
            spread = (r[4] - r[3]) if (r[4] is not None and r[3] is not None) else None
            print(
                f"  {r[0]:>12.1f}  {r[1]:10s}  {r[2] or 0:>7.4f}  "
                f"{spread or 0:>7.4f}  {r[5]}/{r[6]:>4}  {r[7]:>6}"
            )
        return

    s = archive_stats(conn)
    print("book_archive summary:")
    print(json.dumps(s, indent=2))
    if s["total_snapshots"]:
        kb = s["blob_bytes_total"] / 1024
        avg = kb / s["total_snapshots"]
        print(f"  storage: {kb:.0f} KB total, {avg:.2f} KB avg per snapshot")


if __name__ == "__main__":
    main()
