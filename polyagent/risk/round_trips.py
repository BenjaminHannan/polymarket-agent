"""Round-trip P&L attribution.

For per-strategy realized P&L *as fills happen* (not waiting for market
resolution), pair every closing fill against an earlier opening fill
on the same (strategy, token_id) using FIFO matching. The result is a
ledger of completed round-trips with:

  open_price, close_price, captured_spread, fees, rebate, realized_pnl

This is the missing piece for measuring whether `passive_poster_v2` is
actually winning round-trips on the maker side, vs. accumulating
one-sided inventory that just happens to settle in our favor at
resolution. Per pmwhy.md: "is the maker actually winning round-trips,
or just accumulating one-sided inventory and getting saved by
mid-drift?"

FIFO matching mirrors the standard accounting convention. A token that
gets bought 30 then bought 20 then sold 40 produces:
  closed: 30 shares matched to the first BUY,
          10 shares matched to the second BUY (partial)
  open:   10 shares of the second BUY remaining

Generic across strategies — wire from broker.submit() so any strategy
that opens and closes positions populates the ledger.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS round_trip_legs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy      TEXT NOT NULL,
            token_id      TEXT NOT NULL,
            condition_id  TEXT NOT NULL,
            open_fill_id  INTEGER NOT NULL,    -- fills.id of the opening BUY
            close_fill_id INTEGER,             -- fills.id of the closing SELL (NULL while open)
            open_ts       REAL NOT NULL,
            close_ts      REAL,
            size          REAL NOT NULL,
            open_price    REAL NOT NULL,
            close_price   REAL,
            open_fees     REAL DEFAULT 0,      -- fee paid on open (taker) or rebate received (maker, signed)
            close_fees    REAL DEFAULT 0,
            gross_pnl     REAL,                -- (close_price - open_price) × size
            net_pnl       REAL                 -- gross_pnl - open_fees + open_rebate - close_fees + close_rebate
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS rt_legs_token_strat_open ON round_trip_legs(token_id, strategy, close_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS rt_legs_strategy ON round_trip_legs(strategy, close_ts)")
    conn.commit()


@dataclass
class FillContext:
    fill_id: int
    strategy: str
    condition_id: str
    token_id: str
    side: str           # "BUY" opens / adds, "SELL" closes
    price: float
    size: float
    ts: float
    fees_paid: float = 0.0      # taker fee paid (positive number)
    rebate_credited: float = 0.0  # maker rebate received (positive number)


def _signed_cost(fc: FillContext) -> float:
    """Net cash flow per share at this fill: BUY pays out, SELL takes in,
    fee debits, rebate credits.
    Returns the *fees applied to one share* — used in net_pnl
    accounting on each leg.
    """
    fee_per_share = (fc.fees_paid - fc.rebate_credited) / max(fc.size, 1e-9)
    return fee_per_share


def record_fill(conn: sqlite3.Connection, fc: FillContext) -> dict:
    """Record a fill and update the round-trip ledger via FIFO matching.

    On BUY (open or add):
      - Insert a new round_trip_legs row with close_* NULL (open lot).

    On SELL (close):
      - Find oldest open lots (close_ts IS NULL) for this
        (strategy, token_id), match `size` against them in FIFO order,
        update each matched leg's close_* fields, compute net_pnl.
      - If SELL size exceeds total open lots (we're going short), the
        excess is recorded as a SELL-opening lot — Polymarket allows
        going short on the YES token because the NO token is a
        separate ledger. (For paper-mode this matches broker semantics.)

    Returns a summary dict: {"opened": n, "closed_legs": [...], "remaining_to_close": x}
    """
    ensure_table(conn)
    summary: dict = {"opened": 0, "closed_legs": [], "remaining_to_close": 0.0}

    if fc.side == "BUY":
        # Open lot
        fee_signed = fc.fees_paid - fc.rebate_credited
        conn.execute(
            """INSERT INTO round_trip_legs
               (strategy, token_id, condition_id, open_fill_id,
                open_ts, size, open_price, open_fees)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fc.strategy, fc.token_id, fc.condition_id, fc.fill_id,
             fc.ts, fc.size, fc.price, fee_signed),
        )
        conn.commit()
        summary["opened"] = 1
        return summary

    # SELL: match against oldest open lots
    rows = conn.execute(
        """SELECT id, size, open_price, open_fees
           FROM round_trip_legs
           WHERE strategy = ? AND token_id = ? AND close_ts IS NULL
           ORDER BY open_ts ASC""",
        (fc.strategy, fc.token_id),
    ).fetchall()

    remaining = fc.size
    fee_per_share_close = (fc.fees_paid - fc.rebate_credited) / max(fc.size, 1e-9)
    for leg_id, leg_size, open_price, open_fees in rows:
        if remaining <= 0:
            break
        match = min(remaining, leg_size)
        fraction = match / leg_size if leg_size > 0 else 1.0
        leg_open_fee_share = (open_fees or 0.0) * fraction
        leg_close_fee = fee_per_share_close * match
        gross = (fc.price - open_price) * match
        net = gross - leg_open_fee_share - leg_close_fee
        if match >= leg_size - 1e-9:
            # Full close of this leg
            conn.execute(
                """UPDATE round_trip_legs
                   SET close_fill_id = ?, close_ts = ?, close_price = ?,
                       close_fees = ?, gross_pnl = ?, net_pnl = ?
                   WHERE id = ?""",
                (fc.fill_id, fc.ts, fc.price, leg_close_fee, gross, net, leg_id),
            )
            summary["closed_legs"].append({
                "id": leg_id, "size": match, "gross_pnl": gross, "net_pnl": net,
            })
        else:
            # Partial close: split the leg into a closed portion + remaining open portion
            remaining_size = leg_size - match
            remaining_open_fee = (open_fees or 0.0) * (remaining_size / leg_size)
            # Update original to reflect the closed portion
            conn.execute(
                """UPDATE round_trip_legs
                   SET size = ?, open_fees = ?, close_fill_id = ?, close_ts = ?,
                       close_price = ?, close_fees = ?, gross_pnl = ?, net_pnl = ?
                   WHERE id = ?""",
                (match, leg_open_fee_share, fc.fill_id, fc.ts, fc.price,
                 leg_close_fee, gross, net, leg_id),
            )
            # Insert a new "still open" leg with the remaining portion,
            # preserving the open_fill_id pointer so it stays attributable.
            conn.execute(
                """INSERT INTO round_trip_legs
                   (strategy, token_id, condition_id, open_fill_id,
                    open_ts, size, open_price, open_fees)
                   SELECT strategy, token_id, condition_id, open_fill_id,
                          open_ts, ?, open_price, ?
                   FROM round_trip_legs WHERE id = ?""",
                (remaining_size, remaining_open_fee, leg_id),
            )
            summary["closed_legs"].append({
                "id": leg_id, "size": match, "gross_pnl": gross, "net_pnl": net,
            })
        remaining -= match

    if remaining > 1e-9:
        # We sold more than we held — remaining becomes a SHORT open lot.
        # Mark by storing a NEGATIVE size (so a later BUY will close it
        # with the same FIFO logic in reverse). Simpler: just record the
        # leftover as an "uncovered SELL" leg with size>0 but using the
        # close_ price as the "open_ price" of a short, and let a future
        # BUY do the matching. We add minimal support here — tracked but
        # not paired until a buy-back happens.
        fee_signed = (fc.fees_paid - fc.rebate_credited) * (remaining / fc.size)
        conn.execute(
            """INSERT INTO round_trip_legs
               (strategy, token_id, condition_id, open_fill_id,
                open_ts, size, open_price, open_fees)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fc.strategy, fc.token_id, fc.condition_id, fc.fill_id,
             fc.ts, -remaining, fc.price, fee_signed),  # negative size = short
        )
        summary["remaining_to_close"] = remaining

    conn.commit()
    return summary


def strategy_summary(conn: sqlite3.Connection, strategy: str) -> dict:
    """Aggregate ledger view for the dashboard."""
    closed = conn.execute(
        """SELECT
              COUNT(*),
              COALESCE(SUM(gross_pnl), 0),
              COALESCE(SUM(net_pnl), 0),
              COALESCE(SUM(close_fees + open_fees), 0),
              COALESCE(AVG(close_price - open_price), 0),
              COALESCE(SUM(size), 0)
           FROM round_trip_legs
           WHERE strategy = ? AND close_ts IS NOT NULL""",
        (strategy,),
    ).fetchone()
    open_legs = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(AVG(open_price), 0)
           FROM round_trip_legs
           WHERE strategy = ? AND close_ts IS NULL""",
        (strategy,),
    ).fetchone()
    return {
        "strategy": strategy,
        "closed_round_trips": int(closed[0] or 0),
        "gross_pnl_realized": round(float(closed[1]), 4),
        "net_pnl_realized": round(float(closed[2]), 4),
        "fees_total": round(float(closed[3]), 4),
        "avg_captured_spread": round(float(closed[4]), 4),
        "total_size_round_tripped": round(float(closed[5]), 2),
        "open_legs": int(open_legs[0] or 0),
        "open_size_total": round(float(open_legs[1]), 2),
        "open_avg_price": round(float(open_legs[2]), 4),
    }
