"""Per-strategy realized P&L attribution.

Joins `fills` to `resolutions` on condition_id; for each fill, determines
whether the bought token was the winner and computes payout.

P&L per fill:
    payout = 1.0 if (token_id == yes_token AND yes_won)
                 OR (token_id == no_token AND NOT yes_won)
             else 0.0
    pnl = (payout - price) * size

Aggregates per strategy and per day. Open (unresolved) positions don't
contribute — by design, throttling is based on realized outcomes only.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import structlog

log = structlog.get_logger()


@dataclass
class StrategyAttribution:
    strategy: str
    n_trades_resolved: int = 0
    n_wins: int = 0
    realized_pnl: float = 0.0
    total_notional: float = 0.0
    daily_pnl: dict[str, float] = field(default_factory=dict)  # "YYYY-MM-DD" -> pnl

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades_resolved if self.n_trades_resolved else 0.0

    @property
    def pnl_pct_of_notional(self) -> float:
        return self.realized_pnl / self.total_notional if self.total_notional else 0.0


def attribute_pnl(db_path: str, since_ts: float | None = None) -> dict[str, StrategyAttribution]:
    """Compute per-strategy realized P&L. Filters to fills whose markets resolved
    after `since_ts` (use now - 30d for the 30-day window)."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")

    where = ""
    params: tuple = ()
    if since_ts is not None:
        where = "WHERE r.resolved_ts >= ?"
        params = (since_ts,)

    sql = f"""
        SELECT
            f.strategy, f.ts, f.condition_id, f.token_id, f.side,
            f.price, f.size, f.notional,
            r.yes_won, r.yes_token_id, r.no_token_id
        FROM fills f
        INNER JOIN resolutions r ON f.condition_id = r.condition_id
        {where}
    """
    rows = list(conn.execute(sql, params))
    conn.close()

    out: dict[str, StrategyAttribution] = defaultdict(lambda: StrategyAttribution(strategy=""))
    for r in rows:
        strat = r["strategy"]
        att = out[strat]
        att.strategy = strat

        token_id = r["token_id"]
        yes_token = r["yes_token_id"]
        no_token = r["no_token_id"]
        yes_won = bool(r["yes_won"])

        if token_id == yes_token:
            won = yes_won
        elif token_id == no_token:
            won = not yes_won
        else:
            # Token not recognized as either leg; skip (shouldn't happen).
            continue

        payout = 1.0 if won else 0.0
        pnl = (payout - float(r["price"])) * float(r["size"])

        att.n_trades_resolved += 1
        if won:
            att.n_wins += 1
        att.realized_pnl += pnl
        att.total_notional += float(r["notional"])

        day = time.strftime("%Y-%m-%d", time.gmtime(r["ts"]))
        att.daily_pnl[day] = att.daily_pnl.get(day, 0.0) + pnl

    return dict(out)
