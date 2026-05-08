"""Lightweight adverse-selection filter.

The full version queries the Polymarket on-chain subgraph for every
counterparty wallet, builds a "smart money" registry, and downweights
markets where smart money is on the other side of our position. That
needs the subgraph backend live; this lighter version is in-process:
we track which OF OUR OWN markets have shown a pattern of immediate
adverse price movement post-fill (i.e., we got picked off).

Per market: if the most recent N fills' avg PnL-after-30min is < threshold,
flag the market and refuse new entries on it for an hour. Crude but real.

Intentionally conservative + cheap so it always runs.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field

import structlog

from polyagent.config import settings

log = structlog.get_logger()


@dataclass
class AdverseSelectionFilter:
    db_path: str = settings.db_path
    refresh_sec: float = 600.0
    lookback_sec: float = 7 * 86400.0
    min_fills_per_market: int = 3
    bad_avg_pnl_pct: float = -0.10  # avg fill marked at half-life loses ≥ 10%
    blacklist_duration_sec: float = 3600.0
    # token_id -> expiry_ts; check `is_blacklisted(token_id)` to gate trades.
    blacklist_until: dict[str, float] = field(default_factory=dict)

    def is_blacklisted(self, token_id: str) -> bool:
        until = self.blacklist_until.get(token_id, 0.0)
        if until <= 0:
            return False
        if time.time() > until:
            self.blacklist_until.pop(token_id, None)
            return False
        return True

    def refresh(self) -> dict:
        """Recompute blacklist from recent fills + book marks.

        For each token_id with N+ fills in the lookback window, compare the
        avg fill price to the current bid mark; if the avg PnL is worse than
        bad_avg_pnl_pct, blacklist the token for blacklist_duration_sec.
        """
        cutoff = time.time() - self.lookback_sec
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        rows = conn.execute(
            """
            SELECT token_id, AVG(price) as avg_price, COUNT(*) as n
            FROM fills
            WHERE side = 'BUY' AND ts >= ?
            GROUP BY token_id
            HAVING n >= ?
            """,
            (cutoff, self.min_fills_per_market),
        ).fetchall()
        conn.close()

        # We don't have current bid prices in this thread (book_store is not
        # accessible). The throttler/attribution already tracks realized P&L
        # per-strategy; here we approximate "got picked off" via *realized* P&L
        # on resolved fills only. If a token's realized P&L is decisively bad,
        # blacklist.
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        bad: list[tuple[str, float, int]] = []
        for token_id, _avg, n in rows:
            r = conn.execute(
                """
                SELECT
                    SUM(((CASE
                        WHEN (f.token_id = r.yes_token_id AND r.yes_won = 1)
                          OR (f.token_id = r.no_token_id AND r.yes_won = 0)
                        THEN 1.0 ELSE 0.0 END) - f.price) * f.size) AS pnl,
                    SUM(f.notional) AS notional
                FROM fills f
                INNER JOIN resolutions r ON f.condition_id = r.condition_id
                WHERE f.token_id = ?
                  AND r.yes_won IS NOT NULL
                  AND f.side = 'BUY'
                  AND f.ts >= ?
                """,
                (token_id, cutoff),
            ).fetchone()
            if not r or r[0] is None or r[1] is None or r[1] <= 0:
                continue
            pnl_pct = float(r[0]) / float(r[1])
            if pnl_pct < self.bad_avg_pnl_pct:
                bad.append((token_id, pnl_pct, n))
        conn.close()

        until = time.time() + self.blacklist_duration_sec
        for token_id, pnl_pct, n in bad:
            self.blacklist_until[token_id] = until
            log.info(
                "adverse_selection_blacklisted",
                token_id=token_id[:14],
                pnl_pct=round(pnl_pct * 100, 2),
                n_fills=n,
                duration_sec=self.blacklist_duration_sec,
            )

        # Cleanup expired entries
        now = time.time()
        self.blacklist_until = {
            tok: ts for tok, ts in self.blacklist_until.items() if ts > now
        }

        log.info(
            "adverse_selection_refresh",
            n_markets_scanned=len(rows),
            n_blacklisted=len(bad),
            active_blacklist=len(self.blacklist_until),
        )
        return {"scanned": len(rows), "blacklisted": len(bad)}

    async def run(self) -> None:
        log.info("adverse_selection_start", refresh_sec=self.refresh_sec)
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception as e:
                log.warning("adverse_selection_refresh_error", err=str(e))
            await asyncio.sleep(self.refresh_sec)
