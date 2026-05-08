"""Periodic visibility task for arbitrage opportunities.

The yes_no_arb strategy fires continuously on every WSS book update, so true
arbs are taken within milliseconds. This task is purely observability: every
30s it scans all books and reports the arb landscape — how many markets are
in deep-arb vs active-arb vs near-arb vs normal-priced bands, plus the top
opportunities in each band.

This is what makes the "agent constantly scanning" behavior visible in logs.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import structlog

from polyagent.gamma import Market
from polyagent.orderbook import BookStore

log = structlog.get_logger()


# Bucket thresholds on YES_ask + NO_ask
DEEP_ARB = 0.95   # >5pp risk-free edge
ACTIVE_ARB = 0.99 # 1-5pp arb (where yes_no_arb trades)
NEAR_ARB = 1.01   # within 1pp of arb territory


def _question_prefix(q: str) -> str:
    """Strip "by <date>" / "before <date>" / "in <month>" tails to find sibling markets."""
    s = re.sub(r"\s+(by|before|after|in|on)\s+.+$", "", q or "", flags=re.IGNORECASE).strip()
    return s[:80]


@dataclass
class ArbScanner:
    book_store: BookStore
    markets: list[Market]
    poll_sec: float = 30.0

    async def run(self) -> None:
        log.info("arb_scanner_start", n_markets=len(self.markets), poll_sec=self.poll_sec)
        while True:
            await asyncio.sleep(self.poll_sec)
            self._scan_once()

    def _scan_once(self) -> None:
        deep: list[tuple[Market, float, float, float]] = []
        active: list[tuple[Market, float, float, float]] = []
        near: list[tuple[Market, float, float, float]] = []
        normal = 0
        no_quote = 0

        for m in self.markets:
            yes_book = self.book_store.books.get(m.yes_token_id)
            no_book = self.book_store.books.get(m.no_token_id)
            if yes_book is None or no_book is None:
                no_quote += 1
                continue
            ya = yes_book.best_ask()
            na = no_book.best_ask()
            if ya is None or na is None:
                no_quote += 1
                continue
            total = ya[0] + na[0]
            entry = (m, ya[0], na[0], total)
            if total < DEEP_ARB:
                deep.append(entry)
            elif total < ACTIVE_ARB:
                active.append(entry)
            elif total < NEAR_ARB:
                near.append(entry)
            else:
                normal += 1

        log.info(
            "arb_scan",
            deep_arbs=len(deep),
            active_arbs=len(active),
            near_arbs=len(near),
            normal_priced=normal,
            no_quote=no_quote,
            scanned=len(self.markets),
        )

        # Surface the strongest opportunities so the user can see what's there
        for m, ya, na, total in sorted(deep, key=lambda t: t[3])[:5]:
            log.info(
                "arb_opportunity_deep",
                question=m.question[:80],
                yes_ask=round(ya, 4),
                no_ask=round(na, 4),
                sum=round(total, 4),
                edge_pp=round((1 - total) * 100, 2),
            )
        for m, ya, na, total in sorted(active, key=lambda t: t[3])[:3]:
            log.info(
                "arb_opportunity_active",
                question=m.question[:80],
                yes_ask=round(ya, 4),
                no_ask=round(na, 4),
                sum=round(total, 4),
                edge_pp=round((1 - total) * 100, 2),
            )

        # Monotonicity scan: groups of "by date" sibling markets
        groups: dict[str, list[Market]] = defaultdict(list)
        for m in self.markets:
            groups[_question_prefix(m.question)].append(m)
        violations: list[tuple[str, Market, Market, float]] = []
        for prefix, group in groups.items():
            if len(group) < 2:
                continue
            # Compute YES mid for each member (skip those without both-side quotes)
            mids: list[tuple[Market, float]] = []
            for m in group:
                b = self.book_store.books.get(m.yes_token_id)
                if b is None:
                    continue
                mid = b.mid()
                if mid is None:
                    continue
                mids.append((m, mid))
            if len(mids) < 2:
                continue
            # We can't reliably extract dates without a parser, so just report when
            # the spread between cheapest and richest is large — those pairs are
            # candidates for monotonicity arb but need date-aware ordering.
            mids.sort(key=lambda t: t[1])
            cheap = mids[0]
            rich = mids[-1]
            if rich[1] - cheap[1] > 0.10:
                violations.append((prefix, cheap[0], rich[0], rich[1] - cheap[1]))

        if violations:
            for prefix, cheap, rich, gap in sorted(violations, key=lambda t: -t[3])[:3]:
                log.info(
                    "arb_monotonicity_candidate",
                    prefix=prefix,
                    cheap=cheap.question[:60],
                    cheap_yes_mid=round(self.book_store.books[cheap.yes_token_id].mid() or 0, 3),
                    rich=rich.question[:60],
                    rich_yes_mid=round(self.book_store.books[rich.yes_token_id].mid() or 0, 3),
                    spread_pp=round(gap * 100, 2),
                )
