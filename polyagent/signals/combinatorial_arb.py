"""Combinatorial arbitrage detector — NegRisk-scoped.

The 2025 IMDEA paper "Unravelling the Probabilistic Forest" found that
intra-NegRisk rebalancing is **73% of all realized combinatorial arbitrage
profit on Polymarket** ($29M of $40M total). Cross-event combinatorial arb
is 0.24%. Our previous date-prefix approach was the wrong scope.

NegRisk events are sets of N mutually-exclusive binary outcomes whose YES
prices must satisfy sum(p_yes) = 1. Deviations from that constraint are
direct arb signals.

We also keep a fallback monotonicity detector for nested-date question
families that aren't tagged as NegRisk (these are still real but rarer).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np
import structlog

from polyagent.gamma import Market
from polyagent.models.embedder import embed_batch
from polyagent.orderbook import BookStore

log = structlog.get_logger()


_DATE_RE = re.compile(
    r"\b(?:by|before|on)\s+"
    r"(?:end\s+of\s+)?"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:\s+(\d{1,2}))?"
    r",?\s*(\d{4})?",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(
    r"\s+(by|before|after|in|on)\s+.+$", re.IGNORECASE
)


_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _question_prefix(q: str) -> str:
    return _PREFIX_RE.sub("", q or "").strip().lower()[:80]


def _parse_date(q: str) -> tuple[int, int, int] | None:
    """Best-effort (year, month, day) from a question's date phrase."""
    m = _DATE_RE.search(q or "")
    if not m:
        return None
    month_word = (m.group(1) or "").lower()[:3]
    month = _MONTH_NUM.get(month_word)
    if month is None:
        return None
    day = int(m.group(2)) if m.group(2) else (28 if month == 2 else 30)
    year_s = m.group(3)
    if year_s:
        year = int(year_s)
    else:
        # Default to current year + roll forward if month is past.
        from datetime import datetime
        now = datetime.utcnow()
        year = now.year
        if month < now.month:
            year += 1
    return (year, month, day)


@dataclass
class CombinatorialArb:
    book_store: BookStore
    markets: list[Market]
    poll_sec: float = 60.0
    min_violation: float = 0.05  # 5pp minimum monotonicity violation
    log_top_n: int = 5
    trader: Optional[Callable[..., Awaitable[None]]] = None

    _groups: dict[str, list[Market]] = field(default_factory=dict)
    _ordered: dict[str, list[tuple[Market, tuple[int, int, int]]]] = field(default_factory=dict)
    _non_neg_events: dict[str, list[Market]] = field(default_factory=dict)

    # NegRisk groups (event_id-keyed) — primary signal source.
    _negrisk_events: dict[str, list[Market]] = field(default_factory=dict)

    def _build_groups(self) -> None:
        # 1. NegRisk grouping: bucket markets by event_id where neg_risk=True.
        #    Within each bucket the YES prices must sum to 1; deviations are arb.
        negrisk: dict[str, list[Market]] = defaultdict(list)
        for m in self.markets:
            if m.neg_risk and m.event_id:
                negrisk[m.event_id].append(m)
        # Even non-flagged events that share event_id form a partition where
        # sum-of-YES should be ≤ 1. Use a softer check on those.
        non_neg: dict[str, list[Market]] = defaultdict(list)
        for m in self.markets:
            if (not m.neg_risk) and m.event_id:
                non_neg[m.event_id].append(m)
        non_neg_kept = {eid: g for eid, g in non_neg.items() if len(g) >= 2}

        # 2. Date-prefix fallback for any markets without event_id but whose
        #    question phrasing indicates nested-date relationships.
        groups: dict[str, list[Market]] = defaultdict(list)
        for m in self.markets:
            if m.event_id is not None:
                continue  # already in NegRisk/non-NegRisk grouping
            groups[_question_prefix(m.question)].append(m)
        kept: dict[str, list[Market]] = {}
        ordered: dict[str, list[tuple[Market, tuple[int, int, int]]]] = {}
        for prefix, group in groups.items():
            if len(group) < 2:
                continue
            with_dates: list[tuple[Market, tuple[int, int, int]]] = []
            for m in group:
                d = _parse_date(m.question)
                if d is not None:
                    with_dates.append((m, d))
            if len(with_dates) < 2:
                continue
            with_dates.sort(key=lambda t: t[1])
            kept[prefix] = group
            ordered[prefix] = with_dates

        self._negrisk_events = {
            eid: g for eid, g in negrisk.items() if len(g) >= 2
        }
        self._non_neg_events = non_neg_kept
        self._groups = kept
        self._ordered = ordered
        log.info(
            "combinatorial_arb_groups_built",
            negrisk_events=len(self._negrisk_events),
            non_negrisk_events=len(non_neg_kept),
            date_prefix_groups=len(kept),
            total_markets_grouped=(
                sum(len(g) for g in self._negrisk_events.values())
                + sum(len(g) for g in non_neg_kept.values())
                + sum(len(g) for g in kept.values())
            ),
        )

    async def run(self) -> None:
        self._build_groups()
        if not self._negrisk_events and not self._groups and not self._non_neg_events:
            log.info("combinatorial_arb_no_groups")
            await asyncio.Event().wait()
            return
        log.info(
            "combinatorial_arb_start",
            poll_sec=self.poll_sec,
            n_negrisk_events=len(self._negrisk_events),
            n_date_groups=len(self._groups),
        )
        while True:
            await asyncio.sleep(self.poll_sec)
            try:
                self._scan_once()
            except Exception as e:
                log.warning("combinatorial_arb_scan_error", err=str(e))

    def _scan_once(self) -> None:
        violations: list[dict] = []

        # 1. NegRisk sum-to-1 check (the high-EV arb per IMDEA $29M).
        #    For each NegRisk event, sum YES asks across all outcomes; if the
        #    sum < 1 - threshold, you can buy YES on every outcome and lock
        #    in (1 - sum) profit. If sum > 1 + threshold, buy NO on every
        #    outcome to lock the inverse.
        for event_id, group in self._negrisk_events.items():
            asks = []
            for m in group:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    asks = None
                    break
                a = book.best_ask()
                if a is None:
                    asks = None
                    break
                asks.append((m, a[0]))
            if asks is None or len(asks) < 2:
                continue
            sum_yes = sum(p for _, p in asks)
            if sum_yes < 1.0 - self.min_violation:
                violations.append(
                    {
                        "kind": "negrisk_sum_lt_1",
                        "event_id": event_id,
                        "sum_yes": sum_yes,
                        "edge": 1.0 - sum_yes,
                        "n_outcomes": len(asks),
                        "members": asks,
                    }
                )
            elif sum_yes > 1.0 + self.min_violation:
                violations.append(
                    {
                        "kind": "negrisk_sum_gt_1",
                        "event_id": event_id,
                        "sum_yes": sum_yes,
                        "edge": sum_yes - 1.0,
                        "n_outcomes": len(asks),
                        "members": asks,
                    }
                )

        # 2. Non-NegRisk same-event sum-≤-1 check (softer constraint).
        for event_id, group in self._non_neg_events.items():
            mids = []
            for m in group:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    continue
                mid = book.mid()
                if mid is None:
                    continue
                mids.append((m, mid))
            if len(mids) < 2:
                continue
            sum_yes = sum(p for _, p in mids)
            # If sum > 1 + threshold across mutually-related outcomes,
            # at least some are too rich. Surface for inspection.
            if sum_yes > 1.0 + self.min_violation:
                violations.append(
                    {
                        "kind": "non_neg_sum_gt_1",
                        "event_id": event_id,
                        "sum_yes": sum_yes,
                        "edge": sum_yes - 1.0,
                        "n_outcomes": len(mids),
                        "members": mids,
                    }
                )

        # 3. Date-prefix monotonicity (legacy fallback for non-event markets).
        for prefix, ordered in self._ordered.items():
            # Among ordered (earliest -> latest), YES probability should be
            # non-decreasing. Find pairs where earlier > later by min_violation.
            mids: list[tuple[Market, float, tuple[int, int, int]]] = []
            for m, d in ordered:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    continue
                mid = book.mid()
                if mid is None:
                    continue
                mids.append((m, mid, d))
            if len(mids) < 2:
                continue
            for i in range(len(mids)):
                for j in range(i + 1, len(mids)):
                    earlier, p_e, _ = mids[i]
                    later, p_l, _ = mids[j]
                    if p_e > p_l + self.min_violation:
                        violations.append(
                            {
                                "prefix": prefix,
                                "earlier": earlier,
                                "p_earlier": p_e,
                                "later": later,
                                "p_later": p_l,
                                "edge": p_e - p_l,
                            }
                        )
        violations.sort(key=lambda v: -v.get("edge", 0))
        log.info(
            "combinatorial_arb_scan",
            n_violations=len(violations),
            n_negrisk=len(self._negrisk_events),
            n_non_neg=len(self._non_neg_events),
            n_date_groups=len(self._ordered),
        )
        for v in violations[: self.log_top_n]:
            kind = v.get("kind", "date_monotonicity")
            if kind in ("negrisk_sum_lt_1", "negrisk_sum_gt_1", "non_neg_sum_gt_1"):
                log.info(
                    "combinatorial_arb_candidate",
                    kind=kind,
                    event_id=v["event_id"],
                    sum_yes=round(v["sum_yes"], 4),
                    edge_pp=round(v["edge"] * 100, 2),
                    n_outcomes=v["n_outcomes"],
                    sample_questions=[m.question[:60] for m, _ in v["members"][:3]],
                )
                # Dispatch each member market individually with the implied
                # rebalanced fair probability so the trader can take it.
                if self.trader is not None and kind == "negrisk_sum_lt_1":
                    sum_yes = v["sum_yes"]
                    for m, p_obs in v["members"]:
                        # Each YES is undervalued by (1 - sum_yes) on a per-event basis
                        p_fair = p_obs + (1.0 - sum_yes) / v["n_outcomes"]
                        try:
                            asyncio.create_task(
                                self.trader(
                                    market=m,
                                    p_combined=min(0.99, p_fair),
                                    p_market=p_obs,
                                    category="combinatorial_arb",
                                )
                            )
                        except Exception as e:
                            log.warning("combinatorial_arb_dispatch_error", err=str(e))
            else:
                # Date-monotonicity violation
                log.info(
                    "combinatorial_arb_candidate",
                    kind="date_monotonicity",
                    prefix=v["prefix"][:60],
                    earlier=v["earlier"].question[:70],
                    p_earlier=round(v["p_earlier"], 3),
                    later=v["later"].question[:70],
                    p_later=round(v["p_later"], 3),
                    edge_pp=round(v["edge"] * 100, 2),
                )
                if self.trader is not None:
                    try:
                        asyncio.create_task(
                            self.trader(
                                market=v["earlier"],
                                p_combined=v["p_later"],
                                p_market=v["p_earlier"],
                                category="combinatorial_arb",
                            )
                        )
                    except Exception as e:
                        log.warning("combinatorial_arb_dispatch_error", err=str(e))
