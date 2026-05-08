"""Wash-trade hygiene filter (Dubach 2026).

Polymarket has a median 1% wash share with a 22% upper tail. The doc's
guidance: filter out the upper-tail markets, but don't expect a Sharpe
lift — it's hygiene, not edge.

We don't have on-chain wallet data in paper mode, so we use a runtime
proxy that's surprisingly effective: the **price-impact-per-trade**
heuristic. On a real, non-wash market each trade event moves the mid
or consumes depth; on a wash market the same wallet trades against
itself at the touch with the book unchanged. We track the ratio of
"trades that occurred without book change" as a per-token wash proxy
and refuse to trade on tokens where it crosses the configured
threshold over a meaningful sample.

Strategies query ``WashFilter.is_blacklisted(token_id)`` before placing
entries. The filter runs as an event hook hooked into the BookStore's
trade and book-change events.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class _PerTokenStats:
    # Recent trade events: True if accompanied by a book change in the
    # ``window_sec`` around the trade, else False (suspected wash).
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    last_book_change_ts: float = 0.0
    blacklisted: bool = False


@dataclass
class WashFilter:
    max_wash_share: float = 0.30   # threshold above which a token is skipped
    window_sec: float = 30.0        # tolerance window between trade and book change
    min_samples: int = 30           # need at least this many trades before blacklisting
    enabled: bool = True
    _stats: dict[str, _PerTokenStats] = field(default_factory=dict)

    def _get(self, token_id: str) -> _PerTokenStats:
        s = self._stats.get(token_id)
        if s is None:
            s = _PerTokenStats()
            self._stats[token_id] = s
        return s

    def on_book_change(self, token_id: str) -> None:
        if not self.enabled:
            return
        s = self._get(token_id)
        s.last_book_change_ts = time.time()

    def on_trade(self, token_id: str) -> None:
        if not self.enabled:
            return
        s = self._get(token_id)
        # If the most recent book change is within window, treat as a "real"
        # trade. Otherwise treat as wash-suspect.
        moved = (time.time() - s.last_book_change_ts) <= self.window_sec
        s.events.append(bool(moved))
        if len(s.events) >= self.min_samples:
            wash_share = 1.0 - (sum(1 for e in s.events if e) / len(s.events))
            if wash_share > self.max_wash_share and not s.blacklisted:
                s.blacklisted = True
                log.warning(
                    "wash_filter_blacklist",
                    token_id=token_id[:12],
                    wash_share=round(wash_share, 3),
                    n_samples=len(s.events),
                )
            elif wash_share <= self.max_wash_share / 2 and s.blacklisted:
                # Hysteresis: only un-blacklist when share drops well below
                # threshold.
                s.blacklisted = False
                log.info("wash_filter_unblacklist", token_id=token_id[:12])

    def is_blacklisted(self, token_id: str) -> bool:
        s = self._stats.get(token_id)
        return bool(s and s.blacklisted)

    def summary(self) -> dict:
        n_total = len(self._stats)
        n_black = sum(1 for s in self._stats.values() if s.blacklisted)
        return {"n_tokens_seen": n_total, "n_blacklisted": n_black}
