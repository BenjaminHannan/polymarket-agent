"""Match incoming USGS / EONET natural events to active Polymarket markets.

The structured matchers here are deliberately narrow: each rule looks for
a specific question pattern + parses the threshold and deadline out of the
question, then checks whether the inbound event satisfies them. Anything
the rules don't recognise produces no signal — we'd rather under-emit than
mis-classify.

Resolution direction is deterministic, not probabilistic:
  - event satisfies "X by [date]?" before [date] → direction = YES
  - event would have satisfied but [date] passed → direction = NO
  - otherwise → no signal

Signals land in the `signals` table with strategy='natural_event_match'.
Trade activation is gated on the strategy_certificates table just like
every other strategy in this codebase.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, Optional

import structlog

from polyagent.data.natural_events import NaturalEvent
from polyagent.gamma import Market
from polyagent.news_store import NewsStore

log = structlog.get_logger()

# ── Deadline parsers ───────────────────────────────────────────────────
# Polymarket weather questions usually carry a deadline like
# "by April 30, 2026?" or "by 2026-05-31?" or "by May 31?".
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_BY_DATE_RE = re.compile(
    r"\bby\s+(?:"
    r"(?P<month>january|jan|february|feb|march|mar|april|apr|may|june|jun|"
    r"july|jul|august|aug|september|sept|sep|october|oct|november|nov|december|dec)"
    r"\s+(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?"
    r"|(?P<iso>\d{4}-\d{2}-\d{2})"
    r")",
    re.I,
)


def _parse_deadline(question: str, fallback_iso: str | None = None) -> float | None:
    """Return a UTC unix timestamp for the question's deadline, or None.

    Falls back to the market's `end_date_iso` if no in-question deadline
    is found.
    """
    m = _BY_DATE_RE.search(question or "")
    if m:
        if m.group("iso"):
            try:
                return datetime.fromisoformat(m.group("iso") + "T23:59:59+00:00").timestamp()
            except Exception:
                pass
        else:
            month = _MONTH_NAMES[m.group("month").lower()]
            day = int(m.group("day"))
            year = int(m.group("year")) if m.group("year") else datetime.now(timezone.utc).year
            try:
                return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc).timestamp()
            except Exception:
                pass
    if fallback_iso:
        try:
            return datetime.fromisoformat(fallback_iso.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    return None


# ── Per-rule magnitude parsing ─────────────────────────────────────────
# Order matters: the most-specific patterns are tried first.
_EQ_PATTERNS = (
    # "magnitude 7.0" or "magnitude 7"
    re.compile(r"\bmagnitude\s+(\d(?:\.\d)?)\b", re.I),
    # "M7.0+" / "M 7.0" / "M7"
    re.compile(r"\bM\s*(\d(?:\.\d)?)\s*\+?", re.I),
    # "7.0 or above earthquake" / "7.0 or higher earthquake" / "7+ earthquake"
    re.compile(
        r"\b(\d(?:\.\d)?)\s*(?:or\s+(?:above|higher|greater)|\+)?\s+earthquake",
        re.I,
    ),
)


def _parse_eq_threshold(question: str) -> float | None:
    """Find the magnitude threshold in a Polymarket earthquake question.

    Examples that match:
      "Another 7.0 or above earthquake by April 30, 2026?"   → 7.0
      "earthquake of magnitude 7.0 or higher worldwide"       → 7.0
      "M7+ earthquake"                                        → 7.0
      "exactly 1 earthquake of magnitude 7.0"                 → 7.0  (skips '1')
    """
    if not question:
        return None
    for pat in _EQ_PATTERNS:
        m = pat.search(question)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


# Region/place gate. If the question constrains to a region (e.g., "in
# California"), don't fire on events elsewhere. Most of the EQ markets
# we've seen are "worldwide" — no constraint — so this is a soft filter
# that returns True (allow) when no obvious region clause is present.
_REGION_RE = re.compile(
    r"\bin\s+(california|alaska|japan|chile|indonesia|turkey|"
    r"the (?:us|united states|usa)|nepal|peru|mexico|"
    r"philippines|new zealand|iran|italy|greece|"
    r"taiwan|fiji|tonga|vanuatu|papua new guinea|alaska|hawaii)\b",
    re.I,
)


def _passes_region(question: str, place: str | None) -> bool:
    m = _REGION_RE.search(question or "")
    if not m:
        return True  # no region constraint → matches anywhere
    region = m.group(1).lower().replace("the ", "")
    if not place:
        return False
    return region.split()[0] in place.lower()


# ── Match rules ─────────────────────────────────────────────────────────
@dataclass
class MatchResult:
    direction: str           # "yes" | "no"
    confidence: float        # 1.0 when deterministic
    rule: str
    reason: str
    market: Market
    event: NaturalEvent
    deadline_ts: float | None = None
    threshold: float | None = None


def _match_earthquake(
    market: Market, event: NaturalEvent, *, now_ts: float | None = None
) -> MatchResult | None:
    """Match USGS earthquake → Polymarket "≥X by [date]?" market.

    YES if event.magnitude >= threshold AND event.occurred_ts <= deadline.
    Doesn't emit NO directly — that's the job of the deadline-passed
    sweeper, not the streaming matcher.
    """
    if event.source != "usgs" or event.category != "earthquakes":
        return None
    if event.magnitude is None:
        return None
    if "earthquake" not in (market.question or "").lower():
        return None
    threshold = _parse_eq_threshold(market.question)
    if threshold is None:
        return None
    deadline_ts = _parse_deadline(market.question, market.end_date_iso)
    if deadline_ts is None:
        return None
    if not _passes_region(market.question, event.place):
        return None
    occurred = event.occurred_ts or (now_ts or time.time())
    if event.magnitude < threshold:
        return None
    if occurred > deadline_ts:
        return None  # event happened after the question's deadline
    return MatchResult(
        direction="yes",
        confidence=1.0,
        rule="usgs_eq_above_threshold_before_deadline",
        reason=(
            f"USGS M{event.magnitude} >= threshold M{threshold}; "
            f"occurred {time.strftime('%Y-%m-%d', time.gmtime(occurred))} "
            f"before deadline {time.strftime('%Y-%m-%d', time.gmtime(deadline_ts))}"
        ),
        market=market,
        event=event,
        deadline_ts=deadline_ts,
        threshold=threshold,
    )


# Future rules can be added here (severeStorms / wildfires / volcanoes).
# Each rule is independent and may return None to abstain.
_RULES: list[Callable[[Market, NaturalEvent], Optional[MatchResult]]] = [
    _match_earthquake,
]


# ── Matcher entry point ─────────────────────────────────────────────────
@dataclass
class NaturalEventMatcher:
    markets: list[Market]
    news_store: NewsStore
    trader: Optional[Callable[[MatchResult], Awaitable[None]]] = None

    async def on_event(self, evt: NaturalEvent) -> int:
        """Run all rules against all (relevant) markets. Returns matches found."""
        n_matches = 0
        for m in self.markets:
            if (m.category or "") != "weather":
                continue
            for rule in _RULES:
                res = rule(m, evt)
                if res is None:
                    continue
                n_matches += 1
                detail = {
                    "rule": res.rule,
                    "reason": res.reason,
                    "event_id": evt.event_id,
                    "event_source": evt.source,
                    "event_category": evt.category,
                    "event_magnitude": evt.magnitude,
                    "event_title": evt.title[:160],
                    "event_url": evt.url,
                    "deadline_ts": res.deadline_ts,
                    "threshold": res.threshold,
                    "question": m.question[:160],
                    "category": m.category,
                }
                await self.news_store.insert_signal(
                    strategy="natural_event_match",
                    condition_id=m.condition_id,
                    direction=res.direction,
                    score=res.confidence,
                    news_hash=evt.event_id,
                    detail=detail,
                )
                log.info(
                    "natural_event_signal",
                    rule=res.rule,
                    direction=res.direction,
                    confidence=res.confidence,
                    question=m.question[:90],
                    reason=res.reason[:200],
                )
                if self.trader is not None:
                    try:
                        await self.trader(res)
                    except Exception as e:
                        log.warning("natural_event_trader_error", err=str(e))
        return n_matches
