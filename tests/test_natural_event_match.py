"""Tests for the natural-event → market matcher.

The rules are pure functions over (market, event), so we test them
without the async wiring. Goal: confirm the parser pulls the right
threshold and deadline from a real Polymarket question, then matches
or abstains correctly given event magnitude / time / region.
"""
from __future__ import annotations

from datetime import datetime, timezone

from polyagent.data.natural_events import NaturalEvent
from polyagent.gamma import Market
from polyagent.signals.natural_event_match import (
    _match_earthquake,
    _parse_deadline,
    _parse_eq_threshold,
    _passes_region,
)


def _market(question: str, end_date_iso: str = "2030-01-01T00:00:00Z") -> Market:
    return Market(
        condition_id="0xfake",
        question=question,
        yes_token_id="t_yes",
        no_token_id="t_no",
        end_date_iso=end_date_iso,
        liquidity=1000.0,
        volume_24h=1000.0,
        accepting_orders=True,
        category="weather",
    )


def _eq_event(mag: float, place: str, occurred_iso: str) -> NaturalEvent:
    ts = datetime.fromisoformat(occurred_iso.replace("Z", "+00:00")).timestamp()
    return NaturalEvent(
        event_id=f"usgs:test_{mag}",
        source="usgs",
        category="earthquakes",
        title=f"M{mag} {place}",
        magnitude=mag,
        magnitude_unit="M",
        place=place,
        lat=0.0, lon=0.0,
        occurred_ts=ts,
    )


# ── Threshold parsing ─────────────────────────────────────────────────

def test_parse_threshold_or_above():
    assert _parse_eq_threshold("Another 7.0 or above earthquake by April 30, 2026?") == 7.0


def test_parse_threshold_or_higher():
    assert _parse_eq_threshold("Will there be exactly 1 earthquake of magnitude 7.0 or higher worldwide by June 30?") == 7.0


def test_parse_threshold_no_match():
    assert _parse_eq_threshold("Will Trump win the election?") is None


# ── Deadline parsing ──────────────────────────────────────────────────

def test_parse_deadline_us_date():
    ts = _parse_deadline("Another 7.0+ earthquake by April 30, 2026?")
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert (d.year, d.month, d.day) == (2026, 4, 30)


def test_parse_deadline_iso():
    ts = _parse_deadline("X by 2026-12-15?")
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert (d.year, d.month, d.day) == (2026, 12, 15)


def test_parse_deadline_falls_back_to_market_end():
    ts = _parse_deadline("Question with no deadline phrase", "2026-08-15T00:00:00Z")
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert (d.year, d.month, d.day) == (2026, 8, 15)


# ── End-to-end rule tests ─────────────────────────────────────────────

def test_match_yes_eq_above_threshold_before_deadline():
    mkt = _market("Another 7.0 or above earthquake by June 30, 2026?")
    evt = _eq_event(7.4, "100km ENE of Miyako, Japan", "2026-06-01T12:00:00Z")
    res = _match_earthquake(mkt, evt)
    assert res is not None
    assert res.direction == "yes"
    assert res.confidence == 1.0
    assert res.threshold == 7.0


def test_no_match_below_threshold():
    mkt = _market("Another 7.0 or above earthquake by June 30, 2026?")
    evt = _eq_event(6.5, "Chile", "2026-06-01T12:00:00Z")
    assert _match_earthquake(mkt, evt) is None


def test_no_match_after_deadline():
    mkt = _market("Another 7.0 or above earthquake by June 30, 2026?")
    evt = _eq_event(7.5, "Chile", "2026-07-01T12:00:00Z")
    assert _match_earthquake(mkt, evt) is None


def test_no_match_wrong_category():
    mkt = _market("Will Bitcoin hit $150k by June 30, 2026?")
    evt = _eq_event(7.5, "Chile", "2026-06-01T12:00:00Z")
    assert _match_earthquake(mkt, evt) is None


def test_region_constraint_rejects_wrong_place():
    mkt = _market("Will there be a 6.0+ earthquake in California by June 30, 2026?")
    evt = _eq_event(6.5, "Chile", "2026-06-01T12:00:00Z")
    assert _passes_region(mkt.question, evt.place) is False


def test_region_constraint_accepts_correct_place():
    mkt = _market("Will there be a 6.0+ earthquake in California by June 30, 2026?")
    evt = _eq_event(6.5, "5km W of Cobb, California", "2026-06-01T12:00:00Z")
    assert _passes_region(mkt.question, evt.place) is True


def test_no_region_constraint_allows_anywhere():
    mkt = _market("Another 7.0 or above earthquake by April 30, 2026?")
    assert _passes_region(mkt.question, "Chile") is True
    assert _passes_region(mkt.question, "Indonesia") is True


def test_event_with_no_magnitude_skipped():
    mkt = _market("Another 7.0 or above earthquake by June 30, 2026?")
    evt = NaturalEvent(
        event_id="usgs:malformed", source="usgs", category="earthquakes",
        title="Unknown event", occurred_ts=datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    assert _match_earthquake(mkt, evt) is None
