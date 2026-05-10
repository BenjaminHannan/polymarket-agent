"""Tests for the LLM-augmented weather forecaster.

The orchestration is async + LLM-dependent and lives behind a flag, so
we test the pure pieces: question classification, base-rate
computation against an in-memory DB, and prompt construction.
"""
from __future__ import annotations

import sqlite3
import time
import json

from polyagent.gamma import Market
from polyagent.signals.weather_forecaster import (
    _classify_question,
    _eq_base_rate,
    _recent_events,
    build_prompt,
)


def _setup_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE natural_events (
            event_id TEXT PRIMARY KEY, source TEXT, category TEXT,
            title TEXT, magnitude REAL, magnitude_unit TEXT,
            place TEXT, lat REAL, lon REAL,
            occurred_ts REAL, seen_ts REAL, url TEXT, raw TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE resolutions (
            condition_id TEXT, resolved_ts REAL, yes_won INT,
            yes_token_id TEXT, no_token_id TEXT,
            yes_size REAL, no_size REAL,
            yes_avg_cost REAL, no_avg_cost REAL,
            yes_payout REAL, no_payout REAL,
            pnl REAL, detail TEXT
        )"""
    )
    return conn


def test_classify_earthquake_question():
    cls = _classify_question("Another 7.0 or above earthquake by April 30, 2026?")
    assert cls is not None
    assert cls["kind"] == "earthquake_count"
    assert cls["magnitude"] == 7.0


def test_classify_hurricane_question():
    cls = _classify_question("Will Hurricane Beryl reach Cat 5 by July 31?")
    assert cls is not None
    assert cls["kind"] == "tropical_cyclone"


def test_classify_wildfire_question():
    cls = _classify_question("Will the Pine Mountain wildfire exceed 5000 acres by May 31?")
    assert cls is not None
    assert cls["kind"] == "wildfire"


def test_classify_volcano_question():
    cls = _classify_question("Will Mt. Etna have a confirmed eruption by August 1?")
    assert cls is not None
    assert cls["kind"] == "volcano"


def test_classify_unrelated_returns_none():
    assert _classify_question("Will Trump win the 2024 election?") is None
    assert _classify_question("Will Bitcoin hit $150k?") is None


def test_eq_base_rate_uses_historical_resolutions(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE natural_events (
            event_id TEXT PRIMARY KEY, source TEXT, category TEXT,
            title TEXT, magnitude REAL, magnitude_unit TEXT,
            place TEXT, lat REAL, lon REAL,
            occurred_ts REAL, seen_ts REAL, url TEXT, raw TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE resolutions (
            condition_id TEXT, resolved_ts REAL, yes_won INT,
            yes_token_id TEXT, no_token_id TEXT,
            yes_size REAL, no_size REAL,
            yes_avg_cost REAL, no_avg_cost REAL,
            yes_payout REAL, no_payout REAL,
            pnl REAL, detail TEXT
        )"""
    )
    # 6 historical "M7+ earthquake by [date]" markets, 4 YES 2 NO
    samples = [
        (1, "Another 7.0 or above earthquake by April 30, 2026?"),
        (1, "Another 7.0 or above earthquake by May 31, 2026?"),
        (1, "Another 7.0 or above earthquake by March 31, 2026?"),
        (1, "Another 7.0 or above earthquake by January 15, 2026?"),
        (0, "Another 7.0 or above earthquake by February 1, 2026?"),
        (0, "Another 7.0 or above earthquake by April 30, 2026?"),
    ]
    for i, (yw, q) in enumerate(samples):
        conn.execute(
            "INSERT INTO resolutions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"cid{i}", time.time(), yw, "yt", "nt",
             0, 0, 0, 0, 0, 0, 0,
             json.dumps({"question": q})),
        )
    conn.commit()
    conn.close()

    br = _eq_base_rate(str(db), magnitude=7.0, window_days=30)
    assert br is not None
    assert br["method"] == "historical_resolutions"
    assert br["n"] == 6
    assert abs(br["p_yes"] - 4 / 6) < 1e-9


def test_eq_base_rate_falls_back_to_poisson_when_no_resolutions(tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE natural_events (
            event_id TEXT PRIMARY KEY, source TEXT, category TEXT,
            title TEXT, magnitude REAL, magnitude_unit TEXT,
            place TEXT, lat REAL, lon REAL,
            occurred_ts REAL, seen_ts REAL, url TEXT, raw TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE resolutions (
            condition_id TEXT, resolved_ts REAL, yes_won INT,
            yes_token_id TEXT, no_token_id TEXT,
            yes_size REAL, no_size REAL,
            yes_avg_cost REAL, no_avg_cost REAL,
            yes_payout REAL, no_payout REAL,
            pnl REAL, detail TEXT
        )"""
    )
    # 12 USGS earthquakes spanning 365 days, 6 of M7+
    base_ts = time.time() - 365 * 86400
    for i in range(12):
        mag = 7.5 if i % 2 == 0 else 6.0
        conn.execute(
            """INSERT INTO natural_events
            (event_id, source, category, title, magnitude, magnitude_unit,
             place, lat, lon, occurred_ts, seen_ts, url, raw)
            VALUES (?, 'usgs', 'earthquakes', ?, ?, 'M', ?, 0, 0, ?, ?, '', '{}')""",
            (f"id{i}", f"M{mag} test", mag, "place", base_ts + i * 30 * 86400, time.time()),
        )
    conn.commit()
    conn.close()

    br = _eq_base_rate(str(db), magnitude=7.0, window_days=30)
    assert br is not None
    assert br["method"] == "poisson_rate_natural_events"
    assert br["n"] == 6  # 6 M7+ events
    # 6 events over ~330 days = ~0.018/day. P(>=1 in 30) = 1 - e^(-0.55) ≈ 0.42
    assert 0.30 < br["p_yes"] < 0.55


def test_recent_events_filters_by_category(tmp_path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE natural_events (
            event_id TEXT PRIMARY KEY, source TEXT, category TEXT,
            title TEXT, magnitude REAL, magnitude_unit TEXT,
            place TEXT, lat REAL, lon REAL,
            occurred_ts REAL, seen_ts REAL, url TEXT, raw TEXT
        )"""
    )
    now = time.time()
    rows = [
        ("1", "usgs", "earthquakes", "M7", 7.0, now - 5 * 86400),
        ("2", "eonet", "wildfires", "Pine Mountain", 5000.0, now - 1 * 86400),
        ("3", "eonet", "wildfires", "Old", 100.0, now - 60 * 86400),  # too old
    ]
    for r in rows:
        conn.execute(
            """INSERT INTO natural_events
            (event_id, source, category, title, magnitude, magnitude_unit,
             place, lat, lon, occurred_ts, seen_ts, url, raw)
            VALUES (?,?,?,?,?,'?', '', 0, 0, ?, ?, '', '{}')""",
            (r[0], r[1], r[2], r[3], r[4], r[5], time.time()),
        )
    conn.commit()
    conn.close()

    eq = _recent_events(str(db), days=30, category="earthquakes")
    assert len(eq) == 1
    assert eq[0]["title"] == "M7"
    fires = _recent_events(str(db), days=30, category="wildfires")
    assert len(fires) == 1  # the 60-day-old one is filtered out
    assert fires[0]["title"] == "Pine Mountain"
    all_evs = _recent_events(str(db), days=90, category=None)
    assert len(all_evs) == 3


def test_build_prompt_contains_required_fields():
    market = Market(
        condition_id="0xabc", question="Another 7.0 or above earthquake by June 30, 2026?",
        yes_token_id="yt", no_token_id="nt",
        end_date_iso="2026-06-30T00:00:00Z",
        liquidity=10000.0, volume_24h=5000.0,
        accepting_orders=True, category="weather",
    )
    cls = {"kind": "earthquake_count", "magnitude": 7.0}
    base_rate = {"p_yes": 0.42, "n": 6, "method": "poisson_rate_natural_events", "rate_per_day": 0.018}
    events = [
        {"title": "M7.4 Japan", "magnitude": 7.4, "place": "Japan",
         "occurred_iso": "2026-04-15", "category": "earthquakes"},
    ]
    prompt = build_prompt(market, classification=cls, base_rate=base_rate, events=events, days_to_deadline=14)
    assert "Probability:" in prompt
    assert "Another 7.0 or above earthquake" in prompt
    assert "Days to deadline: 14" in prompt
    assert "M7.0" in prompt or "M7" in prompt
    assert "0.42" in prompt or "0.420" in prompt
    assert "M7.4 Japan" in prompt
    assert "earthquake" in prompt.lower()
    # Anti-confabulation rule should be present
    assert "do NOT confabulate" in prompt or "do not confabulate" in prompt.lower()
