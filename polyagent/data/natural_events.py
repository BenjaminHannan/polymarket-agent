"""Natural-event ingest from USGS earthquake feeds and NASA EONET.

Both APIs are free / no-auth / no-key. We poll on a tighter cadence than
RSS because these events are time-sensitive: an M7+ earthquake reported by
USGS within minutes is usually ahead of social-media propagation, and
Polymarket "Another 7.0+ earthquake by [date]?" markets often take 10-30
minutes to fully reprice. EONET tracks severe storms, wildfires, volcanoes,
floods, etc. — slower-moving events but the same trade-then-confirm logic.

Persisted to a `natural_events` table. The matcher (signals/natural_event_match.py)
consumes these records and emits market-direction signals.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import structlog

from polyagent.config import settings

log = structlog.get_logger()

USGS_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
EONET_API = "https://eonet.gsfc.nasa.gov/api/v3/events"

# USGS feeds we poll. Each is the "significant_*" feed (curated list).
USGS_FEEDS = {
    "significant_hour": f"{USGS_FEED}/significant_hour.geojson",
    "significant_day": f"{USGS_FEED}/significant_day.geojson",
    "significant_week": f"{USGS_FEED}/significant_week.geojson",
}

# EONET categories worth watching (skip drought/seaLakeIce/etc. that
# don't usually have Polymarket markets).
EONET_CATEGORIES = ["severeStorms", "wildfires", "volcanoes", "floods"]


@dataclass
class NaturalEvent:
    event_id: str          # provider-stable id ("usgs:us7000abcd" / "eonet:EONET_12345")
    source: str            # "usgs" | "eonet"
    category: str          # "earthquakes" | "severeStorms" | "wildfires" | ...
    title: str             # human-readable summary, e.g. "M7.4 100km ENE of Miyako, Japan"
    magnitude: Optional[float] = None  # M for earthquakes, wind speed / acres / etc. for EONET
    magnitude_unit: Optional[str] = None
    place: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    occurred_ts: float = 0.0  # event time
    seen_ts: float = field(default_factory=time.time)
    url: Optional[str] = None
    raw: dict = field(default_factory=dict)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS natural_events (
            event_id     TEXT PRIMARY KEY,
            source       TEXT NOT NULL,
            category     TEXT NOT NULL,
            title        TEXT NOT NULL,
            magnitude    REAL,
            magnitude_unit TEXT,
            place        TEXT,
            lat          REAL,
            lon          REAL,
            occurred_ts  REAL NOT NULL,
            seen_ts      REAL NOT NULL,
            url          TEXT,
            raw          TEXT
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nat_evt_occurred ON natural_events(occurred_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nat_evt_category ON natural_events(category)")
    conn.commit()


def _persist(conn: sqlite3.Connection, evt: NaturalEvent) -> bool:
    """Insert if new. Returns True if inserted (i.e. first time seen)."""
    cur = conn.execute("SELECT 1 FROM natural_events WHERE event_id = ?", (evt.event_id,))
    if cur.fetchone():
        return False
    conn.execute(
        """INSERT INTO natural_events
           (event_id, source, category, title, magnitude, magnitude_unit,
            place, lat, lon, occurred_ts, seen_ts, url, raw)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            evt.event_id, evt.source, evt.category, evt.title,
            evt.magnitude, evt.magnitude_unit, evt.place,
            evt.lat, evt.lon, evt.occurred_ts, evt.seen_ts, evt.url,
            json.dumps(evt.raw),
        ),
    )
    conn.commit()
    return True


# ── USGS earthquakes ────────────────────────────────────────────────────

async def _fetch_usgs(session: aiohttp.ClientSession, url: str) -> list[NaturalEvent]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                log.warning("usgs_http", status=r.status)
                return []
            data = await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("usgs_error", err=str(e))
        return []
    out: list[NaturalEvent] = []
    for f in data.get("features", []):
        p = f.get("properties", {}) or {}
        g = f.get("geometry", {}) or {}
        coords = g.get("coordinates") or []
        try:
            lon = float(coords[0]) if len(coords) >= 1 else None
            lat = float(coords[1]) if len(coords) >= 2 else None
        except (TypeError, ValueError):
            lon = lat = None
        try:
            mag = float(p.get("mag")) if p.get("mag") is not None else None
        except (TypeError, ValueError):
            mag = None
        try:
            ts_ms = float(p.get("time", 0))
            occurred = ts_ms / 1000.0
        except (TypeError, ValueError):
            occurred = 0.0
        out.append(NaturalEvent(
            event_id=f"usgs:{f.get('id', '')}",
            source="usgs",
            category="earthquakes",
            title=f"M{mag} {p.get('place','')}".strip() if mag is not None else (p.get("place") or "earthquake"),
            magnitude=mag,
            magnitude_unit="M",
            place=p.get("place"),
            lat=lat,
            lon=lon,
            occurred_ts=occurred,
            url=p.get("url"),
            raw=p,
        ))
    return out


# ── NASA EONET ──────────────────────────────────────────────────────────

async def _fetch_eonet(session: aiohttp.ClientSession, days: int = 30) -> list[NaturalEvent]:
    out: list[NaturalEvent] = []
    for cat in EONET_CATEGORIES:
        params = {"category": cat, "days": str(days), "status": "all", "limit": "200"}
        try:
            async with session.get(
                EONET_API, params=params, timeout=aiohttp.ClientTimeout(total=20)
            ) as r:
                if r.status != 200:
                    log.warning("eonet_http", status=r.status, category=cat)
                    continue
                data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("eonet_error", category=cat, err=str(e))
            continue
        for e in data.get("events", []):
            geom = (e.get("geometry") or [])
            # Use most recent geometry point as the event "occurred" anchor;
            # earlier points are storm-track history.
            mag = mag_unit = lat = lon = None
            occurred = 0.0
            if geom:
                last = geom[-1]
                try:
                    mag = float(last.get("magnitudeValue")) if last.get("magnitudeValue") is not None else None
                except (TypeError, ValueError):
                    mag = None
                mag_unit = last.get("magnitudeUnit")
                coords = last.get("coordinates") or []
                if last.get("type") == "Point" and len(coords) >= 2:
                    try:
                        lon = float(coords[0]); lat = float(coords[1])
                    except (TypeError, ValueError):
                        pass
                ts = last.get("date")
                if isinstance(ts, str):
                    try:
                        from datetime import datetime
                        occurred = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        pass
            cats = e.get("categories") or []
            cat_id = cats[0]["id"] if cats and cats[0].get("id") else cat
            out.append(NaturalEvent(
                event_id=f"eonet:{e.get('id','')}",
                source="eonet",
                category=cat_id,
                title=e.get("title") or "EONET event",
                magnitude=mag,
                magnitude_unit=mag_unit,
                place=e.get("description"),
                lat=lat,
                lon=lon,
                occurred_ts=occurred,
                url=e.get("link"),
                raw=e,
            ))
    return out


# ── Background polling task ─────────────────────────────────────────────

async def run(callback=None) -> None:
    """Background poller. callback (optional) is awaited for each new event."""
    if not getattr(settings, "enable_natural_events", False):
        log.warning("natural_events_disabled")
        await asyncio.Event().wait()
        return
    poll_sec = float(getattr(settings, "natural_events_poll_sec", 180))
    eonet_poll_sec = float(getattr(settings, "natural_events_eonet_poll_sec", 1800))
    log.info("natural_events_start", poll_sec=poll_sec, eonet_poll_sec=eonet_poll_sec)

    conn = sqlite3.connect(settings.db_path)
    _ensure_table(conn)
    last_eonet = 0.0
    async with aiohttp.ClientSession() as session:
        while True:
            usgs_events: list[NaturalEvent] = []
            for url in USGS_FEEDS.values():
                usgs_events.extend(await _fetch_usgs(session, url))

            eonet_events: list[NaturalEvent] = []
            now = time.time()
            if now - last_eonet >= eonet_poll_sec:
                eonet_events = await _fetch_eonet(session, days=30)
                last_eonet = now

            new_count = 0
            for evt in usgs_events + eonet_events:
                inserted = _persist(conn, evt)
                if inserted:
                    new_count += 1
                    log.info(
                        "natural_event_new",
                        source=evt.source,
                        category=evt.category,
                        mag=evt.magnitude,
                        title=evt.title[:120],
                        occurred=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(evt.occurred_ts)),
                    )
                    if callback is not None:
                        try:
                            await callback(evt)
                        except Exception as e:
                            log.warning("natural_event_cb_error", err=str(e))
            if new_count:
                log.info("natural_events_polled", new=new_count, usgs=len(usgs_events), eonet=len(eonet_events))
            await asyncio.sleep(poll_sec)
