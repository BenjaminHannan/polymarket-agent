"""SQLite-backed news + signal store with content-hash dedup."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiosqlite
import structlog

from polyagent.config import settings

log = structlog.get_logger()

_SEEN_MAX = 100_000  # bound the in-memory dedup ring

# Source credibility weights for news_match aggregation. Reuters/AP wires
# get full weight; aggregators and partisan outlets get less; firehose
# sources (4chan, telegram channels) get the least. Unknown sources -> 1.0.
_SOURCE_WEIGHTS: dict[str, float] = {
    "rss:reuters_world": 1.5,
    "rss:ap_top": 1.5,
    "rss:ap_politics": 1.4,
    "rss:bbc_news": 1.3,
    "rss:bbc_world": 1.3,
    "rss:bbc_business": 1.3,
    "rss:npr_news": 1.2,
    "rss:npr_politics": 1.2,
    "rss:guardian_world": 1.2,
    "rss:cnbc_top": 1.1,
    "rss:cnbc_business": 1.1,
    "rss:fed_press": 1.5,
    "rss:fed_monetary": 1.5,
    "rss:treasury": 1.4,
    "rss:sec_press": 1.4,
    "rss:federal_register": 1.0,
    "rss:scotusblog": 1.3,
    "rss:ecb_press": 1.3,
    "rss:imf_news": 1.2,
    "rss:whitehouse_briefings": 1.2,
    "rss:thehill": 1.0,
    "rss:politico": 1.1,
    "rss:aljazeera": 1.0,
    "fred": 1.5,
    "bls": 1.5,
    "congress": 1.2,
    "courtlistener": 1.2,
    "sec_edgar": 1.4,
    "bluesky": 0.8,
}


def source_weight(src: str) -> float:
    return _SOURCE_WEIGHTS.get(src, 1.0)


@dataclass
class NewsEvent:
    source: str  # "rss:reuters_world", "bluesky", "fred", "edgar", "courtlistener", ...
    title: str
    body: str = ""
    url: str = ""
    ts: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    def hash(self) -> str:
        key = f"{self.source}|{self.url or self.title}"
        return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:16]


class NewsStore:
    def __init__(self, db_path: str = settings.db_path) -> None:
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None
        # OrderedDict acts as a bounded LRU. SQL `INSERT OR IGNORE` is the
        # ground-truth dedup; this is just a fast-path so we don't hit the DB
        # for every duplicate.
        self._seen: OrderedDict[str, None] = OrderedDict()

    async def open(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.db_path, timeout=30.0)
        await self.db.execute("PRAGMA journal_mode=WAL")
        await self.db.execute("PRAGMA synchronous=NORMAL")
        await self.db.execute("PRAGMA busy_timeout=10000")
        await self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS news (
                hash TEXT PRIMARY KEY,
                ts REAL,
                source TEXT,
                title TEXT,
                body TEXT,
                url TEXT,
                extra TEXT
            );
            CREATE INDEX IF NOT EXISTS news_ts ON news(ts);
            CREATE INDEX IF NOT EXISTS news_source_ts ON news(source, ts);

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                strategy TEXT,
                condition_id TEXT,
                direction TEXT,
                score REAL,
                news_hash TEXT,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS signals_ts ON signals(ts);
            """
        )
        await self.db.commit()

        async with self.db.execute(
            f"SELECT hash FROM news ORDER BY ts DESC LIMIT {_SEEN_MAX}"
        ) as cur:
            async for row in cur:
                self._seen[row[0]] = None
        log.info("news_store_open", warm_dedup=len(self._seen))

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    async def insert(self, evt: NewsEvent) -> bool:
        """Returns True if newly inserted (i.e., not a duplicate)."""
        if self.db is None:
            return False
        h = evt.hash()
        if h in self._seen:
            self._seen.move_to_end(h)
            return False
        self._seen[h] = None
        if len(self._seen) > _SEEN_MAX:
            # LRU eviction to keep memory bounded; SQL still dedups via INSERT OR IGNORE.
            self._seen.popitem(last=False)
        try:
            await self.db.execute(
                "INSERT OR IGNORE INTO news(hash, ts, source, title, body, url, extra) VALUES (?,?,?,?,?,?,?)",
                (h, evt.ts, evt.source, evt.title, evt.body, evt.url, json.dumps(evt.extra)),
            )
            await self.db.commit()
        except Exception as e:
            log.warning("news_insert_error", err=str(e))
            return False
        return True

    async def insert_signal(
        self,
        *,
        strategy: str,
        condition_id: str,
        direction: str,
        score: float,
        news_hash: str,
        detail: dict,
    ) -> None:
        if self.db is None:
            return
        # Sustained-write contention can occasionally beat the
        # busy_timeout. Retry a few times with backoff so transient
        # OperationalError doesn't crash the supervised task.
        import asyncio as _aio
        import sqlite3 as _sq3
        for attempt in range(4):
            try:
                await self.db.execute(
                    "INSERT INTO signals(ts, strategy, condition_id, direction, score, news_hash, detail) VALUES (?,?,?,?,?,?,?)",
                    (time.time(), strategy, condition_id, direction, score, news_hash, json.dumps(detail)),
                )
                await self.db.commit()
                return
            except _sq3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 3:
                    await _aio.sleep(0.2 * (2 ** attempt))
                    continue
                raise

    async def news_velocity(self, condition_id: str, short_sec: float = 3600.0, long_sec: float = 86400.0) -> tuple[int, int, float] | None:
        """Returns (n_short, n_long, velocity_ratio) for a market.

        velocity_ratio = (n_short / short_sec) / max(1, n_long / long_sec).
        > 1 means recent news is more active than the daily baseline; useful
        gate for "this market is hot right now."
        """
        if self.db is None:
            return None
        now = time.time()
        async with self.db.execute(
            "SELECT COUNT(*) FROM signals WHERE strategy = 'news_keyword_match' AND condition_id = ? AND ts >= ?",
            (condition_id, now - short_sec),
        ) as cur:
            n_short = int((await cur.fetchone() or [0])[0])
        async with self.db.execute(
            "SELECT COUNT(*) FROM signals WHERE strategy = 'news_keyword_match' AND condition_id = ? AND ts >= ?",
            (condition_id, now - long_sec),
        ) as cur:
            n_long = int((await cur.fetchone() or [0])[0])
        rate_short = n_short / max(1.0, short_sec)
        rate_long = n_long / max(1.0, long_sec)
        velocity = rate_short / max(1e-9, rate_long)
        return (n_short, n_long, velocity)

    async def sentiment_zscore(self, condition_id: str, window_sec: float = 7 * 86400) -> float | None:
        """Recent news direction*confidence vs 7-day baseline z-score.

        z > 0  → recent news more bullish than usual on this market
        z < 0  → recent news more bearish than usual
        Useful as a "rate of change" signal, complementing the level signal
        in news_match_p_yes.
        """
        if self.db is None:
            return None
        now = time.time()
        ys: list[float] = []
        ts_list: list[float] = []
        async with self.db.execute(
            "SELECT direction, detail, ts FROM signals "
            "WHERE strategy = 'news_keyword_match' AND condition_id = ? AND ts >= ?",
            (condition_id, now - window_sec),
        ) as cur:
            async for direction, detail, ts in cur:
                try:
                    d = json.loads(detail or "{}")
                except json.JSONDecodeError:
                    continue
                conf = float(d.get("confidence") or 0.0)
                dr = (direction or "").lower()
                if dr == "yes":
                    y = conf
                elif dr == "no":
                    y = -conf
                else:
                    y = 0.0
                ys.append(y)
                ts_list.append(float(ts or now))
        if len(ys) < 5:
            return None
        # Compare last 1h to full window
        recent_cutoff = now - 3600
        recent = [y for y, t in zip(ys, ts_list) if t >= recent_cutoff]
        if not recent:
            return 0.0
        import statistics as _stats
        try:
            mu = _stats.mean(ys)
            sigma = _stats.pstdev(ys) or 1e-6
        except _stats.StatisticsError:
            return None
        recent_mean = sum(recent) / len(recent)
        return (recent_mean - mu) / sigma

    async def news_match_p_yes(
        self,
        condition_id: str,
        window_sec: float = 86400.0,
        half_life_sec: float | None = None,
    ) -> float | None:
        """Aggregate recent news_keyword_match signals for one market into P(YES).

        direction*confidence -> y_i in [-1, 1]. If `half_life_sec` is set,
        signals are weighted by exp(-age / tau) where tau = half_life / ln(2);
        otherwise simple uniform mean (legacy behavior).

        Returns None if there are no recent signals.
        """
        if self.db is None:
            return None
        now = time.time()
        cutoff = now - window_sec
        tau = (half_life_sec / math.log(2)) if (half_life_sec and half_life_sec > 0) else None
        weighted_y = 0.0
        total_w = 0.0
        ys_uniform: list[float] = []
        async with self.db.execute(
            "SELECT direction, detail, ts FROM signals "
            "WHERE strategy = 'news_keyword_match' AND condition_id = ? AND ts >= ?",
            (condition_id, cutoff),
        ) as cur:
            async for direction, detail, ts in cur:
                try:
                    d = json.loads(detail or "{}")
                except json.JSONDecodeError:
                    continue
                conf = float(d.get("confidence") or 0.0)
                src = d.get("source") or ""
                src_w = source_weight(src)
                dr = (direction or "").lower()
                if dr == "yes":
                    y = conf
                elif dr == "no":
                    y = -conf
                else:
                    y = 0.0
                if tau is None:
                    # Source weighting acts on the unweighted aggregate too:
                    # we still average ys, but each y is scaled by source.
                    ys_uniform.append(y * src_w)
                else:
                    age = now - float(ts or now)
                    w = math.exp(-age / tau) * src_w
                    weighted_y += y * w
                    total_w += w
        if tau is None:
            if not ys_uniform:
                return None
            ybar = sum(ys_uniform) / len(ys_uniform)
        else:
            if total_w <= 0:
                return None
            ybar = weighted_y / total_w
        return max(0.001, min(0.999, 0.5 + 0.5 * ybar))
