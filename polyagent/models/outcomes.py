"""Signal-outcome materializer.

When a market resolves, we want one labeled training row per market with a
column per expert (stat_lgbm, news_match, ...). This module:

1. Defines the `signal_outcomes` table.
2. `materialize_outcome(condition_id, question, yes_won, predictor)` —
   queries signals + computes stat_lgbm probability, writes the row.
3. `bootstrap_from_resolutions(predictor)` — populates rows for every
   already-resolved market in the resolutions table (no live signals
   needed: stat_lgbm is question-only).

The combiner trainer reads from this table.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import aiosqlite
import structlog

from polyagent.config import settings
from polyagent.models.categorize import categorize
from polyagent.models.lgbm import Predictor

log = structlog.get_logger()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_outcomes (
    condition_id TEXT PRIMARY KEY,
    resolved_ts REAL,
    yes_won INTEGER,
    question TEXT,
    p_stat_lgbm REAL,
    p_news_match REAL,
    p_market_pre REAL,
    n_news_signals INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS signal_outcomes_ts ON signal_outcomes(resolved_ts);
"""

# Extra columns added later. SQLite doesn't allow IF NOT EXISTS on ADD COLUMN,
# so we check pragma + skip if present.
_EXTRA_COLUMNS = [
    ("p_market_1h", "REAL"),
    ("p_market_6h", "REAL"),
    ("p_market_24h", "REAL"),
    ("p_market_7d", "REAL"),
    ("category", "TEXT"),
]


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)")}
    for name, type_ in _EXTRA_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {name} {type_}")
    conn.commit()


async def _ensure_table_async(db) -> None:
    await db.executescript(SCHEMA_SQL)
    async with db.execute("PRAGMA table_info(signal_outcomes)") as cur:
        cols = {row[1] async for row in cur}
    for name, type_ in _EXTRA_COLUMNS:
        if name not in cols:
            await db.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {name} {type_}")
    await db.commit()


def _aggregate_news_signals(
    rows: list,
    half_life_sec: float | None = None,
    now_ts: float | None = None,
) -> tuple[float | None, int, float | None]:
    """Aggregate news_keyword_match signals -> (P_yes_from_news, count, _).

    Same decay logic as `NewsStore.news_match_p_yes`: direction*confidence
    weighted by exp(-age/tau) when half_life_sec is set, otherwise uniform.

    `rows` items must support sequence access for ts (index 1 = ts) when
    decay is in use; supports both sqlite3.Row (passed by sync materializer)
    and dict-shaped rows (passed by async materializer).
    """
    tau = (half_life_sec / math.log(2)) if (half_life_sec and half_life_sec > 0) else None
    now_ts = now_ts if now_ts is not None else time.time()
    weighted_y = 0.0
    total_w = 0.0
    ys_uniform: list[float] = []
    for r in rows:
        # Support sqlite3.Row, plain dict, and tuple
        strategy = r["strategy"] if hasattr(r, "keys") else r.get("strategy")
        if strategy != "news_keyword_match":
            continue
        detail = r["detail"] if hasattr(r, "keys") else r.get("detail")
        direction = r["direction"] if hasattr(r, "keys") else r.get("direction")
        ts = r["ts"] if hasattr(r, "keys") and "ts" in r.keys() else (r.get("ts") if not hasattr(r, "keys") else None)
        try:
            d = json.loads(detail or "{}")
        except json.JSONDecodeError:
            continue
        confidence = float(d.get("confidence") or 0.0)
        dr = (direction or "").lower()
        if dr == "yes":
            y = confidence
        elif dr == "no":
            y = -confidence
        else:
            y = 0.0
        if tau is None or ts is None:
            ys_uniform.append(y)
        else:
            age = now_ts - float(ts)
            w = math.exp(-age / tau)
            weighted_y += y * w
            total_w += w
    p_news = None
    if tau is None or ts is None:
        if ys_uniform:
            ybar = sum(ys_uniform) / len(ys_uniform)
            p_news = max(0.001, min(0.999, 0.5 + 0.5 * ybar))
        return p_news, len(ys_uniform), None
    if total_w > 0:
        ybar = weighted_y / total_w
        p_news = max(0.001, min(0.999, 0.5 + 0.5 * ybar))
    return p_news, int(total_w > 0), None


def _market_price_from_stat_signals(rows: list[sqlite3.Row]) -> float | None:
    """Take the last stat_lgbm signal's recorded p_market (book mid at signal time)."""
    last = None
    for r in rows:
        if r["strategy"] != "stat_lgbm":
            continue
        try:
            d = json.loads(r["detail"] or "{}")
        except json.JSONDecodeError:
            continue
        last = d.get("p_market")
    if last is None:
        return None
    try:
        return float(last)
    except (TypeError, ValueError):
        return None


def materialize_outcome_sync(
    *,
    condition_id: str,
    question: str,
    yes_won: bool,
    resolved_ts: float | None = None,
    liquidity: float = 0.0,
    volume: float = 0.0,
    predictor: Predictor | None = None,
    db_path: str = settings.db_path,
    half_life_sec: float | None = None,
) -> bool:
    """Synchronous version (used by bootstrap script)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_table(conn)
        existing = conn.execute(
            "SELECT condition_id FROM signal_outcomes WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if existing is not None:
            return False
        rows = list(
            conn.execute(
                "SELECT strategy, direction, score, detail, ts FROM signals WHERE condition_id = ?",
                (condition_id,),
            )
        )
        p_news, n_news, _ = _aggregate_news_signals(rows, half_life_sec=half_life_sec)
        p_mkt = _market_price_from_stat_signals(rows)
        p_stat: float | None = None
        if predictor is not None:
            try:
                pred = predictor.predict(question, liquidity=liquidity, volume=volume)
                p_stat = pred["calibrated"]
            except Exception as e:
                log.warning("predict_failed", err=str(e))
        ts = resolved_ts if resolved_ts is not None else time.time()
        detail = {"n_signal_rows": len(rows)}
        cat = categorize(question)
        conn.execute(
            """INSERT INTO signal_outcomes(
                condition_id, resolved_ts, yes_won, question,
                p_stat_lgbm, p_news_match, p_market_pre,
                n_news_signals, detail, category
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                condition_id,
                ts,
                1 if yes_won else 0,
                question,
                p_stat,
                p_news,
                p_mkt,
                n_news,
                json.dumps(detail),
                cat,
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


async def materialize_outcome_async(
    *,
    condition_id: str,
    question: str,
    yes_won: bool,
    resolved_ts: float | None = None,
    liquidity: float = 0.0,
    volume: float = 0.0,
    predictor: Predictor | None = None,
    db_path: str = settings.db_path,
    half_life_sec: float | None = None,
) -> bool:
    """Async version for use inside the running bot."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(db_path)
    try:
        await _ensure_table_async(db)
        async with db.execute(
            "SELECT condition_id FROM signal_outcomes WHERE condition_id = ?",
            (condition_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            return False
        rows = []
        async with db.execute(
            "SELECT strategy, direction, score, detail, ts FROM signals WHERE condition_id = ?",
            (condition_id,),
        ) as cur:
            async for r in cur:
                rows.append({"strategy": r[0], "direction": r[1], "score": r[2], "detail": r[3], "ts": r[4]})
        p_news, _n_news, _ = _aggregate_news_signals(rows, half_life_sec=half_life_sec)
        last_p_mkt = None
        for r in rows:
            if r["strategy"] != "stat_lgbm":
                continue
            try:
                d = json.loads(r["detail"] or "{}")
            except json.JSONDecodeError:
                continue
            last_p_mkt = d.get("p_market")
        p_mkt = float(last_p_mkt) if last_p_mkt is not None else None

        p_stat: float | None = None
        if predictor is not None:
            try:
                pred = predictor.predict(question, liquidity=liquidity, volume=volume)
                p_stat = pred["calibrated"]
            except Exception as e:
                log.warning("predict_failed", err=str(e))

        ts = resolved_ts if resolved_ts is not None else time.time()
        await db.execute(
            """INSERT INTO signal_outcomes(
                condition_id, resolved_ts, yes_won, question,
                p_stat_lgbm, p_news_match, p_market_pre,
                n_news_signals, detail, category
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                condition_id,
                ts,
                1 if yes_won else 0,
                question,
                p_stat,
                p_news,
                p_mkt,
                _n_news,
                json.dumps({"n_signal_rows": len(rows)}),
                categorize(question),
            ),
        )
        await db.commit()
        return True
    finally:
        await db.close()


def bootstrap_from_resolutions(
    *,
    db_path: str = settings.db_path,
    predictor: Predictor | None = None,
    limit: int | None = None,
    batch_size: int = 512,
) -> dict:
    """For every row in resolutions, ensure a corresponding signal_outcomes row.

    Batched implementation: one DB connection, batched LGBM predict via the
    embedder's GPU batch path, single transaction for all inserts. Roughly
    30× faster than the per-row version.
    """
    if predictor is None:
        return {"inserted": 0, "skipped": 0, "total_resolutions": 0, "warning": "no predictor"}

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    _ensure_table(conn)

    # Pre-compute the set of already-materialized condition_ids so we skip them.
    existing = set(
        r[0] for r in conn.execute("SELECT condition_id FROM signal_outcomes")
    )

    rows = conn.execute(
        "SELECT condition_id, resolved_ts, yes_won, detail FROM resolutions"
    ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    total = len(rows)

    # Build the work list once, skipping rows that don't need materializing.
    pending: list[dict] = []
    skipped = 0
    for r in rows:
        cid = r["condition_id"]
        if cid in existing:
            skipped += 1
            continue
        try:
            d = json.loads(r["detail"] or "{}")
        except json.JSONDecodeError:
            d = {}
        question = d.get("question") or ""
        if not question:
            skipped += 1
            continue
        pending.append(
            {
                "cid": cid,
                "resolved_ts": r["resolved_ts"],
                "yes_won": bool(r["yes_won"]),
                "question": question,
                "liquidity": d.get("liquidity") or 0.0,
                "volume": d.get("volume") or 0.0,
            }
        )

    log.info("bootstrap_pending", n=len(pending), already=len(existing))

    inserted = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        # Batched GPU predict — embeds the whole chunk in one call.
        features = [(p["question"], p["liquidity"], p["volume"], None) for p in chunk]
        try:
            preds = predictor.predict_batch(features)
        except Exception as e:
            log.warning("bootstrap_predict_failed", err=str(e), n=len(chunk))
            continue
        # Pre-fetch existing signals once per chunk's condition_ids — much
        # cheaper than per-row queries.
        cids = [p["cid"] for p in chunk]
        sig_rows: dict[str, list[tuple[str, str]]] = {c: [] for c in cids}
        placeholder = ",".join("?" * len(cids))
        for sr in conn.execute(
            f"SELECT condition_id, strategy, direction, detail FROM signals "
            f"WHERE condition_id IN ({placeholder})",
            cids,
        ):
            sig_rows[sr[0]].append((sr[1], sr[2], sr[3]))
        # Aggregate news + market price per cid, then insert in one transaction.
        rowvals = []
        ts_now = time.time()
        for p, pred in zip(chunk, preds):
            cid = p["cid"]
            ys: list[float] = []
            last_p_mkt = None
            for strategy, direction, detail in sig_rows.get(cid, []):
                if strategy != "news_keyword_match" and strategy != "stat_lgbm":
                    continue
                try:
                    d = json.loads(detail or "{}")
                except json.JSONDecodeError:
                    continue
                if strategy == "stat_lgbm":
                    last_p_mkt = d.get("p_market")
                else:
                    conf = float(d.get("confidence") or 0.0)
                    dr = (direction or "").lower()
                    if dr == "yes":
                        ys.append(conf)
                    elif dr == "no":
                        ys.append(-conf)
                    else:
                        ys.append(0.0)
            p_news = (
                max(0.001, min(0.999, 0.5 + 0.5 * (sum(ys) / len(ys))))
                if ys
                else None
            )
            p_mkt = float(last_p_mkt) if last_p_mkt is not None else None
            p_stat = pred.get("calibrated") if pred else None
            rowvals.append(
                (
                    cid,
                    p["resolved_ts"] if p["resolved_ts"] is not None else ts_now,
                    1 if p["yes_won"] else 0,
                    p["question"],
                    p_stat,
                    p_news,
                    p_mkt,
                    len(ys),
                    json.dumps({"n_signal_rows": len(sig_rows.get(cid, []))}),
                    categorize(p["question"]),
                )
            )
        conn.executemany(
            """INSERT OR IGNORE INTO signal_outcomes(
                condition_id, resolved_ts, yes_won, question,
                p_stat_lgbm, p_news_match, p_market_pre,
                n_news_signals, detail, category
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rowvals,
        )
        conn.commit()
        inserted += len(rowvals)
        log.info(
            "bootstrap_progress",
            done=start + len(chunk),
            of=len(pending),
            inserted=inserted,
        )

    conn.close()
    return {"inserted": inserted, "skipped": skipped, "total_resolutions": total}
