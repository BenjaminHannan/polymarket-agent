"""Track and classify model prediction failures.

Walks the `signal_outcomes` table after each settlement (or in bulk via
`backfill_failures`) and writes structured rows to `model_failures`
documenting where, how, and how badly each prediction missed. The
output is used by:

  - the dashboard, to show recent failures + failure rate by category
  - retraining, to surface the slices where the model underperforms
  - certification, to hold strategies accountable to their claimed edge

Failure types (ordered by severity, most severe first):

  high_confidence_wrong         |p_model − 0.5| ≥ 0.30 AND model wrong-side
  medium_confidence_wrong       |p_model − 0.5| ≥ 0.15 AND wrong-side
  model_loud_wrong_market_right p_model contradicts p_market by ≥ 0.30
                                AND market was directionally correct
  model_disagrees_market_right  p_model contradicts p_market by ≥ 0.10
                                AND market was directionally correct
  combined_wrong_traded         we placed a real fill and the position
                                lost money (joined to fills + resolutions)

A single row in `signal_outcomes` can produce multiple failure rows if
multiple criteria fire (e.g., a high-confidence wrong call that we
also traded on). Each is recorded separately so the dashboard can
display per-type breakdowns honestly.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable

import structlog

log = structlog.get_logger()


# ── Schema ──────────────────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS model_failures (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id  TEXT NOT NULL,
            resolved_ts   REAL NOT NULL,
            yes_won       INTEGER NOT NULL,
            p_model       REAL,
            p_market      REAL,
            log_loss_model  REAL,
            log_loss_market REAL,
            failure_type  TEXT NOT NULL,
            severity      REAL NOT NULL,
            category      TEXT,
            question      TEXT,
            notional_traded REAL DEFAULT 0,
            realized_pnl  REAL DEFAULT 0,
            detail        TEXT,
            created_ts    REAL NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_resolved ON model_failures(resolved_ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_type ON model_failures(failure_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failures_cat ON model_failures(category)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_failures_unique "
        "ON model_failures(condition_id, failure_type)"
    )
    conn.commit()


# ── Classification ──────────────────────────────────────────────────────
@dataclass
class FailureRecord:
    failure_type: str
    severity: float  # 0..1, larger = worse miss
    detail: dict


def _ll(p: float, y: int) -> float:
    """Per-row log-loss. p clipped to [1e-3, 1-1e-3]."""
    p = max(1e-3, min(1 - 1e-3, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def classify(
    *, p_model: float | None, p_market: float | None, yes_won: int,
    notional_traded: float = 0.0, realized_pnl: float = 0.0,
) -> list[FailureRecord]:
    """Return zero or more failure records describing how the prediction missed."""
    out: list[FailureRecord] = []

    # Model wrong-side, classified by confidence
    if p_model is not None:
        wrong_side = (p_model > 0.5 and yes_won == 0) or (p_model < 0.5 and yes_won == 1)
        if wrong_side:
            confidence = abs(p_model - 0.5) * 2  # 0..1
            severity = float(_ll(p_model, yes_won))
            if confidence >= 0.6:
                out.append(FailureRecord(
                    "high_confidence_wrong", severity,
                    {"p_model": round(p_model, 4), "confidence": round(confidence, 3)},
                ))
            elif confidence >= 0.3:
                out.append(FailureRecord(
                    "medium_confidence_wrong", severity,
                    {"p_model": round(p_model, 4), "confidence": round(confidence, 3)},
                ))

    # Model disagreement vs market, classified by gap
    if p_model is not None and p_market is not None:
        market_correct = (p_market > 0.5 and yes_won == 1) or (p_market < 0.5 and yes_won == 0)
        model_correct = (p_model > 0.5 and yes_won == 1) or (p_model < 0.5 and yes_won == 0)
        gap = abs(p_model - p_market)
        if market_correct and not model_correct:
            severity = float(_ll(p_model, yes_won) - _ll(p_market, yes_won))
            if gap >= 0.30:
                out.append(FailureRecord(
                    "model_loud_wrong_market_right", severity,
                    {"p_model": round(p_model, 4), "p_market": round(p_market, 4), "gap": round(gap, 3)},
                ))
            elif gap >= 0.10:
                out.append(FailureRecord(
                    "model_disagrees_market_right", severity,
                    {"p_model": round(p_model, 4), "p_market": round(p_market, 4), "gap": round(gap, 3)},
                ))

    # Trade-side failure: we actually fired a position and it lost money
    if notional_traded > 0 and realized_pnl < 0:
        # severity = loss as a fraction of notional, clipped to [0, 1]
        severity = float(min(1.0, abs(realized_pnl) / max(notional_traded, 1e-6)))
        out.append(FailureRecord(
            "combined_wrong_traded", severity,
            {"notional_traded": round(notional_traded, 2),
             "realized_pnl": round(realized_pnl, 2)},
        ))

    return out


# ── Persistence ─────────────────────────────────────────────────────────
def record_failures(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    resolved_ts: float,
    yes_won: int,
    p_model: float | None,
    p_market: float | None,
    category: str | None,
    question: str | None,
    notional_traded: float = 0.0,
    realized_pnl: float = 0.0,
) -> int:
    """Classify and persist failures for one (condition_id, settlement) pair.

    Returns the number of failure rows written.
    """
    ensure_table(conn)
    failures = classify(
        p_model=p_model, p_market=p_market, yes_won=int(yes_won),
        notional_traded=notional_traded, realized_pnl=realized_pnl,
    )
    if not failures:
        return 0
    ll_model = _ll(p_model, int(yes_won)) if p_model is not None else None
    ll_market = _ll(p_market, int(yes_won)) if p_market is not None else None
    written = 0
    for f in failures:
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO model_failures
                   (condition_id, resolved_ts, yes_won, p_model, p_market,
                    log_loss_model, log_loss_market, failure_type, severity,
                    category, question, notional_traded, realized_pnl,
                    detail, created_ts)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    condition_id, resolved_ts, int(yes_won),
                    p_model, p_market,
                    ll_model, ll_market,
                    f.failure_type, f.severity,
                    category, question, notional_traded, realized_pnl,
                    json.dumps(f.detail), time.time(),
                ),
            )
            # cursor.rowcount is per-statement: 1 = insert succeeded,
            # 0 = unique-index collision (INSERT OR IGNORE no-op).
            if cur.rowcount > 0:
                written += 1
        except sqlite3.OperationalError as e:
            log.warning("failure_insert_error", err=str(e))
    conn.commit()
    return written


# ── Backfill from signal_outcomes ───────────────────────────────────────
def backfill_failures(db_path: str, *, batch_size: int = 500) -> dict:
    """Walk signal_outcomes + resolutions and populate model_failures.

    Joins to fills/resolutions to compute the trade-side notional+pnl
    column for the `combined_wrong_traded` failure type.
    """
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    rows = conn.execute(
        """SELECT s.condition_id, s.resolved_ts, s.yes_won,
                  s.p_stat_lgbm,
                  COALESCE(s.p_market_24h, s.p_market_6h, s.p_market_1h, s.p_market_pre) AS p_market,
                  s.category, s.question
           FROM signal_outcomes s
           WHERE s.yes_won IS NOT NULL"""
    ).fetchall()
    log.info("failure_backfill_start", n=len(rows))
    written = 0
    skipped = 0
    for i, (cid, ts, yw, p_m, p_mkt, cat, q) in enumerate(rows):
        # Trade-side aggregation: total notional bought + realized P&L on this market
        try:
            row = conn.execute(
                """SELECT
                       COALESCE(SUM(f.notional), 0) AS notional,
                       COALESCE(SUM(((CASE WHEN (f.token_id=r.yes_token_id AND r.yes_won=1)
                                            OR (f.token_id=r.no_token_id AND r.yes_won=0)
                                          THEN 1.0 ELSE 0.0 END) - f.price) * f.size), 0) AS pnl
                   FROM fills f
                   INNER JOIN resolutions r ON r.condition_id = f.condition_id
                   WHERE f.condition_id = ? AND f.side='BUY'""",
                (cid,),
            ).fetchone()
            notional, pnl = float(row[0] or 0.0), float(row[1] or 0.0)
        except Exception:
            notional, pnl = 0.0, 0.0
        n = record_failures(
            conn,
            condition_id=cid,
            resolved_ts=float(ts or 0.0),
            yes_won=int(yw),
            p_model=float(p_m) if p_m is not None else None,
            p_market=float(p_mkt) if p_mkt is not None else None,
            category=cat,
            question=q,
            notional_traded=notional,
            realized_pnl=pnl,
        )
        if n > 0:
            written += n
        else:
            skipped += 1
        if (i + 1) % batch_size == 0:
            log.info("failure_backfill_progress", done=i + 1, of=len(rows), written=written, skipped=skipped)
    conn.close()
    log.info("failure_backfill_done", total=len(rows), written=written, skipped=skipped)
    return {"total": len(rows), "written": written, "skipped": skipped}
