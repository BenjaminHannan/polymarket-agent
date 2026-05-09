"""Live Sharpe-honesty harness.

Runs nightly (default every 24h) and on-demand. Pulls realized
per-trade returns from ``resolutions`` joined to ``fills``, computes
PSR / DSR / MTRL via :mod:`sharpe_honesty`, persists a row in
``sharpe_history``, and logs a structured ``sharpe_report`` event.

Also exposes a *strategy-certificate gate*: any new strategy variant
registered through :class:`StrategyCertRegistry` must clear
``DSR > min_dsr`` on its CPCV holdout before its enable flag is set
to True. The traders consult ``registry.is_enabled(name)`` before
admitting a signal — making the gate live, not advisory.

This module is the single most important deliverable of Week 1: it is
how we *know* whether anything else is helping.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from polyagent.config import settings
from polyagent.eval.sharpe_honesty import HonestyReport, report

log = structlog.get_logger()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sharpe_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    window_label TEXT NOT NULL,
    n_returns INTEGER NOT NULL,
    mean_r REAL,
    std_r REAL,
    skew REAL,
    excess_kurt REAL,
    sr REAL,
    psr_zero REAL,
    psr_skill REAL,
    dsr REAL,
    mtrl_zero_95 INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS sharpe_history_ts ON sharpe_history(ts);

CREATE TABLE IF NOT EXISTS strategy_certificates (
    name TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    dsr_holdout REAL,
    n_holdout INTEGER,
    issued_ts REAL,
    detail TEXT
);
"""


@dataclass
class StrategyCertRegistry:
    """Single source of truth for which strategy variants may trade live.

    A variant must call :meth:`register` with a CPCV holdout DSR; if
    DSR ≥ ``min_dsr`` (default 0.5), the certificate's ``enabled``
    column is set. Traders consult :meth:`is_enabled` on every signal.

    Variants without a registered certificate fall back to
    ``default_enabled`` so we don't accidentally break the existing
    bot — this is a *new* discipline layer that gates *new* variants
    introduced after the harness ships.
    """
    db_path: str = settings.db_path
    min_dsr: float = 0.5
    default_enabled: bool = True

    _cache: dict[str, bool] = field(default_factory=dict)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def is_enabled(self, name: str) -> bool:
        if name in self._cache:
            return self._cache[name]
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT enabled FROM strategy_certificates WHERE name = ?",
                (name,),
            ).fetchone()
        except sqlite3.OperationalError:
            # Table may not exist yet (fresh DB). Default-enable.
            self._cache[name] = self.default_enabled
            return self.default_enabled
        finally:
            conn.close()
        ok = bool(row[0]) if row else self.default_enabled
        self._cache[name] = ok
        return ok

    def register(
        self,
        name: str,
        dsr_holdout: float,
        n_holdout: int,
        detail: dict | None = None,
    ) -> bool:
        """Register a variant's CPCV-holdout DSR. Sets enabled=1 iff
        DSR >= min_dsr. Returns the resulting enabled flag."""
        enabled = 1 if dsr_holdout >= self.min_dsr else 0
        conn = self._conn()
        try:
            conn.execute(SCHEMA)
            conn.execute(
                "INSERT INTO strategy_certificates(name, enabled, dsr_holdout, n_holdout, issued_ts, detail) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled, "
                "dsr_holdout=excluded.dsr_holdout, n_holdout=excluded.n_holdout, "
                "issued_ts=excluded.issued_ts, detail=excluded.detail",
                (name, enabled, dsr_holdout, n_holdout, time.time(), json.dumps(detail or {})),
            )
            conn.commit()
        finally:
            conn.close()
        self._cache[name] = bool(enabled)
        log.info(
            "strategy_certificate_registered",
            name=name,
            dsr=round(dsr_holdout, 4),
            n=n_holdout,
            enabled=bool(enabled),
            min_dsr=self.min_dsr,
        )
        return bool(enabled)


def _ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _fetch_returns(
    db_path: str,
    since_ts: float | None = None,
) -> np.ndarray:
    """Realized per-trade returns from ``resolutions`` table.

    For each resolution row, return = pnl / max(1e-6, entry_notional)
    where entry_notional = yes_size * yes_avg_cost + no_size * no_avg_cost.

    Filters out rows with zero entry notional (we never bought) — those
    are pure observation rows, not trades.
    """
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        sql = (
            "SELECT pnl, yes_size, no_size, yes_avg_cost, no_avg_cost "
            "FROM resolutions"
        )
        args: tuple = ()
        if since_ts is not None:
            sql += " WHERE resolved_ts >= ?"
            args = (since_ts,)
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return np.zeros(0)
    finally:
        conn.close()
    out: list[float] = []
    for pnl, ys, ns, yc, nc in rows:
        try:
            entry = float(ys or 0) * float(yc or 0) + float(ns or 0) * float(nc or 0)
            if entry < 1e-6:
                continue
            out.append(float(pnl) / entry)
        except (TypeError, ValueError):
            continue
    return np.asarray(out, dtype=float)


def _persist(db_path: str, window_label: str, r: HonestyReport) -> None:
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        conn.execute(
            "INSERT INTO sharpe_history(ts, window_label, n_returns, mean_r, std_r, "
            "skew, excess_kurt, sr, psr_zero, psr_skill, dsr, mtrl_zero_95, detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                window_label,
                r.n,
                r.mean_r,
                r.std_r,
                r.skew,
                r.excess_kurt,
                r.sr,
                r.psr_zero,
                r.psr_skill,
                r.dsr,
                r.mtrl_zero_95,
                json.dumps(r.to_dict()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


@dataclass
class SharpeHarness:
    db_path: str = settings.db_path
    interval_sec: float = 86400.0     # nightly
    min_returns: int = 10
    sr_skill_benchmark: float = 0.5

    def refresh(self) -> dict:
        _ensure_schema(self.db_path)
        out: dict[str, dict] = {}
        windows = (
            ("1d", 86400.0),
            ("7d", 7 * 86400.0),
            ("30d", 30 * 86400.0),
            ("all", None),
        )
        now = time.time()
        for label, lookback in windows:
            since = now - lookback if lookback is not None else None
            rs = _fetch_returns(self.db_path, since)
            if rs.size < self.min_returns:
                out[label] = {"n": int(rs.size), "skipped": True}
                continue
            rep = report(rs, sr_skill_benchmark=self.sr_skill_benchmark)
            _persist(self.db_path, label, rep)
            out[label] = rep.to_dict()
        log.info("sharpe_report", **{k: v for k, v in out.items()})
        return out

    async def run(self) -> None:
        log.info("sharpe_harness_start", interval_sec=self.interval_sec)
        # First run after a small delay so the bot has time to settle in.
        await asyncio.sleep(60)
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception as e:
                log.warning("sharpe_harness_error", err=str(e))
            await asyncio.sleep(self.interval_sec)
