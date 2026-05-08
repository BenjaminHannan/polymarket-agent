"""Live calibration drift monitor on rolling resolved markets.

Complements `PSIMonitor` (which compares the *predicted* distribution to
the historical training distribution). This monitor compares the
*calibrated* probability to the *realized* outcome on rows that have
actually resolved in the live regime (last 30 days), and flags drift
when live ECE diverges materially from historical ECE.

Inputs: `signal_outcomes` (one row per resolved market with
`p_stat_lgbm` and `yes_won`).
Outputs: structured `live_ece` log lines and a single dashboard-friendly
attribute `last_summary`. Strategy code can read `last_ece_live` to gate
decisions when calibration drift crosses the action threshold.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

from polyagent.config import settings

log = structlog.get_logger()


def _ece(probs: list[float], labels: list[int], n_bins: int = 10) -> tuple[float, int]:
    """Standard expected calibration error (Naeini et al. 2015). Returns
    (ece, n_used). Returns (nan, 0) if either input is empty."""
    if not probs:
        return float("nan"), 0
    n = len(probs)
    cuts = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = cuts[i], cuts[i + 1]
        idxs = [j for j, p in enumerate(probs) if (p >= lo and p < hi) or (i == n_bins - 1 and p == hi)]
        if not idxs:
            continue
        avg_p = sum(probs[j] for j in idxs) / len(idxs)
        avg_y = sum(labels[j] for j in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(avg_p - avg_y)
    return ece, n


@dataclass
class LiveECEMonitor:
    db_path: str = settings.db_path
    interval_sec: float = 1800.0          # 30 min
    live_window_days: float = 30.0
    historical_window_days: float = 365.0
    drift_warn: float = 0.05              # +0.05 ECE relative to historical = warn
    drift_alert: float = 0.10             # +0.10 = alert (actionable)
    last_summary: dict | None = None
    last_ece_live: float | None = None
    last_ece_hist: float | None = None
    last_drift: float | None = None

    def _fetch(self, since_ts: float | None, until_ts: float | None) -> tuple[list[float], list[int]]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            sql = (
                "SELECT p_stat_lgbm, yes_won FROM signal_outcomes "
                "WHERE p_stat_lgbm IS NOT NULL"
            )
            args: list = []
            if since_ts is not None:
                sql += " AND resolved_ts >= ?"
                args.append(since_ts)
            if until_ts is not None:
                sql += " AND resolved_ts < ?"
                args.append(until_ts)
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        ps: list[float] = []
        ys: list[int] = []
        for p, y in rows:
            if p is None or y is None:
                continue
            try:
                ps.append(float(p))
                ys.append(int(y))
            except (TypeError, ValueError):
                continue
        return ps, ys

    def refresh(self) -> dict:
        now = time.time()
        live_since = now - self.live_window_days * 86400
        hist_until = live_since
        hist_since = now - self.historical_window_days * 86400

        live_p, live_y = self._fetch(live_since, None)
        hist_p, hist_y = self._fetch(hist_since, hist_until)

        if len(live_p) < 50 or len(hist_p) < 200:
            summary = {
                "n_live": len(live_p),
                "n_hist": len(hist_p),
                "ece_live": None,
                "ece_hist": None,
                "drift": None,
                "alert": False,
                "note": "insufficient_rows",
            }
            self.last_summary = summary
            return summary

        ece_live, _ = _ece(live_p, live_y)
        ece_hist, _ = _ece(hist_p, hist_y)
        drift = ece_live - ece_hist

        self.last_ece_live = ece_live
        self.last_ece_hist = ece_hist
        self.last_drift = drift

        summary = {
            "n_live": len(live_p),
            "n_hist": len(hist_p),
            "ece_live": round(ece_live, 4),
            "ece_hist": round(ece_hist, 4),
            "drift": round(drift, 4),
            "alert": drift >= self.drift_alert,
            "warn": drift >= self.drift_warn,
        }
        self.last_summary = summary
        if drift >= self.drift_alert:
            log.warning("live_ece_alert", **summary)
        elif drift >= self.drift_warn:
            log.info("live_ece_warn", **summary)
        else:
            log.info("live_ece", **summary)
        return summary

    async def run(self) -> None:
        log.info("live_ece_start", interval_sec=self.interval_sec)
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception as e:
                log.warning("live_ece_error", err=str(e))
            await asyncio.sleep(self.interval_sec)
