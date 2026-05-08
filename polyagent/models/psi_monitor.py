"""Population Stability Index live calibration drift monitor.

Compares the *recent* paper-fill predicted-probability distribution to
the *historical* training distribution. PSI > 0.10 = noticeable shift,
> 0.25 = strong shift (action required).

Runs as a periodic task. Surfaces a single PSI metric in the dashboard
and a logged warning when drift crosses thresholds.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

from polyagent.config import settings

log = structlog.get_logger()


def psi(p_old: list[float], p_new: list[float], bins: int = 10) -> float:
    """Population Stability Index between two distributions of [0,1] probs."""
    if not p_old or not p_new:
        return 0.0
    cuts = [i / bins for i in range(bins + 1)]
    p_o = [0.0] * bins
    p_n = [0.0] * bins
    for x in p_old:
        b = min(bins - 1, max(0, int(x * bins)))
        p_o[b] += 1
    for x in p_new:
        b = min(bins - 1, max(0, int(x * bins)))
        p_n[b] += 1
    total_o = sum(p_o) or 1
    total_n = sum(p_n) or 1
    out = 0.0
    eps = 1e-6
    for i in range(bins):
        ro = max(eps, p_o[i] / total_o)
        rn = max(eps, p_n[i] / total_n)
        out += (rn - ro) * math.log(rn / ro)
    return out


@dataclass
class PSIMonitor:
    db_path: str = settings.db_path
    interval_sec: float = 1800.0   # 30 min
    warn_threshold: float = 0.10
    alert_threshold: float = 0.25
    last_psi: float | None = None
    last_psi_ts: float = 0.0

    def _historical_probs(self) -> list[float]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            rows = conn.execute(
                "SELECT p_stat_lgbm FROM signal_outcomes WHERE p_stat_lgbm IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        return [float(r[0]) for r in rows]

    def _recent_probs(self) -> list[float]:
        cutoff = time.time() - 7 * 86400
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            rows = conn.execute(
                "SELECT detail FROM signals WHERE strategy = 'stat_lgbm' AND ts >= ? ORDER BY ts DESC LIMIT 5000",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            try:
                d = json.loads(r[0] or "{}")
            except json.JSONDecodeError:
                continue
            p = d.get("p_model_calibrated") or d.get("p_model_raw")
            if p is not None:
                try:
                    out.append(float(p))
                except (TypeError, ValueError):
                    continue
        return out

    def refresh(self) -> dict:
        old = self._historical_probs()
        new = self._recent_probs()
        if len(old) < 100 or len(new) < 100:
            return {"psi": None, "n_old": len(old), "n_new": len(new), "alert": False}
        v = psi(old, new)
        self.last_psi = v
        self.last_psi_ts = time.time()
        if v > self.alert_threshold:
            log.warning("psi_alert", psi=round(v, 3), n_old=len(old), n_new=len(new))
        elif v > self.warn_threshold:
            log.info("psi_warn", psi=round(v, 3))
        return {
            "psi": v,
            "n_old": len(old),
            "n_new": len(new),
            "alert": v > self.alert_threshold,
        }

    async def run(self) -> None:
        log.info("psi_monitor_start", interval_sec=self.interval_sec)
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception as e:
                log.warning("psi_monitor_error", err=str(e))
            await asyncio.sleep(self.interval_sec)
