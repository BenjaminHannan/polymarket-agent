"""Exit-policy redesign per §10 of the design doc.

Replaces the fixed-percentage stop-loss with a two-track policy:

  1. Kaminski-Lo gated price stop. Kaminski & Lo (2014, "When Do
     Stop-Loss Rules Stop Losses?", J. Financial Markets) prove that
     stop-loss rules raise Sharpe iff the AR(1) coefficient φ of the
     strategy's returns is at least the daily-frequency Sharpe ratio.
     For φ < SR_daily the stop bleeds mean without proportionally
     cutting variance. Polyagent's daily Sharpe is empirically near
     zero and resolved-trade returns are plausibly mean-reverting
     (φ negative), so the canonical 40% / 70% stop has been a mean-
     drag generator. The gate disables price stops unless the
     measured φ exceeds the measured SR_daily.

  2. Near-resolution lock-in. The "early resolution of uncertainty"
     principle: when our position is materially in profit and time-
     to-resolution is small, the residual variance from holding to
     resolution is no longer compensated by expected return. Close
     to lock in the gain. (E.g., we hold YES at $0.30, mid is now
     $0.92, resolution is in 3 hours: closing at $0.92 captures
     ~$0.62 of the maximum-possible $0.70.)

The Kaminski-Lo statistics are computed nightly from the resolutions
table and cached on the `KaminskiLoStopGate`. The stop-loss loop
consults the gate on every check.

Note: the stop-loss DELEVERAGE branch (the existing 70% threshold for
under-$0.10 tokens) is preserved unconditionally — it's not a Sharpe
question, it's a "we made an obvious mistake on a longshot, exit"
question. Kaminski-Lo addresses the *normal* range of stops.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np
import structlog

from polyagent.config import settings

log = structlog.get_logger()


def ar1_coefficient(returns: np.ndarray) -> float:
    """OLS estimate of φ in r_t = φ · r_{t-1} + ε_t. Returns NaN if
    fewer than 5 returns or zero variance."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size < 5:
        return float("nan")
    x = arr[:-1]
    y = arr[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    # Centered OLS slope
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float((xm * xm).sum())
    if denom == 0:
        return float("nan")
    return float((xm * ym).sum() / denom)


def daily_sharpe(returns: np.ndarray, ts_seconds: np.ndarray) -> float:
    """Aggregate per-trade returns into per-day buckets, return SR.
    Falls back to per-trade SR if there are fewer than 5 days."""
    arr = np.asarray(returns, dtype=float).ravel()
    ts = np.asarray(ts_seconds, dtype=float).ravel()
    if arr.size < 5 or ts.size != arr.size:
        return float("nan")
    days = (ts // 86400).astype(int)
    unique_days = np.unique(days)
    if unique_days.size < 5:
        # not enough days; fall back to per-trade SR (still informative)
        sd = float(np.std(arr, ddof=1))
        return float(np.mean(arr) / sd) if sd > 0 else float("nan")
    daily = np.array([arr[days == d].sum() for d in unique_days])
    sd = float(np.std(daily, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(daily) / sd)


def fetch_resolved_returns(db_path: str = settings.db_path) -> tuple[np.ndarray, np.ndarray]:
    """Pull (returns, ts) for resolved positions where we held.
    Sorted by resolved_ts ascending."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        rows = conn.execute(
            """SELECT resolved_ts, pnl,
                      (yes_size * yes_avg_cost + no_size * no_avg_cost) AS entry
               FROM resolutions
               WHERE (yes_size > 0 OR no_size > 0)
               ORDER BY resolved_ts ASC"""
        ).fetchall()
    except sqlite3.OperationalError:
        return np.zeros(0), np.zeros(0)
    finally:
        conn.close()
    rs: list[float] = []
    ts: list[float] = []
    for resolved_ts, pnl, entry in rows:
        try:
            e = float(entry or 0)
            if e < 1e-6:
                continue
            rs.append(float(pnl or 0) / e)
            ts.append(float(resolved_ts or 0))
        except (TypeError, ValueError):
            continue
    return np.asarray(rs, dtype=float), np.asarray(ts, dtype=float)


@dataclass
class KaminskiLoStopGate:
    """Stateful gate: should we apply a price-based stop right now?

    Refreshes ``phi`` and ``sr_daily`` on demand. The result of the gate
    is cached for ``cache_sec`` so we don't recompute on every fill.
    """
    db_path: str = settings.db_path
    cache_sec: float = 3600.0       # recompute hourly
    min_returns_for_estimate: int = 30

    phi: float | None = None
    sr_daily: float | None = None
    n_used: int = 0
    last_refresh_ts: float = 0.0

    def refresh(self) -> dict:
        rs, ts = fetch_resolved_returns(self.db_path)
        if rs.size < self.min_returns_for_estimate:
            self.phi = None
            self.sr_daily = None
            self.n_used = int(rs.size)
            self.last_refresh_ts = time.time()
            return {
                "phi": None,
                "sr_daily": None,
                "n_used": self.n_used,
                "stops_enabled": True,  # default to enabled when we don't know
                "note": "insufficient_resolved_returns",
            }
        phi = ar1_coefficient(rs)
        sr_d = daily_sharpe(rs, ts)
        self.phi = phi
        self.sr_daily = sr_d
        self.n_used = int(rs.size)
        self.last_refresh_ts = time.time()
        return self.summary()

    def stops_enabled(self) -> bool:
        """Per Kaminski-Lo 2014: stops raise Sharpe iff φ ≥ SR_daily.
        When either is unknown (insufficient data), default ENABLED so
        we keep the existing safety net while statistics accumulate."""
        if self.phi is None or self.sr_daily is None:
            return True
        if np.isnan(self.phi) or np.isnan(self.sr_daily):
            return True
        return self.phi >= self.sr_daily

    def maybe_refresh(self) -> None:
        if time.time() - self.last_refresh_ts > self.cache_sec:
            try:
                self.refresh()
                log.info("kaminski_lo_refresh", **self.summary())
            except Exception as e:
                log.warning("kaminski_lo_refresh_error", err=str(e))

    def summary(self) -> dict:
        return {
            "phi": round(self.phi, 4) if self.phi is not None and not np.isnan(self.phi) else None,
            "sr_daily": round(self.sr_daily, 4) if self.sr_daily is not None and not np.isnan(self.sr_daily) else None,
            "n_used": self.n_used,
            "stops_enabled": self.stops_enabled(),
        }


@dataclass
class NearResolutionLockIn:
    """Close materially-in-profit positions when time-to-resolution is
    short. The residual variance of holding to resolution is small but
    non-zero; lock in the gain.
    """
    min_unrealized_pct: float = 0.50   # mid moved this far in our favor relative to entry
    max_hours_to_resolution: float = 6.0
    min_lock_value_usd: float = 5.0     # ignore tiny lock-ins

    def should_exit(
        self,
        avg_cost: float,
        size: float,
        bid: float,
        hours_to_resolution: float | None,
    ) -> tuple[bool, str | None]:
        if avg_cost <= 0 or size <= 0 or bid <= 0:
            return False, None
        unreal_pct = (bid - avg_cost) / avg_cost
        if unreal_pct < self.min_unrealized_pct:
            return False, None
        if hours_to_resolution is not None and hours_to_resolution > self.max_hours_to_resolution:
            return False, None
        lock_value = (bid - avg_cost) * size
        if lock_value < self.min_lock_value_usd:
            return False, None
        return True, (
            f"near_resolution_lock_in unreal_pct={unreal_pct*100:.1f}% "
            f"ttr_h={hours_to_resolution:.1f} value=${lock_value:.2f}"
            if hours_to_resolution is not None
            else f"near_resolution_lock_in unreal_pct={unreal_pct*100:.1f}% value=${lock_value:.2f}"
        )
