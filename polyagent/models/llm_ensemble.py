"""Multi-LLM probability ensemble (Schoenegger 2024 *Science Advances*).

Direct implementation of the doc's Problem-2 fix #7 and Problem-9
fix #4. Schoenegger et al. 2024 showed a 12-LLM median is statistically
indistinguishable from a 925-human crowd on judgmental forecasting.
arXiv 2510.01499 ("Beyond Majority Voting", 2025) improves further by
weighting each LLM by historical accuracy and inter-LLM correlation.

For Polyagent the most useful ensemble is a small set of decorrelated
small/medium models:
  - Qwen3-8B-NVFP4 (general reasoning)
  - gpt-oss-20B    (Phi-4-mini fallback when gpt-oss not loaded)
  - DeepSeek-R1-14B (alternate reasoning style)

These already live in `polyagent/models/llm_forecaster.py` as
single-model adapters. This module composes them.

Aggregation modes
-----------------
1. **simple_median** — pure median; baseline.
2. **simple_mean**   — pure mean; included for ablation.
3. **accuracy_weighted_median** — weight = (1 − historical Brier) per
   model; weighted median.
4. **higher_order_aggregation** — arXiv 2510.01499. Sequentially
   account for accuracy and pairwise correlation between models; falls
   back to accuracy_weighted_median if pairwise stats are absent.

Brier statistics per model are persisted to `llm_brier_history`:
  model_name TEXT PRIMARY KEY
  n_resolved INTEGER
  brier_sum  REAL
  log_loss_sum REAL
  last_updated REAL

so we can update with each new resolution and pick weights at decision
time from a single SELECT.
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class LLMVote:
    model: str
    p_yes: float
    confidence: float = 1.0     # optional; used as floor weight


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS llm_brier_history (
            model_name TEXT PRIMARY KEY,
            n_resolved INTEGER NOT NULL DEFAULT 0,
            brier_sum  REAL NOT NULL DEFAULT 0.0,
            log_loss_sum REAL NOT NULL DEFAULT 0.0,
            last_updated REAL NOT NULL
        )"""
    )
    conn.commit()


def record_resolution(
    conn: sqlite3.Connection,
    model: str,
    p_yes: float,
    outcome_yes: bool,
) -> None:
    """Update Brier statistics for `model` with one resolution.
    Outcome is 1 (YES wins) or 0 (NO wins)."""
    ensure_table(conn)
    y = 1.0 if outcome_yes else 0.0
    p = float(max(1e-6, min(1 - 1e-6, p_yes)))
    brier = (p - y) ** 2
    ll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    now = time.time()
    conn.execute(
        """INSERT INTO llm_brier_history (model_name, n_resolved, brier_sum, log_loss_sum, last_updated)
           VALUES (?, 1, ?, ?, ?)
           ON CONFLICT(model_name) DO UPDATE SET
              n_resolved = n_resolved + 1,
              brier_sum = brier_sum + excluded.brier_sum,
              log_loss_sum = log_loss_sum + excluded.log_loss_sum,
              last_updated = excluded.last_updated""",
        (model, brier, ll, now),
    )
    conn.commit()


def _brier_weights(
    conn: sqlite3.Connection,
    models: list[str],
    *,
    min_resolved: int = 20,
) -> dict[str, float]:
    """Weight = max(0.01, 1 − avg_brier). Models without enough data
    get an equal-weight floor of 1.0 (so we don't silently drop a new
    model on its first deployment)."""
    ensure_table(conn)
    weights: dict[str, float] = {}
    for m in models:
        row = conn.execute(
            "SELECT n_resolved, brier_sum FROM llm_brier_history WHERE model_name=?",
            (m,),
        ).fetchone()
        if row is None or row[0] < min_resolved:
            weights[m] = 1.0
            continue
        avg_brier = row[1] / max(1, row[0])
        weights[m] = max(0.01, 1.0 - float(avg_brier))
    # Normalize
    total = sum(weights.values())
    if total <= 0:
        return {m: 1.0 / len(models) for m in models}
    return {m: w / total for m, w in weights.items()}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: the value at which the cumulative weight crosses 0.5."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = int(np.searchsorted(cum, 0.5))
    idx = max(0, min(len(v_sorted) - 1, idx))
    return float(v_sorted[idx])


def aggregate(
    votes: list[LLMVote],
    *,
    mode: str = "accuracy_weighted_median",
    conn: sqlite3.Connection | None = None,
    higher_order_correlation: float | None = None,
) -> float:
    """Aggregate a list of LLMVote into a single P(YES) in [0, 1].

    `mode` ∈ {simple_mean, simple_median, accuracy_weighted_median,
    higher_order_aggregation}. The last requires `conn` to read Brier
    stats; otherwise falls back to accuracy_weighted_median with
    uniform weights.
    """
    if not votes:
        return 0.5
    probs = np.array([float(v.p_yes) for v in votes], dtype=float)
    probs = np.clip(probs, 1e-4, 1 - 1e-4)

    if mode == "simple_mean":
        return float(np.mean(probs))
    if mode == "simple_median":
        return float(np.median(probs))

    if conn is None:
        # Without history, treat all models equally.
        w = np.ones_like(probs)
    else:
        wm = _brier_weights(conn, [v.model for v in votes])
        w = np.array([wm.get(v.model, 1.0) for v in votes], dtype=float)
    if mode == "accuracy_weighted_median":
        return _weighted_median(probs, w)

    if mode == "higher_order_aggregation":
        # arXiv 2510.01499 lite: shrink each model's prob toward the
        # weighted mean by an amount proportional to pairwise correlation.
        # Without an estimated pairwise correlation matrix we apply a
        # uniform `higher_order_correlation` shrinkage as a placeholder.
        wmean = float(np.average(probs, weights=w))
        rho = float(higher_order_correlation if higher_order_correlation is not None else 0.3)
        shrunk = (1 - rho) * probs + rho * wmean
        return _weighted_median(shrunk, w)

    raise ValueError(f"unknown mode: {mode}")


def aggregate_with_log_pool(
    votes: list[LLMVote],
    *,
    market_p: float | None = None,
    market_weight: float = 0.6,
    conn: sqlite3.Connection | None = None,
) -> float:
    """Akey 2026 "shrink-toward-market" prior + ensemble.

    First aggregate the LLM votes via accuracy-weighted median, then
    log-pool with the market price using `market_weight` for the
    market and (1 − market_weight) for the LLM aggregate.

    Implementation matches the existing `combiner.py log_pool` recipe
    so calibration regimes are consistent.
    """
    agg = aggregate(votes, mode="accuracy_weighted_median", conn=conn)
    if market_p is None:
        return agg
    mw = float(max(0.0, min(1.0, market_weight)))
    log_odds = ((1 - mw) * math.log(agg / (1 - agg))
                + mw * math.log(market_p / (1 - market_p)))
    p = 1.0 / (1.0 + math.exp(-log_odds))
    return float(max(1e-4, min(1 - 1e-4, p)))
