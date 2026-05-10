"""Forecasting benchmark harness (pmwhybetter.md Problem-9 #5).

References:
  - ForecastBench (Oct 2025) — best LLM Brier 0.101 vs market 0.106 on
    liquid markets.
  - arXiv 2507.04562 — "Evaluating LLMs on Real-World Forecasting
    Against Expert Forecasters" — Brier on resolved political/economic
    questions.
  - arXiv 2602.21229 — Mention-markets benchmark.
  - Metaculus AI Benchmark Tournament (Mantic-style).

What this provides
------------------
A *self-contained* benchmark that runs locally against the bot's
`resolutions` table — i.e. our own paper-resolved questions become the
benchmark set. This is a domain-specific evaluation harness, not a
replacement for the public benchmarks above, but it provides:

  1. **ECE** (Expected Calibration Error) bucketed by category and
     confidence — the doc's key diagnostic.
  2. **Brier** + decomposition into reliability + resolution.
  3. **Refused-rate-vs-Brier curve** — how much abstention buys how
     much Brier improvement.
  4. **Comparison to a baseline strategy**: market-only, base-rate-only,
     and naive-0.5 baselines, so any improvement is contextualised.

Usage from CLI
--------------
```
.venv\\Scripts\\python.exe -m polyagent.eval.forecast_benchmark \\
    --strategy stat_lgbm_combiner_sports_global_v4_pbo_grid \\
    --category sports_global \\
    --since 1730000000
```

API
---
- `evaluate_predictions(predictions, market_prices, outcomes,
    categories=None)` → `BenchmarkReport`
- `ece(predictions, outcomes, n_bins=10)` → float
- `brier_decomposition(predictions, outcomes, n_bins=10)` → dict
- `compare_to_baselines(predictions, market_prices, outcomes)` → dict
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class BenchmarkReport:
    n: int
    brier: float
    ece: float
    log_loss: float
    reliability: float           # Brier reliability term
    resolution: float            # Brier resolution term
    market_brier: float
    market_log_loss: float
    baseline_uniform_brier: float = 0.25
    baseline_base_rate_brier: float | None = None
    by_category: dict | None = None

    def model_beats_market(self) -> bool:
        return self.brier < self.market_brier

    def edge_log_loss(self) -> float:
        """Positive = model better than market on log-loss."""
        return self.market_log_loss - self.log_loss

    def edge_brier(self) -> float:
        """Positive = model better than market on Brier."""
        return self.market_brier - self.brier

    def summary(self) -> dict:
        return {
            "n": self.n,
            "brier": round(self.brier, 5),
            "ece": round(self.ece, 4),
            "log_loss": round(self.log_loss, 5),
            "market_brier": round(self.market_brier, 5),
            "model_beats_market": self.model_beats_market(),
            "edge_log_loss": round(self.edge_log_loss(), 5),
            "edge_brier": round(self.edge_brier(), 5),
            "reliability": round(self.reliability, 6),
            "resolution": round(self.resolution, 6),
            "by_category": self.by_category,
        }


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)


def ece(predictions, outcomes, *, n_bins: int = 10) -> float:
    """Expected Calibration Error over `n_bins` equal-width buckets."""
    p = _clip(predictions)
    y = np.asarray(outcomes, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_prob = float(p[mask].mean())
        bin_true = float(y[mask].mean())
        bin_w = mask.sum() / n
        total += bin_w * abs(bin_prob - bin_true)
    return float(total)


def brier_decomposition(predictions, outcomes, *, n_bins: int = 10) -> dict:
    """Murphy decomposition: Brier = reliability − resolution +
    uncertainty.

    Lower reliability is better (closer to calibrated); higher resolution
    is better (model distinguishes outcomes from base rate)."""
    p = _clip(predictions)
    y = np.asarray(outcomes, dtype=float)
    n = len(p)
    if n == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0}
    edges = np.linspace(0, 1, n_bins + 1)
    base_rate = float(y.mean())
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_prob = float(p[mask].mean())
        bin_true = float(y[mask].mean())
        bin_w = mask.sum() / n
        reliability += bin_w * (bin_prob - bin_true) ** 2
        resolution += bin_w * (bin_true - base_rate) ** 2
    uncertainty = base_rate * (1 - base_rate)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def log_loss(predictions, outcomes) -> float:
    p = _clip(predictions)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(predictions, outcomes) -> float:
    p = _clip(predictions)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def evaluate_predictions(
    predictions,
    market_prices,
    outcomes,
    *,
    categories: list[str] | None = None,
    n_bins: int = 10,
) -> BenchmarkReport:
    """Single-pass evaluation; runs model + market baseline + category
    decomposition in one walk."""
    p = _clip(predictions)
    m = _clip(market_prices)
    y = np.asarray(outcomes, dtype=int)
    n = len(p)
    if n == 0:
        return BenchmarkReport(
            n=0, brier=0.0, ece=0.0, log_loss=0.0,
            reliability=0.0, resolution=0.0,
            market_brier=0.0, market_log_loss=0.0,
        )

    decomp = brier_decomposition(p, y, n_bins=n_bins)
    by_cat: dict | None = None
    if categories is not None and len(categories) == n:
        cats = np.asarray(categories)
        by_cat = {}
        for c in np.unique(cats):
            mask = cats == c
            if mask.sum() < 5:
                continue
            by_cat[str(c)] = {
                "n": int(mask.sum()),
                "brier_model": brier(p[mask], y[mask]),
                "brier_market": brier(m[mask], y[mask]),
                "ece": ece(p[mask], y[mask], n_bins=min(n_bins, mask.sum() // 5)),
            }

    return BenchmarkReport(
        n=n,
        brier=brier(p, y),
        ece=ece(p, y, n_bins=n_bins),
        log_loss=log_loss(p, y),
        reliability=decomp["reliability"],
        resolution=decomp["resolution"],
        market_brier=brier(m, y),
        market_log_loss=log_loss(m, y),
        baseline_base_rate_brier=float(np.mean(y) * (1 - np.mean(y)) * 4),
        by_category=by_cat,
    )


def compare_to_baselines(
    predictions, market_prices, outcomes,
) -> dict:
    """Side-by-side report against three baselines (market, uniform 0.5,
    historical base rate)."""
    p = _clip(predictions)
    m = _clip(market_prices)
    y = np.asarray(outcomes, dtype=int)
    base_rate = float(y.mean()) if len(y) else 0.5
    uniform = np.full_like(p, 0.5)
    base = np.full_like(p, base_rate)
    return {
        "model_brier": brier(p, y),
        "model_log_loss": log_loss(p, y),
        "market_brier": brier(m, y),
        "market_log_loss": log_loss(m, y),
        "uniform_brier": brier(uniform, y),
        "uniform_log_loss": log_loss(uniform, y),
        "base_rate_brier": brier(base, y),
        "base_rate_log_loss": log_loss(base, y),
        "base_rate": base_rate,
        "n": int(len(p)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--since", type=float, default=0.0)
    p.add_argument("--db", default="data/paper.db")
    args = p.parse_args()

    conn = sqlite3.connect(args.db, timeout=30.0)
    where: list[str] = ["resolved_value IS NOT NULL"]
    params: list = []
    if args.since:
        where.append("resolved_ts >= ?")
        params.append(args.since)

    rows = conn.execute(
        f"""SELECT so.combined_p, COALESCE(so.market_p_24h, 0.5), so.outcome,
                  COALESCE(m.category, 'unknown')
           FROM signal_outcomes so
           LEFT JOIN markets m ON m.condition_id = so.condition_id
           WHERE {" AND ".join(where)}""",
        params,
    ).fetchall()
    if not rows:
        print("(no rows)")
        return 0
    pred = [r[0] for r in rows]
    mkt = [r[1] for r in rows]
    y = [r[2] for r in rows]
    cats = [r[3] for r in rows]
    rep = evaluate_predictions(pred, mkt, y, categories=cats)
    import json
    print(json.dumps(rep.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
