"""Ternary UP/FLAT/DOWN selective classification (pmwhybetter.md Problem-8 #2).

Implements the "Trading via selective classification" recipe (Coletta
et al., ACM ICAIF 2021; later surveyed in ACM Computing Surveys 2025
doi:10.1145/3727633). Instead of a binary "act or abstain" gate, the
model classifies each signal into three buckets:

  - **UP**:    confident the market price is below true probability →
               buy YES
  - **DOWN**:  confident the market price is above true probability →
               buy NO (or sell YES)
  - **FLAT**:  not confident the gap is real → abstain

The bucket boundaries are calibrated from holdout data so the *realized
hit rate* in each non-FLAT bucket exceeds a configured floor (e.g.
≥60%). Below that floor, the bucket boundary widens (more abstention).

Directly supports our monotonic-Sharpe-with-confidence finding from the
high-confidence-tail analysis (PROJECT.md): we want to *act loudly* on
the top tail and *abstain* on the middle.

How it pairs with existing gates
--------------------------------
- `selective_gate.py` is binary: width-quantile → admit/reject.
- `likelihood_ratio_gate.py` is binary: Heng-Soh RLog → admit/reject.
- `ternary_gate.py` is three-way: directional, with separate edge
  thresholds for the BUY vs SELL sides.

The three gates compose by *short-circuit conjunction*: the trade
proceeds iff all gates admit AND the ternary gate returns UP or DOWN
matching the proposed direction.

API
---
- `TernaryGate.fit(predictions, market_prices, outcomes)` — calibrate
  thresholds from holdout to achieve `min_hit_rate`.
- `TernaryGate.classify(p, market_p)` → "UP" | "DOWN" | "FLAT"
- `TernaryGate.hit_rate_at(threshold)` → empirical hit rate at that
  threshold (diagnostic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class TernaryGate:
    """Three-way selective classifier with side-specific thresholds.

    Args:
        min_hit_rate: per-side realized hit rate floor (default 0.60).
            Thresholds are widened until the OOS bucket meets this.
        min_edge_abs: minimum |edge| (= |p − market_p|) below which
            the gate returns FLAT regardless of hit rate.
        coverage_floor: minimum admit rate across both directions
            before widening stops; prevents the gate from rejecting
            everything.
    """
    min_hit_rate: float = 0.60
    min_edge_abs: float = 0.05
    coverage_floor: float = 0.10

    _up_threshold: float | None = None    # edge above which UP is admitted
    _down_threshold: float | None = None  # edge below which DOWN is admitted
    _fit_n: int = 0
    _fit_hit_rate_up: float | None = None
    _fit_hit_rate_down: float | None = None

    def fit(
        self,
        predictions,
        market_prices,
        outcomes,
    ) -> None:
        """Calibrate thresholds from holdout.

        Args:
            predictions: model P(YES) per row, list[float].
            market_prices: market price at decision time per row.
            outcomes: realized binary outcome (0 or 1) per row.
        """
        p = np.asarray(predictions, dtype=float)
        m = np.asarray(market_prices, dtype=float)
        y = np.asarray(outcomes, dtype=int)
        n = len(p)
        if n < 20:
            log.warning("ternary_gate_too_few_samples", n=n)
            return
        edges = p - m
        # For UP (BUY): correct iff y == 1.
        # For DOWN (SELL): correct iff y == 0.
        # We search a grid of thresholds and pick the *narrowest* that
        # satisfies the hit-rate floor at non-trivial coverage.
        up_edges = edges[edges > 0]
        down_edges = -edges[edges < 0]
        if len(up_edges) > 0:
            best_up = None
            for thr in sorted(up_edges):
                if thr < self.min_edge_abs:
                    continue
                mask = edges >= thr
                if mask.sum() < max(5, n * self.coverage_floor / 2):
                    continue
                hit_rate = float((y[mask] == 1).mean())
                if hit_rate >= self.min_hit_rate:
                    best_up = thr
                    self._fit_hit_rate_up = hit_rate
                    break
            self._up_threshold = best_up

        if len(down_edges) > 0:
            best_down = None
            for thr in sorted(down_edges):
                if thr < self.min_edge_abs:
                    continue
                mask = edges <= -thr
                if mask.sum() < max(5, n * self.coverage_floor / 2):
                    continue
                hit_rate = float((y[mask] == 0).mean())
                if hit_rate >= self.min_hit_rate:
                    best_down = thr
                    self._fit_hit_rate_down = hit_rate
                    break
            self._down_threshold = best_down

        self._fit_n = n
        log.info(
            "ternary_gate_fit_done",
            n=n,
            up_threshold=self._up_threshold,
            down_threshold=self._down_threshold,
            up_hit_rate=self._fit_hit_rate_up,
            down_hit_rate=self._fit_hit_rate_down,
        )

    def classify(self, p: float, market_p: float) -> str:
        """Return "UP" | "DOWN" | "FLAT" for a candidate (p, market_p).

        During burn-in (before fit), defaults to FLAT (most conservative)."""
        if self._up_threshold is None and self._down_threshold is None:
            return "FLAT"
        edge = float(p) - float(market_p)
        if self._up_threshold is not None and edge >= self._up_threshold:
            return "UP"
        if self._down_threshold is not None and edge <= -self._down_threshold:
            return "DOWN"
        return "FLAT"

    def admits(self, p: float, market_p: float, proposed_side: str) -> bool:
        """Helper: True iff the ternary classification matches the
        proposed taker side."""
        cls = self.classify(p, market_p)
        if cls == "FLAT":
            return False
        side = proposed_side.upper()
        return (cls == "UP" and side == "BUY") or (cls == "DOWN" and side == "SELL")

    def summary(self) -> dict:
        return {
            "up_threshold": self._up_threshold,
            "down_threshold": self._down_threshold,
            "fit_n": self._fit_n,
            "fit_hit_rate_up": self._fit_hit_rate_up,
            "fit_hit_rate_down": self._fit_hit_rate_down,
            "min_hit_rate": self.min_hit_rate,
        }
