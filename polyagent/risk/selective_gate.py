"""Selective-classification gate (Chow 1970; El-Yaniv & Wiener 2010;
Geifman & El-Yaniv NIPS 2017; Bai & Jin 2026).

In sparse-edge regimes — exactly Polyagent's setting, with a
question-only model whose Brier (~0.13–0.18) trails the market's
(~0.05–0.10) — the optimal policy is to *abstain* on the noisiest
fraction of signals and only act on high-confidence ones. The
selective error on the top (1−c)·N most-confident predictions is
markedly lower than the unconditional error.

Polyagent already produces a per-prediction Venn-Abers interval
(``calibrated_low``, ``calibrated_high``) on every stat-LGBM call.
The interval *width* is a calibrated proxy for predictive
uncertainty: small width → confident, large width → uncertain. This
gate keeps a rolling buffer of recent widths, computes the
``target_coverage``-quantile, and admits a signal iff its width is
below the quantile.

Per-category coverage is logged hourly. To avoid selective-bias
amplification (Jones et al. ICLR 2021) — where a category gets
*structurally* gated out and never recovers — any category whose
admit-rate falls below ``min_admit_rate`` over its recent window has
its threshold auto-relaxed to ``relaxed_coverage``.

Output of this gate is *observable in week 1*: are we systematically
dropping the noisiest 60% of signals? That's binary and verifiable
without the §12 harness. The harness then measures whether the kept
signals' realized P&L Sharpe is materially better.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class SelectiveGate:
    """Width-based selective abstention.

    Args:
        target_coverage: fraction of candidates to ADMIT (0.40 = take
            the top 40% by interval-width). The quantile threshold is
            ``np.quantile(recent_widths, target_coverage)``.
        burn_in: number of widths to observe before the gate starts
            rejecting. Returns True (admit) for the first ``burn_in``
            calls.
        relaxed_coverage: coverage to use for categories whose
            global admit-rate falls below ``min_admit_rate`` (anti-
            disparity safeguard).
        min_admit_rate: floor; below this, auto-relax for that category.
        per_cat_window: how many recent admit/reject decisions per
            category to evaluate the floor against.
        log_interval_sec: emit a per-category coverage log every N
            seconds.
    """
    target_coverage: float = 0.40
    burn_in: int = 200
    relaxed_coverage: float = 0.70
    min_admit_rate: float = 0.20
    per_cat_window: int = 200
    log_interval_sec: float = 3600.0

    _widths: deque = field(default_factory=lambda: deque(maxlen=2000))
    _per_cat: dict = field(default_factory=dict)  # cat -> deque of bool admits
    _last_log_ts: float = 0.0
    # n_admit / n_total counters since instantiation, for audit
    n_seen: int = 0
    n_admitted: int = 0

    def _coverage_for(self, category: str) -> float:
        """Per-category coverage with anti-disparity floor."""
        d = self._per_cat.get(category)
        if d is None or len(d) < 30:
            return self.target_coverage
        admit_rate = sum(d) / len(d)
        if admit_rate < self.min_admit_rate:
            return self.relaxed_coverage
        return self.target_coverage

    def admit(
        self,
        p_low: float | None,
        p_high: float | None,
        category: str = "_default",
    ) -> bool:
        """Decide whether a candidate signal passes the gate.

        Returns True (admit) if:
          - either bound is None (no Venn-Abers cell — pass through),
          - we're still in burn-in,
          - the width is at or below the per-category quantile.
        """
        self.n_seen += 1
        # Missing interval (cell didn't get enough samples for
        # Venn-Abers): pass through. The point of the gate is to use
        # interval information *when we have it*.
        if p_low is None or p_high is None:
            self.n_admitted += 1
            return True
        try:
            width = float(p_high) - float(p_low)
        except (TypeError, ValueError):
            self.n_admitted += 1
            return True
        if width < 0:
            width = 0.0
        self._widths.append(width)
        if len(self._widths) < self.burn_in:
            self.n_admitted += 1
            self._record(category, True)
            return True
        coverage = self._coverage_for(category)
        threshold = float(np.quantile(self._widths, coverage))
        admit = width <= threshold
        if admit:
            self.n_admitted += 1
        self._record(category, admit)
        self._maybe_log()
        return admit

    def _record(self, category: str, admitted: bool) -> None:
        d = self._per_cat.get(category)
        if d is None:
            d = deque(maxlen=self.per_cat_window)
            self._per_cat[category] = d
        d.append(bool(admitted))

    def _maybe_log(self) -> None:
        now = time.time()
        if now - self._last_log_ts < self.log_interval_sec:
            return
        self._last_log_ts = now
        per_cat: dict[str, dict] = {}
        for cat, d in self._per_cat.items():
            if not d:
                continue
            ar = sum(d) / len(d)
            per_cat[cat] = {
                "n": len(d),
                "admit_rate": round(ar, 3),
                "relaxed": ar < self.min_admit_rate,
            }
        global_rate = (self.n_admitted / max(1, self.n_seen))
        log.info(
            "selective_gate_coverage",
            n_seen=self.n_seen,
            n_admitted=self.n_admitted,
            global_admit_rate=round(global_rate, 3),
            target_coverage=self.target_coverage,
            n_widths=len(self._widths),
            current_threshold=(
                round(float(np.quantile(self._widths, self.target_coverage)), 4)
                if len(self._widths) >= self.burn_in
                else None
            ),
            per_category=per_cat,
        )

    def summary(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "n_admitted": self.n_admitted,
            "global_admit_rate": round(self.n_admitted / max(1, self.n_seen), 3),
            "target_coverage": self.target_coverage,
            "n_widths": len(self._widths),
            "n_categories": len(self._per_cat),
        }
