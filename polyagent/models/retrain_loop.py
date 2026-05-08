"""Closed-loop combiner retraining.

Periodically counts `signal_outcomes` rows where every expert column for the
configured expert list is non-null. When that count grows by `increment` rows
since the last training, calls run_pipeline to retrain and atomically swap
combiner.joblib. The CombinedSignaler picks up the new weights via mtime
detection on its next poll.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import structlog

from polyagent.config import settings
from scripts.train_combiner_per_category import count_full_rows, run_pipeline

log = structlog.get_logger()


@dataclass
class RetrainLoop:
    out_path: str
    experts: list[str]
    horizon: str = "p_market_6h"
    min_rows: int = 150
    increment: int = 50
    check_interval_sec: float = 600.0
    test_frac: float = 0.2
    seed: int = 42
    regression_tolerance: float = 0.0
    allow_regression: bool = False
    forward_holdout_k: int = 50
    last_trained_full_rows: int = 0

    def _read_baseline(self) -> int:
        if not Path(self.out_path).exists():
            return 0
        try:
            bundle = joblib.load(self.out_path)
        except Exception as e:
            log.warning("retrain_baseline_read_error", err=str(e))
            return 0
        n = bundle.get("n_full_rows")
        if n is None:
            return 0
        return int(n)

    def _maybe_retrain(self) -> bool:
        current = count_full_rows(self.experts, settings.db_path)
        baseline = max(self.last_trained_full_rows, self._read_baseline())
        delta = current - baseline
        if current < self.min_rows:
            log.info(
                "retrain_skipped_below_min",
                current_full_rows=current,
                min_rows=self.min_rows,
            )
            return False
        if delta < self.increment:
            log.info(
                "retrain_skipped_no_growth",
                current_full_rows=current,
                last_trained=baseline,
                delta=delta,
                increment=self.increment,
            )
            return False
        log.info(
            "retrain_triggered",
            current_full_rows=current,
            last_trained=baseline,
            delta=delta,
            experts=self.experts,
        )
        bundle = run_pipeline(
            experts=self.experts,
            horizon=self.horizon,
            min_rows=self.min_rows,
            test_frac=self.test_frac,
            seed=self.seed,
            out_path=self.out_path,
            regression_tolerance=self.regression_tolerance,
            allow_regression=self.allow_regression,
            forward_holdout_k=self.forward_holdout_k,
        )
        if bundle is None:
            log.warning("retrain_failed")
            return False
        if bundle.get("regression_blocked"):
            # Quality gate blocked the swap. Don't update baseline so we'll
            # try again next interval (with new data accumulated by then).
            log.warning(
                "retrain_blocked_by_quality_gate",
                old_logloss=bundle["regression_check"].get("old_logloss"),
                new_logloss=bundle["regression_check"].get("new_logloss"),
            )
            return False
        self.last_trained_full_rows = bundle.get("n_full_rows", current)
        log.info(
            "retrain_done",
            n_full_rows=self.last_trained_full_rows,
            n_categories=len(bundle.get("by_category", {})),
        )
        return True

    async def run(self) -> None:
        log.info(
            "retrain_loop_start",
            check_interval_sec=self.check_interval_sec,
            experts=self.experts,
            increment=self.increment,
            min_rows=self.min_rows,
        )
        # Read existing baseline so we don't immediately retrain on startup.
        self.last_trained_full_rows = self._read_baseline()
        while True:
            await asyncio.sleep(self.check_interval_sec)
            try:
                self._maybe_retrain()
            except Exception as e:
                log.warning("retrain_loop_error", err=str(e))
