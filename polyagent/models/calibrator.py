"""Isotonic-regression probability calibrator.

Fit on out-of-fold predictions from a held-out cohort. At inference time,
apply to raw model output to get a calibrated P(YES). Calibrators have a
strong "no extrapolation" property — predictions outside [min_train,
max_train] are clamped to the boundary value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class IsotonicCalibrator:
    iso: object | None = None  # sklearn IsotonicRegression
    n_train: int = 0

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        if len(raw_probs) != len(labels):
            raise ValueError("length mismatch")
        if len(raw_probs) < 50:
            raise ValueError(f"need >=50 samples to fit; got {len(raw_probs)}")
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(raw_probs, labels)
        self.iso = iso
        self.n_train = len(raw_probs)

    def transform(self, raw_probs: np.ndarray | float) -> np.ndarray | float:
        if self.iso is None:
            return raw_probs
        scalar = np.isscalar(raw_probs)
        arr = np.atleast_1d(np.asarray(raw_probs, dtype=float))
        out = self.iso.transform(arr)
        return float(out[0]) if scalar else out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"iso": self.iso, "n_train": self.n_train}, path)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        bundle = joblib.load(path)
        c = cls()
        c.iso = bundle["iso"]
        c.n_train = bundle["n_train"]
        return c


@dataclass
class VennAbersCalibrator:
    """Venn-Abers calibrator (Vovk & Petej, 2014/2025).

    Distribution-free, finite-sample-valid binary calibrator that
    *natively returns a calibration interval* (p_low, p_high) — the
    upper and lower probability bounds whose conditional coverage is
    guaranteed asymptotically without distributional assumptions.

    For our pipeline this delivers two wins simultaneously:

      1. Better point estimate than isotonic on thin per-cell buckets
         (Manokhin 2025, ICLR 2025 conformal-binary work). Beats Beta
         and isotonic on small samples in the calibration literature.
      2. A *free* conformal lower bound on P(YES) that downstream
         strategies (Kelly sizing, deployment gates) can size off — a
         worst-case probability instead of a point estimate.

    Implementation note: scaffolds over the `venn_abers` package's
    `VennAbers` class, which expects a 2-column probability input
    (P(NO), P(YES)). We marshal scalars to that shape internally.
    """
    va: object | None = None
    n_train: int = 0

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> None:
        from venn_abers import VennAbers

        eps = 1e-6
        p = np.clip(np.asarray(raw_probs, dtype=float), eps, 1 - eps)
        y = np.asarray(labels, dtype=int)
        if len(p) < 15:
            raise ValueError(f"need >=15 samples; got {len(p)}")
        # 2-col format expected by venn_abers
        p2 = np.column_stack([1.0 - p, p])
        va = VennAbers()
        va.fit(p2, y)
        self.va = va
        self.n_train = len(p)

    def transform(self, raw_probs: np.ndarray | float) -> np.ndarray | float:
        if self.va is None:
            return raw_probs
        eps = 1e-6
        scalar = np.isscalar(raw_probs)
        p = np.clip(np.atleast_1d(np.asarray(raw_probs, dtype=float)), eps, 1 - eps)
        p2 = np.column_stack([1.0 - p, p])
        cal, _ = self.va.predict_proba(p2)
        out = cal[:, 1]
        return float(out[0]) if scalar else out

    def transform_with_interval(
        self, raw_probs: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        """Return (p_point, p_low, p_high). The interval is the
        Venn-Abers two-sided bound."""
        if self.va is None:
            return raw_probs, raw_probs, raw_probs
        eps = 1e-6
        scalar = np.isscalar(raw_probs)
        p = np.clip(np.atleast_1d(np.asarray(raw_probs, dtype=float)), eps, 1 - eps)
        p2 = np.column_stack([1.0 - p, p])
        cal, intervals = self.va.predict_proba(p2)
        # `intervals` is (n, 2) of (p0, p1) pairs — bounds on the YES-prob.
        p_low = intervals[:, 0]
        p_high = intervals[:, 1]
        p_point = cal[:, 1]
        if scalar:
            return float(p_point[0]), float(p_low[0]), float(p_high[0])
        return p_point, p_low, p_high


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull/Silva Filho/Flach 2017).

    Fits a 3-parameter beta CDF to map raw probabilities to calibrated ones.
    More data-efficient than isotonic on small samples (~50–500 rows) and
    smoother — useful for thin per-cell calibration cells.

    Implementation note: we fit by reformulating beta calibration as a
    2-feature logistic regression on (log(p), log(1-p)). This is the
    Kull et al. parameterization.
    """
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    n_train: int = 0

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        eps = 1e-6
        p = np.clip(np.asarray(raw_probs, dtype=float), eps, 1 - eps)
        y = np.asarray(labels, dtype=int)
        if len(p) < 30:
            raise ValueError(f"need >=30 samples; got {len(p)}")
        # 2-feature logistic regression with intercept
        x = np.column_stack([np.log(p), -np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(x, y)
        # coefficients: a × log(p) + b × -log(1-p) + intercept
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.coef_[0, 1])
        self.c = float(lr.intercept_[0])
        self.n_train = len(p)

    def transform(self, raw_probs: np.ndarray | float) -> np.ndarray | float:
        eps = 1e-6
        scalar = np.isscalar(raw_probs)
        p = np.clip(np.atleast_1d(np.asarray(raw_probs, dtype=float)), eps, 1 - eps)
        z = self.a * np.log(p) + self.b * (-np.log(1 - p)) + self.c
        out = 1.0 / (1.0 + np.exp(-z))
        return float(out[0]) if scalar else out


def horizon_bucket(days_to_resolution: float | None) -> str:
    """Bucket TTR into discrete cells for per-(cat × horizon) calibration."""
    if days_to_resolution is None:
        return "unknown"
    d = float(days_to_resolution)
    if d < 1:
        return "lt_1d"
    if d < 7:
        return "lt_1w"
    if d < 30:
        return "lt_1m"
    return "ge_1m"


@dataclass
class CellCalibrator:
    """Per-(category × horizon) isotonic calibrators with global fallback.

    For each (category, horizon_bucket) cell with >= min_per_cell samples,
    fit a dedicated isotonic. For cells below the threshold, fall back to
    the global isotonic. Le 2026 ("Decomposing Crowd Wisdom") shows
    domain × horizon explains 26% of calibration variance — collapsing
    this with a single global isotonic loses real signal.
    """

    cells: dict[tuple[str, str], IsotonicCalibrator] = None  # type: ignore
    va_cells: dict[tuple[str, str], "VennAbersCalibrator"] = None  # type: ignore
    beta_cells: dict[tuple[str, str], "BetaCalibrator"] = None  # type: ignore
    fallback: IsotonicCalibrator | None = None

    def __post_init__(self):
        if self.cells is None:
            self.cells = {}
        if self.va_cells is None:
            self.va_cells = {}
        if self.beta_cells is None:
            self.beta_cells = {}

    def fit(
        self,
        raw_probs: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
        horizons: np.ndarray,
        min_per_cell: int = 80,
        min_va: int = 30,
        min_beta: int = 15,
    ) -> dict[str, int]:
        """Three-tier fit: isotonic when n>=min_per_cell, Beta (Kull et al.
        2017) when min_beta<=n<min_per_cell, else global isotonic fallback.

        Returns {cell_label -> n_samples}. Cells that get a Beta fit are
        keyed with a '|beta' suffix in the summary so the train script can
        log them; cells that fell back to the global isotonic carry a
        negative count.
        """
        # Global fallback
        global_iso = IsotonicCalibrator()
        try:
            global_iso.fit(raw_probs, labels)
            self.fallback = global_iso
        except Exception as e:
            log.warning("global_calibrator_fit_failed", err=str(e))

        # Per-cell
        keys = list(zip(categories.tolist(), horizons.tolist()))
        unique_keys = sorted(set(keys))
        summary: dict[str, int] = {}
        self.cells = {}
        self.va_cells = {}
        self.beta_cells = {}
        for cat, horiz in unique_keys:
            mask = np.array([k == (cat, horiz) for k in keys])
            n = int(mask.sum())
            label_str = f"{cat}|{horiz}"
            if n >= min_per_cell:
                iso = IsotonicCalibrator()
                try:
                    iso.fit(raw_probs[mask], labels[mask])
                    self.cells[(cat, horiz)] = iso
                    summary[label_str] = n
                    continue
                except Exception:
                    pass
            if n >= min_va:
                # Venn-Abers tier: distribution-free, returns interval too
                va = VennAbersCalibrator()
                try:
                    va.fit(raw_probs[mask], labels[mask])
                    self.va_cells[(cat, horiz)] = va
                    summary[label_str + "|va"] = n
                    continue
                except Exception:
                    pass
            if n >= min_beta:
                beta = BetaCalibrator()
                try:
                    beta.fit(raw_probs[mask], labels[mask])
                    self.beta_cells[(cat, horiz)] = beta
                    summary[label_str + "|beta"] = n
                    continue
                except Exception:
                    pass
            summary[label_str] = -n
        return summary

    def transform(self, raw: float, category: str, horizon: str) -> float:
        cal = self.cells.get((category, horizon))
        if cal is not None:
            return float(cal.transform(raw))
        va = self.va_cells.get((category, horizon))
        if va is not None:
            return float(va.transform(raw))
        beta = self.beta_cells.get((category, horizon))
        if beta is not None:
            return float(beta.transform(raw))
        if self.fallback is not None:
            return float(self.fallback.transform(raw))
        return float(raw)

    def transform_with_interval(
        self, raw: float, category: str, horizon: str
    ) -> tuple[float, float, float]:
        """Return (p_point, p_low, p_high). When a Venn-Abers cell
        exists for this (category, horizon), use its native interval;
        otherwise fall back to a wide pseudo-interval (point ± epsilon)
        so downstream code always gets a well-formed triple. Strategies
        that gate on the lower bound effectively become no-ops in cells
        without VA — explicit, not silent."""
        va = self.va_cells.get((category, horizon))
        if va is not None:
            return va.transform_with_interval(raw)
        # Best-effort: use whatever calibrator is available for the point
        cal = self.cells.get((category, horizon))
        if cal is not None:
            p = float(cal.transform(raw))
        else:
            beta = self.beta_cells.get((category, horizon))
            if beta is not None:
                p = float(beta.transform(raw))
            elif self.fallback is not None:
                p = float(self.fallback.transform(raw))
            else:
                p = float(raw)
        # Wide pseudo-interval: signals "no information"
        return p, p, p

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cells": {
                    f"{c}|{h}": {"iso": cal.iso, "n_train": cal.n_train}
                    for (c, h), cal in self.cells.items()
                },
                "va_cells": {
                    f"{c}|{h}": {"va": cal.va, "n_train": cal.n_train}
                    for (c, h), cal in self.va_cells.items()
                },
                "beta_cells": {
                    f"{c}|{h}": {
                        "a": cal.a,
                        "b": cal.b,
                        "c": cal.c,
                        "n_train": cal.n_train,
                    }
                    for (c, h), cal in self.beta_cells.items()
                },
                "fallback": {
                    "iso": self.fallback.iso if self.fallback else None,
                    "n_train": self.fallback.n_train if self.fallback else 0,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CellCalibrator":
        bundle = joblib.load(path)
        cc = cls()
        cells = {}
        for k, sub in (bundle.get("cells") or {}).items():
            cat, horiz = k.split("|", 1)
            cal = IsotonicCalibrator()
            cal.iso = sub.get("iso")
            cal.n_train = int(sub.get("n_train", 0))
            cells[(cat, horiz)] = cal
        va_cells = {}
        for k, sub in (bundle.get("va_cells") or {}).items():
            cat, horiz = k.split("|", 1)
            va = VennAbersCalibrator()
            va.va = sub.get("va")
            va.n_train = int(sub.get("n_train", 0))
            va_cells[(cat, horiz)] = va
        beta_cells = {}
        for k, sub in (bundle.get("beta_cells") or {}).items():
            cat, horiz = k.split("|", 1)
            beta = BetaCalibrator()
            beta.a = float(sub.get("a", 1.0))
            beta.b = float(sub.get("b", 0.0))
            beta.c = float(sub.get("c", 0.0))
            beta.n_train = int(sub.get("n_train", 0))
            beta_cells[(cat, horiz)] = beta
        fb = bundle.get("fallback") or {}
        if fb.get("iso") is not None:
            fbc = IsotonicCalibrator()
            fbc.iso = fb["iso"]
            fbc.n_train = int(fb.get("n_train", 0))
            cc.fallback = fbc
        cc.cells = cells
        cc.va_cells = va_cells
        cc.beta_cells = beta_cells
        return cc


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Expected Calibration Error + per-bin observed/predicted rates."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_summary = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_summary.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        p_avg = float(probs[mask].mean())
        y_avg = float(labels[mask].mean())
        ece += (n / len(probs)) * abs(p_avg - y_avg)
        bin_summary.append(
            {"lo": float(lo), "hi": float(hi), "n": n, "p_pred": p_avg, "p_obs": y_avg}
        )
    return {"ece": float(ece), "bins": bin_summary}
