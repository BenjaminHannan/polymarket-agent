"""Tests for forecasting benchmark harness."""
from __future__ import annotations

import numpy as np

from polyagent.eval.forecast_benchmark import (
    ece, brier, log_loss, brier_decomposition,
    evaluate_predictions, compare_to_baselines,
)


def test_ece_perfect_calibration_zero():
    """Perfectly-calibrated forecasts: ECE = 0."""
    p = [0.1] * 10 + [0.9] * 10
    y = [0] * 9 + [1] * 1 + [1] * 9 + [0] * 1
    # Bin [0.1, 0.2): mean p=0.1, mean y=0.1; bin [0.9, 1.0]: mean p=0.9, mean y=0.9
    e = ece(p, y, n_bins=10)
    assert e < 0.05


def test_ece_miscalibrated_positive():
    p = [0.9] * 10  # Confident YES
    y = [0] * 10    # All NO
    e = ece(p, y, n_bins=2)
    # Perfectly miscalibrated → ECE near 0.9 - 0 = 0.9
    assert e > 0.85


def test_brier_perfect_zero():
    assert brier([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0001 * 4 / 4 or brier([1, 1, 0, 0], [1, 1, 0, 0]) < 1e-6


def test_log_loss_perfect_low():
    ll = log_loss([0.99, 0.99, 0.01, 0.01], [1, 1, 0, 0])
    assert ll < 0.02


def test_decomposition_sums_to_brier():
    """Murphy: reliability − resolution + uncertainty = Brier (approximately).
    Our compute_brier_decomposition computes reliability and resolution
    separately; sum into the bound."""
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(0, 1, size=200), 0.01, 0.99)
    y = (rng.uniform(size=200) < p).astype(int)
    decomp = brier_decomposition(p, y, n_bins=10)
    b = brier(p, y)
    approx = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
    # Should be within 5% of true Brier (binned approx)
    assert abs(approx - b) < 0.05


def test_evaluate_predictions_returns_full_report():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=100)
    m = rng.uniform(0, 1, size=100)
    y = (rng.uniform(size=100) < p).astype(int)
    rep = evaluate_predictions(p, m, y)
    assert rep.n == 100
    assert 0 <= rep.brier <= 1
    assert 0 <= rep.ece <= 1
    assert rep.market_brier > 0


def test_evaluate_by_category():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200)
    m = rng.uniform(0, 1, size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    cats = (["sports"] * 100) + (["politics"] * 100)
    rep = evaluate_predictions(p, m, y, categories=cats)
    assert rep.by_category is not None
    assert "sports" in rep.by_category
    assert "politics" in rep.by_category


def test_compare_baselines_includes_all():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    m = rng.uniform(0, 1, size=50)
    y = (rng.uniform(size=50) < p).astype(int)
    res = compare_to_baselines(p, m, y)
    for k in ("model_brier", "market_brier", "uniform_brier", "base_rate_brier", "base_rate"):
        assert k in res
