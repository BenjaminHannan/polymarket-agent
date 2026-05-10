"""Tests for consistency-loss training scaffold."""
from __future__ import annotations

from polyagent.signals.consistency_loss import (
    mee_sum_loss, monotonicity_loss, negation_invariance_loss,
    compute_consistency_loss, consistency_score,
)


def test_mee_sum_zero_when_calibrated():
    assert mee_sum_loss([0.3, 0.4, 0.3]) == 0.0


def test_mee_sum_positive_under_violation():
    loss = mee_sum_loss([0.3, 0.4, 0.5])  # sums to 1.2
    assert loss > 0
    assert abs(loss - 0.04) < 1e-9


def test_monotonicity_zero_when_satisfied():
    assert monotonicity_loss(0.3, 0.7) == 0.0  # subset ≤ superset


def test_monotonicity_positive_when_violated():
    loss = monotonicity_loss(0.8, 0.5)  # subset > superset
    assert abs(loss - 0.09) < 1e-9  # (0.3)²


def test_negation_invariance_zero_when_calibrated():
    assert negation_invariance_loss(0.3, 0.7) == 0.0


def test_negation_invariance_positive():
    loss = negation_invariance_loss(0.5, 0.4)  # sum = 0.9
    assert abs(loss - 0.01) < 1e-9


def test_compute_consistency_loss_combines():
    bundle = compute_consistency_loss(
        mee_clusters=[[0.3, 0.4, 0.5]],
        monotone_pairs=[(0.8, 0.5)],
        yes_no_pairs=[(0.5, 0.4)],
    )
    assert bundle.mee_sum_loss > 0
    assert bundle.monotonicity_loss > 0
    assert bundle.negation_invariance_loss > 0
    total = bundle.total(w_mee=1.0, w_mono=1.0, w_neg=1.0)
    assert abs(total - (0.04 + 0.09 + 0.01)) < 1e-6


def test_consistency_score_in_unit():
    bundle = compute_consistency_loss(
        mee_clusters=[[0.5, 0.5]],
        monotone_pairs=[(0.3, 0.7)],
        yes_no_pairs=[(0.5, 0.5)],
    )
    score = consistency_score(bundle)
    assert 0.0 <= score <= 1.0
    # A bundle with all zero loss should score 1.0.
    assert score > 0.99


def test_total_weights_respected():
    bundle = compute_consistency_loss(
        mee_clusters=[[0.5, 0.6]],
        monotone_pairs=[(0.8, 0.5)],
    )
    t1 = bundle.total(w_mee=1.0, w_mono=0.0, w_neg=1.0)
    t2 = bundle.total(w_mee=0.0, w_mono=1.0, w_neg=1.0)
    assert t1 != t2
