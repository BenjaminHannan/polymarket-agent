"""Tests for Heng-Soh likelihood-ratio selective abstention."""
from __future__ import annotations

import numpy as np

from polyagent.risk.likelihood_ratio_gate import LikelihoodRatioGate


def test_burn_in_admits_everything():
    g = LikelihoodRatioGate(feature_dim=2, coverage=0.4, burn_in=200)
    rng = np.random.default_rng(0)
    for _ in range(50):
        f = rng.normal(0, 1, size=2)
        assert g.admit(f) is True


def test_post_burn_in_rejects_atypical():
    g = LikelihoodRatioGate(feature_dim=2, coverage=0.30, burn_in=50,
                            refit_every=50)
    rng = np.random.default_rng(1)
    # Inject 800 correct observations from N([0,0], I) and 200
    # wrong observations from N([3,3], I).
    for _ in range(800):
        g.add_observation(rng.normal(0, 1, size=2), was_correct=True)
    for _ in range(200):
        g.add_observation(rng.normal(3, 1, size=2), was_correct=False)
    g._refit()
    # Pre-populate the score buffer so the quantile is meaningful.
    for _ in range(150):
        g.rlog([float(rng.normal(0, 1)), float(rng.normal(0, 1))])
    # A signal near [0,0] looks like the "we got it right" distribution
    # ⇒ score should be much higher than at [3,3].
    score_correct = g.rlog([0.1, 0.1])
    score_far = g.rlog([3.0, 3.0])
    assert score_correct is not None and score_far is not None
    assert score_correct > score_far


def test_rlog_returns_none_without_fit():
    g = LikelihoodRatioGate(feature_dim=2)
    assert g.rlog([0.5, 0.5]) is None


def test_dim_mismatch_warns_but_doesnt_crash():
    g = LikelihoodRatioGate(feature_dim=2)
    g.add_observation([0.0, 0.0, 0.0], was_correct=True)  # wrong dim, no crash
    assert len(g._samples) == 0


def test_summary_shape():
    g = LikelihoodRatioGate()
    s = g.summary()
    assert "n_seen" in s
    assert "global_admit_rate" in s
    assert s["fit_ready"] is False
