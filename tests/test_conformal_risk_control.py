"""Tests for Conformal Risk Control (Angelopoulos et al. ICLR 2024)."""
from __future__ import annotations

import numpy as np

from polyagent.risk.conformal_risk_control import ConformalRiskController


def test_burn_in_rejects_everything():
    """Before fit, accept should always return False."""
    c = ConformalRiskController(alpha=0.1)
    assert c.accept(0.99) is False
    assert c.accept(0.01) is False


def test_high_score_high_loss_blocks_low_scores():
    """When low scores have high losses, the controller should pick
    a high λ that admits only the safe (high-score) region."""
    rng = np.random.default_rng(0)
    n = 500
    # Synthetic: high score → low loss, low score → high loss.
    scores = rng.uniform(0, 1, size=n)
    losses = (1 - scores) ** 2  # high loss when score is low
    c = ConformalRiskController(alpha=0.05, higher_is_better=True)
    c.fit(scores, losses)
    assert c._lambda_hat is not None
    # Should admit high scores
    assert c.accept(0.95) is True
    # Should reject low scores
    assert c.accept(0.05) is False


def test_lower_alpha_stricter_threshold():
    """alpha=0.01 should give a stricter (higher) λ than alpha=0.5."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=1000)
    losses = (1 - scores) ** 2
    c_strict = ConformalRiskController(alpha=0.01)
    c_loose = ConformalRiskController(alpha=0.5)
    c_strict.fit(scores, losses)
    c_loose.fit(scores, losses)
    assert c_strict._lambda_hat >= c_loose._lambda_hat


def test_higher_is_better_false_inverts():
    """When higher_is_better=False, low scores should be admitted."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=500)
    # Loss now goes the other way: high score = high loss.
    losses = scores ** 2
    c = ConformalRiskController(alpha=0.05, higher_is_better=False)
    c.fit(scores, losses)
    # Admit low-score side
    assert c.accept(0.05) is True


def test_no_admissible_returns_strictest():
    """If no λ achieves the target, the controller picks the strictest
    (accepts nothing)."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=100)
    losses = np.full_like(scores, 10.0)  # losses way above any reasonable α
    c = ConformalRiskController(alpha=0.01)
    c.fit(scores, losses)
    # Should set λ to the strictest threshold ⇒ admit nothing.
    n_accepted = sum(c.accept(s) for s in scores)
    assert n_accepted == 0


def test_too_few_samples_no_fit():
    """n < 5 should leave the controller in burn-in."""
    c = ConformalRiskController(alpha=0.1)
    c.fit([0.5, 0.6], [0.1, 0.2])
    assert c._lambda_hat is None


def test_risk_at_returns_calibration_risk():
    rng = np.random.default_rng(0)
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    losses = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.01])
    c = ConformalRiskController(alpha=0.1)
    c.fit(scores, losses)
    # Risk above λ=0.4 includes scores 0.5, 0.7, 0.9, 0.95
    risk = c.risk_at(0.4)
    assert risk is not None
    # avg of 0.3, 0.1, 0.05, 0.01 = 0.115
    assert abs(risk - 0.115) < 1e-6


def test_summary_keys():
    c = ConformalRiskController(alpha=0.1)
    s = c.summary()
    assert "alpha_target" in s
    assert "lambda_hat" in s
    assert "n_calibration" in s
