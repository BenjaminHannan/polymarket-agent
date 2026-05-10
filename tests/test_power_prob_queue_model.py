"""Tests for hftbacktest power_prob_queue_model=3 semantics."""
from __future__ import annotations

from polyagent.risk.queue_aware_fills import power_prob_queue_model


def test_front_of_queue_certain():
    """At q=0 (front) the probability is 1."""
    assert power_prob_queue_model(0, 100) == 1.0


def test_back_of_queue_zero():
    """At q=Q (back) the probability is 0."""
    assert power_prob_queue_model(100, 100) == 0.0


def test_monotone_decreasing():
    """As queue position grows, fill probability strictly decreases."""
    p0 = power_prob_queue_model(10, 100)
    p1 = power_prob_queue_model(50, 100)
    p2 = power_prob_queue_model(90, 100)
    assert p0 > p1 > p2


def test_power_3_more_pessimistic_than_power_1():
    """Higher power = more pessimistic at the same queue depth."""
    p_lin = power_prob_queue_model(50, 100, power=1.0)
    p_cube = power_prob_queue_model(50, 100, power=3.0)
    assert p_cube < p_lin  # cubic decay is faster
    assert abs(p_lin - 0.5) < 1e-9
    assert abs(p_cube - 0.125) < 1e-9  # 0.5^3


def test_empty_total_returns_one():
    """Defensive: zero total ⇒ probability 1 (no queue)."""
    assert power_prob_queue_model(0, 0) == 1.0


def test_q_larger_than_Q_handled():
    """If queue_ahead > total, clamps to back-of-queue."""
    p = power_prob_queue_model(200, 100)
    # The function uses Q = max(q, Q), so total becomes 200; ratio = 0.
    assert p == 0.0
