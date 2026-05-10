"""Tests for multi-LLM ensemble aggregator."""
from __future__ import annotations

import sqlite3

from polyagent.models.llm_ensemble import (
    LLMVote, aggregate, aggregate_with_log_pool,
    record_resolution, _brier_weights, ensure_table,
)


def test_simple_mean_basic():
    votes = [LLMVote("a", 0.4), LLMVote("b", 0.6), LLMVote("c", 0.5)]
    p = aggregate(votes, mode="simple_mean")
    assert abs(p - 0.5) < 1e-9


def test_simple_median_robust_to_outlier():
    votes = [LLMVote("a", 0.4), LLMVote("b", 0.5), LLMVote("c", 0.99)]
    p = aggregate(votes, mode="simple_median")
    assert abs(p - 0.5) < 1e-9


def test_accuracy_weighted_median_no_conn():
    votes = [LLMVote("a", 0.4), LLMVote("b", 0.5), LLMVote("c", 0.6)]
    p = aggregate(votes, mode="accuracy_weighted_median")
    # All equal weights ⇒ same as median.
    assert abs(p - 0.5) < 1e-9


def test_record_resolution_updates_history(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    record_resolution(conn, "model_x", 0.8, outcome_yes=True)
    record_resolution(conn, "model_x", 0.3, outcome_yes=False)
    row = conn.execute(
        "SELECT n_resolved, brier_sum FROM llm_brier_history WHERE model_name='model_x'"
    ).fetchone()
    assert row[0] == 2
    # First Brier: (0.8 - 1)² = 0.04; second: (0.3 - 0)² = 0.09. Sum = 0.13.
    assert abs(row[1] - 0.13) < 1e-6


def test_brier_weights_with_minimum_resolved(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_table(conn)
    # Below min_resolved threshold ⇒ uniform fallback.
    record_resolution(conn, "a", 0.5, outcome_yes=True)
    record_resolution(conn, "b", 0.5, outcome_yes=False)
    w = _brier_weights(conn, ["a", "b"], min_resolved=20)
    assert w["a"] == w["b"] == 0.5


def test_higher_order_shrinks_toward_mean():
    votes = [LLMVote("a", 0.2), LLMVote("b", 0.5), LLMVote("c", 0.9)]
    p_no_shrink = aggregate(votes, mode="accuracy_weighted_median")
    p_shrunk = aggregate(votes, mode="higher_order_aggregation",
                         higher_order_correlation=0.8)
    # With heavy shrinkage toward the (equal-weighted) mean 0.533,
    # the median should still be near the original 0.5 since the
    # shrunk values [0.266, 0.506, 0.826] still median at the center.
    assert 0.4 < p_shrunk < 0.6


def test_log_pool_with_market():
    votes = [LLMVote("a", 0.4), LLMVote("b", 0.5), LLMVote("c", 0.6)]
    p_with_market = aggregate_with_log_pool(votes, market_p=0.8, market_weight=0.7)
    # Market is 0.8 with 70% weight; LLM aggregate is 0.5 with 30%.
    # Should land closer to market.
    assert p_with_market > 0.65


def test_log_pool_without_market_returns_aggregate():
    votes = [LLMVote("a", 0.4), LLMVote("b", 0.5)]
    p = aggregate_with_log_pool(votes, market_p=None)
    assert 0.3 < p < 0.6
