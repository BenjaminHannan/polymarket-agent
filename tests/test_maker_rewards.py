"""Tests for Polymarket maker-rewards quadratic-spread tracker."""
from __future__ import annotations

import sqlite3
import time

from polyagent.risk.maker_rewards import (
    compute_reward_score,
    MakerRewardsTracker,
    projected_daily_reward,
    persist_tracker,
)


def test_score_zero_outside_cap():
    assert compute_reward_score(spread_bps=500, size=100, duration_sec=60, max_spread_bps=300) == 0.0


def test_score_max_at_tight_quote():
    s = compute_reward_score(spread_bps=0.001, size=100, duration_sec=60, max_spread_bps=300)
    # Near-zero spread approaches the upper bound of (1 × size × duration)
    assert s > 99 * 60 * 0.99


def test_score_zero_size_or_duration():
    assert compute_reward_score(50, 0, 60) == 0.0
    assert compute_reward_score(50, 100, 0) == 0.0


def test_score_quadratic_in_spread():
    s1 = compute_reward_score(spread_bps=50, size=1, duration_sec=1, max_spread_bps=100)
    s2 = compute_reward_score(spread_bps=80, size=1, duration_sec=1, max_spread_bps=100)
    # Quadratic: 1 - (0.5)² = 0.75 vs 1 - (0.8)² = 0.36
    assert abs(s1 - 0.75) < 1e-6
    assert abs(s2 - 0.36) < 1e-6


def test_projected_daily_reward_proportional():
    """Our share of the pool = our score / total score."""
    r = projected_daily_reward(our_score=100, market_total_score=1000, daily_pool_usd=500)
    assert abs(r - 50.0) < 1e-9


def test_tracker_accumulates():
    t = MakerRewardsTracker()
    # First sample: no time elapsed, returns 0
    t.sample("tok", 50, 100)
    assert t.cum_score("tok") == 0.0
    # Simulate passage of time
    time.sleep(0.05)
    t.sample("tok", 60, 90)
    s = t.cum_score("tok")
    assert s > 0


def test_persist_and_query(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    t = MakerRewardsTracker()
    t.sample("tok_a", 50, 100)
    time.sleep(0.02)
    t.sample("tok_a", 50, 100)
    persist_tracker(conn, t, daily_pool_usd=1000)
    row = conn.execute(
        "SELECT cum_score, projected_daily_usd FROM maker_rewards_score WHERE token_id='tok_a'"
    ).fetchone()
    assert row is not None
    assert row[0] >= 0.0
