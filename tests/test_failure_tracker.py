"""Tests for the model failure tracker.

Covers the pure classifier (which inputs produce which failure types
and severities) and the persistence layer (idempotency on re-insert).
"""
from __future__ import annotations

import sqlite3
import time

from polyagent.models.failure_tracker import (
    classify,
    ensure_table,
    record_failures,
)


def test_classify_high_confidence_wrong():
    # Model said 0.85 YES, actual was NO → high confidence wrong
    out = classify(p_model=0.85, p_market=None, yes_won=0)
    types = [f.failure_type for f in out]
    assert "high_confidence_wrong" in types


def test_classify_medium_confidence_wrong():
    out = classify(p_model=0.70, p_market=None, yes_won=0)
    types = [f.failure_type for f in out]
    assert "medium_confidence_wrong" in types
    assert "high_confidence_wrong" not in types


def test_classify_low_confidence_misses_no_record():
    # 0.55 vs 0 — wrong-side but barely confident; under threshold
    out = classify(p_model=0.55, p_market=None, yes_won=0)
    assert all(f.failure_type != "high_confidence_wrong" for f in out)
    assert all(f.failure_type != "medium_confidence_wrong" for f in out)


def test_classify_correct_no_failure():
    # Model correctly predicted high YES on a YES outcome
    out = classify(p_model=0.85, p_market=0.40, yes_won=1)
    assert out == []


def test_classify_loud_disagreement_market_right():
    # Model 0.10 (says NO), market 0.45 (less sure NO), actual YES.
    # Both are wrong, but the market is closer to truth → not "market_right"
    out = classify(p_model=0.10, p_market=0.45, yes_won=1)
    types = [f.failure_type for f in out]
    assert "model_loud_wrong_market_right" not in types

    # Model 0.10 says NO, market 0.55 says YES, actual YES → market right
    out = classify(p_model=0.10, p_market=0.55, yes_won=1)
    types = [f.failure_type for f in out]
    assert "model_loud_wrong_market_right" in types


def test_classify_disagrees_market_right():
    # Model 0.40 says NO weakly, market 0.55 says YES, actual YES → market correct, model wrong, gap=0.15
    out = classify(p_model=0.40, p_market=0.55, yes_won=1)
    types = [f.failure_type for f in out]
    assert "model_disagrees_market_right" in types
    assert "model_loud_wrong_market_right" not in types  # gap < 0.30


def test_classify_traded_loss():
    # Model called 0.85 YES, market was 0.40, we BUY YES at avg 0.40
    # for $50 notional, actual NO → realized -$50
    out = classify(
        p_model=0.85, p_market=0.40, yes_won=0,
        notional_traded=50.0, realized_pnl=-50.0,
    )
    types = [f.failure_type for f in out]
    assert "combined_wrong_traded" in types
    # And severity for that record should be 1.0 (full notional loss)
    rec = next(f for f in out if f.failure_type == "combined_wrong_traded")
    assert rec.severity == 1.0


def test_record_failures_persists_and_is_idempotent():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    n1 = record_failures(
        conn,
        condition_id="0xabc",
        resolved_ts=time.time(),
        yes_won=0,
        p_model=0.85,
        p_market=0.40,
        category="weather",
        question="Will it rain?",
        notional_traded=50.0,
        realized_pnl=-50.0,
    )
    assert n1 >= 2  # at least high_conf_wrong + traded_loss
    # Re-run with same condition_id+failure_type → unique index drops dupes
    n2 = record_failures(
        conn,
        condition_id="0xabc",
        resolved_ts=time.time(),
        yes_won=0,
        p_model=0.85,
        p_market=0.40,
        category="weather",
        question="Will it rain?",
        notional_traded=50.0,
        realized_pnl=-50.0,
    )
    assert n2 == 0  # all rows already present, INSERT OR IGNORE skipped them

    rows = conn.execute("SELECT failure_type FROM model_failures WHERE condition_id='0xabc'").fetchall()
    types = {r[0] for r in rows}
    assert "high_confidence_wrong" in types
    assert "model_loud_wrong_market_right" in types
    assert "combined_wrong_traded" in types


def test_record_no_failures_when_all_correct():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)
    n = record_failures(
        conn,
        condition_id="0xclean",
        resolved_ts=time.time(),
        yes_won=1,
        p_model=0.92,
        p_market=0.88,
        category="sports_global",
        question="Will the team win?",
    )
    assert n == 0
    cnt = conn.execute("SELECT COUNT(*) FROM model_failures").fetchone()[0]
    assert cnt == 0
