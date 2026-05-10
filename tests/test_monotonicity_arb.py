"""Tests for monotonicity-arb detector."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polyagent.signals.monotonicity_arb import (
    detect_pairs,
    persist_candidates,
    open_candidates,
    _normalize_subject,
    _parse_threshold,
)


@dataclass
class _M:
    token_id: str
    question: str
    yes_price: float


def test_parse_threshold_kmb_suffix():
    assert _parse_threshold("50") == 50.0
    assert _parse_threshold("50k") == 50_000.0
    assert _parse_threshold("2.5M") == 2_500_000.0
    assert _parse_threshold("garbage") is None


def test_normalize_subject_strips_bounds():
    a = _normalize_subject("Will Bitcoin exceed $50k by Dec 31?")
    b = _normalize_subject("Will Bitcoin exceed $80k by Dec 31?")
    assert a == b  # same subject, different threshold


def test_detect_pairs_finds_violation():
    markets = [
        _M("tok_low", "Will Bitcoin exceed $50k by Dec 31?", 0.30),
        _M("tok_high", "Will Bitcoin exceed $80k by Dec 31?", 0.50),  # violation: p(80k) > p(50k)
    ]
    cands = detect_pairs(markets)
    assert len(cands) == 1
    c = cands[0]
    # 80k is the strict subset (tighter constraint) — high threshold goes there.
    assert c.threshold_subset == 80_000.0
    assert c.threshold_superset == 50_000.0
    assert c.gap == 0.20


def test_detect_pairs_no_violation_on_satisfied():
    markets = [
        _M("a", "Will X exceed 50 by D?", 0.70),
        _M("b", "Will X exceed 80 by D?", 0.30),  # satisfied: 0.30 ≤ 0.70
    ]
    cands = detect_pairs(markets)
    assert cands == []


def test_detect_pairs_ignores_singleton():
    markets = [_M("only", "Will X exceed 50?", 0.5)]
    assert detect_pairs(markets) == []


def test_persist_and_query(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    markets = [
        _M("tok_a", "Will Bitcoin exceed $50k by Dec 31?", 0.30),
        _M("tok_b", "Will Bitcoin exceed $80k by Dec 31?", 0.55),
    ]
    cands = detect_pairs(markets)
    n = persist_candidates(conn, cands)
    assert n == 1
    rows = open_candidates(conn, min_gap=0.0)
    assert len(rows) == 1
    assert rows[0]["gap"] > 0
