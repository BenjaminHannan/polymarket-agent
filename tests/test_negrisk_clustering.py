"""Tests for NegRisk semantic clustering scaffold."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from polyagent.signals.negrisk_clustering import (
    detect_arb_candidates, NegRiskCluster,
)


def test_detect_arb_filters_below_gap():
    clusters = [
        NegRiskCluster(
            cluster_id="c1", token_ids=["a", "b"], questions=["q1", "q2"],
            yes_prices=[0.5, 0.51], sum_yes=1.01, arb_gap=0.01,
            mee_confirmed=True, confidence=0.9,
        ),
        NegRiskCluster(
            cluster_id="c2", token_ids=["c", "d"], questions=["q3", "q4"],
            yes_prices=[0.3, 0.8], sum_yes=1.10, arb_gap=0.10,
            mee_confirmed=True, confidence=0.9,
        ),
    ]
    arbs = detect_arb_candidates(clusters, min_arb_gap=0.05)
    assert len(arbs) == 1
    assert arbs[0].cluster_id == "c2"


def test_detect_arb_requires_mee_confirmed():
    cluster = NegRiskCluster(
        cluster_id="c", token_ids=["a"], questions=["q"], yes_prices=[0.5],
        sum_yes=0.5, arb_gap=0.5, mee_confirmed=False, confidence=0.0,
    )
    assert detect_arb_candidates([cluster]) == []


def test_detect_arb_respects_min_leg_size():
    cluster = NegRiskCluster(
        cluster_id="c", token_ids=["a", "b"],
        questions=["q1", "q2"], yes_prices=[0.3, 0.8],
        sum_yes=1.10, arb_gap=0.10, mee_confirmed=True, confidence=0.9,
    )
    # leg size lookup: 'a' has 100, 'b' has 50 ⇒ min = 50.
    sizes = {"a": 100.0, "b": 50.0}
    # min_leg_size=20 ⇒ both legs OK, candidate passes.
    out = detect_arb_candidates([cluster], min_leg_size=20.0,
                                leg_size_lookup=lambda t: sizes[t])
    assert len(out) == 1
    # min_leg_size=100 ⇒ 'b' too thin, candidate rejected.
    out2 = detect_arb_candidates([cluster], min_leg_size=100.0,
                                 leg_size_lookup=lambda t: sizes[t])
    assert out2 == []
