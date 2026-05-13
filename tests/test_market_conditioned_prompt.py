"""Tests for Market-Conditioned Prompting."""
from __future__ import annotations

from polyagent.models.market_conditioned_prompt import (
    build_market_conditioned_prompt,
    parse_market_conditioned_response,
    combine_with_explicit_update,
)


def test_prompt_includes_market_and_question():
    p = build_market_conditioned_prompt(
        question="Will X happen by D?",
        market_p=0.42,
        news_context="Some context",
    )
    assert "Will X happen by D?" in p
    assert "0.420" in p
    assert "Some context" in p


def test_prompt_handles_no_news():
    p = build_market_conditioned_prompt(
        question="Q?", market_p=0.5, news_context="",
    )
    assert "no relevant news retrieved" in p


def test_prompt_includes_base_rate_when_supplied():
    p = build_market_conditioned_prompt(
        question="Q", market_p=0.3, news_context="x", base_rate=0.18,
    )
    assert "0.180" in p


def test_parse_full_response():
    txt = """Reasoning: Strong evidence A and B.
Log-odds update: 0.75
Posterior P(YES): 0.62
"""
    p, lo = parse_market_conditioned_response(txt)
    assert abs(p - 0.62) < 1e-9
    assert abs(lo - 0.75) < 1e-9


def test_parse_handles_missing_fields():
    p, lo = parse_market_conditioned_response("incoherent garbage")
    assert p is None
    assert lo is None


def test_combine_uses_posterior_when_present():
    p = combine_with_explicit_update(market_p=0.3, log_odds_update=2.0, posterior_p=0.7)
    assert abs(p - 0.7) < 1e-9


def test_combine_applies_log_odds_when_no_posterior():
    # Market 0.5 (log-odds 0); +1.0 update → log-odds 1.0 → p ≈ 0.731
    p = combine_with_explicit_update(market_p=0.5, log_odds_update=1.0, posterior_p=None)
    assert 0.7 < p < 0.75


def test_combine_caps_extreme_updates():
    # +10.0 update with cap=1.5 ⇒ effective update is +1.5
    p = combine_with_explicit_update(
        market_p=0.5, log_odds_update=10.0, posterior_p=None, update_cap_nats=1.5,
    )
    # exp(1.5) / (1 + exp(1.5)) ≈ 0.818
    assert 0.81 < p < 0.83


def test_combine_defaults_to_market_when_both_none():
    p = combine_with_explicit_update(market_p=0.42, log_odds_update=None, posterior_p=None)
    assert abs(p - 0.42) < 1e-6
