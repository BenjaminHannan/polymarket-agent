"""Tests for SmartMoneyRegistry.rank_weight (Akey 2026 rank concentration)."""
from __future__ import annotations

from polyagent.risk.smart_money import SmartMoneyRegistry


def test_rank_weight_unknown_wallet_returns_one():
    r = SmartMoneyRegistry()
    r.smart_wallets = {"aaa", "bbb", "ccc"}
    assert r.rank_weight("not_smart") == 1.0


def test_rank_weight_empty_set_returns_one():
    r = SmartMoneyRegistry()
    assert r.rank_weight("anything") == 1.0


def test_rank_weight_top_percentile():
    """Wallet at the top 1% by lexicographic rank gets 4× weight."""
    r = SmartMoneyRegistry()
    # 1000 wallets — top 1% = first 10
    r.smart_wallets = {f"wallet_{i:04d}" for i in range(1000)}
    # wallet_0000 sorts first
    assert r.rank_weight("wallet_0000") == 4.0
    # wallet_0005 is still in top 1% (10 of 1000)
    assert r.rank_weight("wallet_0005") == 4.0


def test_rank_weight_top_10_percent():
    r = SmartMoneyRegistry()
    r.smart_wallets = {f"wallet_{i:04d}" for i in range(1000)}
    # wallet_0050 is at 5% rank — in top 10% but not top 1%
    assert r.rank_weight("wallet_0050") == 2.0


def test_rank_weight_long_tail_returns_one():
    r = SmartMoneyRegistry()
    r.smart_wallets = {f"wallet_{i:04d}" for i in range(1000)}
    # wallet_0950 is in the bottom 5% by rank
    assert r.rank_weight("wallet_0950") == 1.0


def test_rank_weight_case_insensitive():
    r = SmartMoneyRegistry()
    r.smart_wallets = {"abc", "def", "xyz"}
    # is_smart is case-insensitive (lowercases input)
    assert r.is_smart("ABC") is True
    # rank_weight should also work case-insensitively
    assert r.rank_weight("ABC") > 1.0
