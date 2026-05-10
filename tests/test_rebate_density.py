"""Tests for the per-market quote-density rebate adjustment."""
from __future__ import annotations

from polyagent.risk.rebate_density import adjust_rebate, compute_density_share


def test_share_when_alone_in_book():
    """If we're the only maker (visible == our size), share = 1.0."""
    s = compute_density_share(our_quote_size=25, visible_book_size_at_or_better=25)
    assert s.market_share == 1.0
    assert s.other_visible_size == 0.0


def test_share_when_others_dominate():
    """If we're 25/525 of the visible book, share ≈ 0.048."""
    s = compute_density_share(our_quote_size=25, visible_book_size_at_or_better=525)
    assert abs(s.market_share - 25 / 525) < 1e-9
    assert s.other_visible_size == 500.0


def test_share_when_others_smaller():
    s = compute_density_share(our_quote_size=100, visible_book_size_at_or_better=120)
    # others = 20, total = 120, our_share = 100/120 ≈ 0.833
    assert abs(s.market_share - 100 / 120) < 1e-9


def test_share_zero_size_returns_none():
    assert compute_density_share(our_quote_size=0, visible_book_size_at_or_better=100) is None
    assert compute_density_share(our_quote_size=-5, visible_book_size_at_or_better=100) is None


def test_adjust_rebate_alone_gives_full_credit():
    a = adjust_rebate(upper_bound=0.50, our_quote_size=25, visible_book_size_at_or_better=25)
    assert a.upper_bound == 0.50
    assert abs(a.adjusted_rebate - 0.50) < 1e-9
    assert a.market_share == 1.0


def test_adjust_rebate_haircut_proportional():
    """Half the visible book ours → half the rebate credit."""
    a = adjust_rebate(upper_bound=1.00, our_quote_size=50, visible_book_size_at_or_better=100)
    assert abs(a.market_share - 0.5) < 1e-9
    assert abs(a.adjusted_rebate - 0.50) < 1e-9


def test_adjust_rebate_zero_size_returns_zero_credit():
    a = adjust_rebate(upper_bound=1.00, our_quote_size=0, visible_book_size_at_or_better=100)
    assert a.adjusted_rebate == 0.0
