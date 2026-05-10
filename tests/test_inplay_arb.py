"""Tests for in-play sports arb detector."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from polyagent.signals.inplay_arb import find_inplay_arbs, InPlayArb


@dataclass
class _MockBook:
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)

    def best_ask(self):
        if not self.asks:
            return None
        p = sorted(self.asks.keys())[0]
        return (p, self.asks[p])


class _MockBookStore:
    def __init__(self):
        self.books = {}


def test_no_arb_when_sum_under_one():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.40: 100})
    bs.books["t2"] = _MockBook(asks={0.50: 100})
    group = {
        "condition_id": "c1",
        "leg_token_ids": ["t1", "t2"],
        "game_start_ts": time.time() - 60,
        "game_end_ts": time.time() + 3600,
    }
    out = find_inplay_arbs(bs, [group], min_bps_gap=20)
    assert out == []


def test_arb_found_when_sum_over_one():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.40: 100})
    bs.books["t2"] = _MockBook(asks={0.61: 100})  # 0.40 + 0.61 = 1.01 → 100 bps
    group = {
        "condition_id": "c1",
        "leg_token_ids": ["t1", "t2"],
        "game_start_ts": time.time() - 60,
        "game_end_ts": time.time() + 3600,
    }
    out = find_inplay_arbs(bs, [group], min_bps_gap=20)
    assert len(out) == 1
    a = out[0]
    assert abs(a.bps_gap - 100.0) < 1e-6
    assert a.min_leg_size_at_stale == 100.0
    assert a.detected_during_game is True


def test_skip_arb_outside_game_window_when_in_game_only():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.40: 100})
    bs.books["t2"] = _MockBook(asks={0.65: 100})  # 1.05 sum
    group = {
        "condition_id": "c1",
        "leg_token_ids": ["t1", "t2"],
        "game_start_ts": time.time() + 3600,  # game hasn't started
        "game_end_ts": time.time() + 7200,
    }
    # Default: in_game_only=True
    out = find_inplay_arbs(bs, [group], min_bps_gap=20)
    assert out == []
    # Override: include pre-market arbs
    out2 = find_inplay_arbs(bs, [group], min_bps_gap=20, in_game_only=False)
    assert len(out2) == 1


def test_size_capped_by_smallest_leg():
    bs = _MockBookStore()
    bs.books["t1"] = _MockBook(asks={0.40: 50})    # smallest leg
    bs.books["t2"] = _MockBook(asks={0.62: 200})
    group = {
        "condition_id": "c1",
        "leg_token_ids": ["t1", "t2"],
        "game_start_ts": time.time() - 60,
        "game_end_ts": time.time() + 3600,
    }
    out = find_inplay_arbs(bs, [group], min_bps_gap=20)
    assert len(out) == 1
    assert out[0].size_capped == 50.0


def test_missing_book_skips_group():
    bs = _MockBookStore()
    # Only one of two legs has a book
    bs.books["t1"] = _MockBook(asks={0.40: 100})
    group = {
        "condition_id": "c1",
        "leg_token_ids": ["t1", "t2"],
        "game_start_ts": time.time() - 60,
        "game_end_ts": time.time() + 3600,
    }
    assert find_inplay_arbs(bs, [group]) == []
