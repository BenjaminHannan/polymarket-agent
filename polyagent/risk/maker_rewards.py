"""Polymarket maker-rewards quadratic-spread tracker (pmwhybetter.md
Problem-10 fix #3).

Polymarket distributed **$12M in maker rewards in 2025** via a public
quadratic-spread scoring formula:

  score(quote) = (1 − (spread_bps / max_spread_bps)²) × size_at_quote
                 × time_at_quote

The reward pool for each market is divided across all eligible makers
proportional to their score. The doc framing (per Wanguolin's Medium
"two-week deep dive") treats this as **a bonus on top of edge**, not
the engine — but the rewards are real money and our maker quotes are
already shaped (Avellaneda-Stoikov), so we should at least *score*
them locally to know what we're missing.

What this module does
---------------------
1. **`compute_reward_score(spread_bps, size, duration_sec, max_spread_bps)`**
   returns the quadratic-spread score for one (quote, time-window)
   tuple. Higher = better maker reward eligibility.

2. **`MakerRewardsTracker`** is a stateful per-token accumulator that
   takes `(price, ask, bid, our_quote_size, our_quote_price)` snapshots
   and computes the running reward-pool fraction our quote has earned.

3. **`projected_daily_reward(score_share, daily_pool_usd)`** gives a
   dollar estimate of what we'd capture if the pool were $X/day on
   this market.

Why this matters even in paper mode
-----------------------------------
- Even if we *don't* claim the rewards (we'd need a real wallet), the
  score is a published per-tick metric of how *attractive* our quote
  is to the market. A high score means we're posting tight and deep —
  exactly the thing the doc says is the real edge.
- The score is *additive* with the strategy P&L: it's the implied
  per-trade subsidy a real Polymarket MM seat would collect. Dashboarding
  it tells us whether we're leaving rewards on the table by quoting too
  wide.

References
----------
- Polymarket docs.polymarket.com/market-makers/liquidity-rewards
- Wanguolin Medium "Two-week deep dive into Polymarket maker rewards"
- Bartlett & O'Hara 2026 (SSRN 6615739) — frames maker subsidies as
  capturing the systematic-YES-overbet behavioural surplus.
"""
from __future__ import annotations

import math
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


# Polymarket's documented default cap for the quadratic-spread term.
# Quotes wider than this earn zero score regardless of size or time.
DEFAULT_MAX_SPREAD_BPS = 300.0  # 3% half-spread = 300 bps


def compute_reward_score(
    spread_bps: float,
    size: float,
    duration_sec: float,
    *,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
) -> float:
    """Polymarket's quadratic-spread maker score for one quote.

    score = (1 − (s / S)²) × size × duration

    where s = our half-spread in bps and S = the program cap. Zero
    when s ≥ S; near-1 when s = 0. Returns 0 for zero/negative inputs.
    """
    s = max(0.0, float(spread_bps))
    S = max(1.0, float(max_spread_bps))
    sz = max(0.0, float(size))
    dt = max(0.0, float(duration_sec))
    if s >= S or sz == 0 or dt == 0:
        return 0.0
    quad = 1.0 - (s / S) ** 2
    return float(quad * sz * dt)


def projected_daily_reward(
    our_score: float,
    market_total_score: float,
    daily_pool_usd: float,
) -> float:
    """USD reward our quote would have earned for one day if the pool
    is `daily_pool_usd` and aggregate market score is
    `market_total_score`."""
    if market_total_score <= 0:
        return 0.0
    return float(daily_pool_usd) * float(our_score) / float(market_total_score)


@dataclass
class _PerTokenAccum:
    last_sample_ts: float = 0.0
    cum_score: float = 0.0
    n_samples: int = 0
    last_spread_bps: float = 0.0
    last_size: float = 0.0


@dataclass
class MakerRewardsTracker:
    """Per-token rolling maker-rewards score accumulator.

    Call `sample(token_id, our_quote_spread_bps, our_quote_size)` from
    the strategy's quote loop. Every call advances the cumulative score
    by the time-weighted quadratic-spread × size product since the
    last sample.

    Args:
        max_spread_bps: program cap (default 300 = 3% half-spread).
        decay_window_sec: keep scores from the last N seconds; older
            samples are evicted. Default 86400 (1 day) so the score
            is interpretable as "today's earnings."
    """
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS
    decay_window_sec: float = 86400.0
    _state: dict[str, _PerTokenAccum] = field(default_factory=dict)

    def sample(
        self,
        token_id: str,
        our_quote_spread_bps: float,
        our_quote_size: float,
    ) -> float:
        """Update the score for this token and return the incremental
        score earned since the last sample."""
        now = time.time()
        st = self._state.get(token_id)
        if st is None:
            st = _PerTokenAccum(last_sample_ts=now)
            self._state[token_id] = st
            st.last_spread_bps = float(our_quote_spread_bps)
            st.last_size = float(our_quote_size)
            return 0.0
        dt = max(0.0, now - st.last_sample_ts)
        # Score earned during the interval is computed using the *prior*
        # quote — i.e. what was on the book between samples.
        inc = compute_reward_score(
            spread_bps=st.last_spread_bps,
            size=st.last_size,
            duration_sec=dt,
            max_spread_bps=self.max_spread_bps,
        )
        st.cum_score += inc
        st.n_samples += 1
        st.last_sample_ts = now
        st.last_spread_bps = float(our_quote_spread_bps)
        st.last_size = float(our_quote_size)
        return inc

    def cum_score(self, token_id: str) -> float:
        """Cumulative reward score for this token."""
        st = self._state.get(token_id)
        return st.cum_score if st else 0.0

    def total_score(self) -> float:
        return sum(s.cum_score for s in self._state.values())

    def summary(self) -> dict:
        return {
            "n_tokens": len(self._state),
            "total_score": self.total_score(),
            "max_spread_bps": self.max_spread_bps,
            "decay_window_sec": self.decay_window_sec,
        }


# ── Persistence helpers ────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS maker_rewards_score (
            token_id TEXT PRIMARY KEY,
            cum_score REAL NOT NULL,
            n_samples INTEGER NOT NULL,
            last_spread_bps REAL NOT NULL,
            last_size REAL NOT NULL,
            last_sample_ts REAL NOT NULL,
            projected_daily_usd REAL
        )"""
    )
    conn.commit()


def persist_tracker(
    conn: sqlite3.Connection,
    tracker: MakerRewardsTracker,
    *,
    daily_pool_usd: float = 33_000.0,  # $12M/yr ÷ 365 ÷ ~mean markets ≈ rough per-market average
    market_total_score: float | None = None,
) -> None:
    """Persist the tracker's per-token state to sqlite. If
    `market_total_score` is None, use the tracker's own total — the
    projected_daily_usd will then be each market's share of its own
    total, which is degenerate (always = daily_pool_usd / n_tokens).
    Pass a real market-wide aggregate (queried from chain) for an
    honest projection.
    """
    ensure_table(conn)
    total = market_total_score if market_total_score is not None else tracker.total_score()
    for token_id, st in tracker._state.items():
        proj = projected_daily_reward(st.cum_score, total, daily_pool_usd)
        conn.execute(
            """INSERT INTO maker_rewards_score
               (token_id, cum_score, n_samples, last_spread_bps,
                last_size, last_sample_ts, projected_daily_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(token_id) DO UPDATE SET
                  cum_score=excluded.cum_score,
                  n_samples=excluded.n_samples,
                  last_spread_bps=excluded.last_spread_bps,
                  last_size=excluded.last_size,
                  last_sample_ts=excluded.last_sample_ts,
                  projected_daily_usd=excluded.projected_daily_usd""",
            (token_id, st.cum_score, st.n_samples, st.last_spread_bps,
             st.last_size, st.last_sample_ts, proj),
        )
    conn.commit()
