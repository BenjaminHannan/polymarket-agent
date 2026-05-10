"""Brownian σ√Δt cancel-latency drift + last-look slippage model.

Direct implementation of the doc's Problem-4 fix #3 (Polygon-specific
cancel-latency) and fix #4 (Olding 2022 / Barzykin 2026 last-look).
Replaces the existing simple `cancel_latency_penalty()` in
`polyagent/risk/fees.py` with a formal Brownian-drift cancel model and
a symmetric last-look rejection threshold.

Why this matters
----------------
Polymarket's CTF runs on Polygon, which has ~2 s block time and
~73 ms AWS-eu-west-2 baseline RTT. Once we POST a cancel:

  1. ~50–150 ms TCP/TLS/REST to Polymarket's gateway.
  2. ~50–250 ms gateway → matching engine ack (Dubach 2026 fact #6).
  3. ~2 s for the next Polygon block where the on-chain `OrderFilled`
     could land — meaning a 2 s window where our resting quote is
     **still fillable** even though we've "cancelled" it.

In equity HFT one models this as a Brownian drift: the mid moves
σ√Δt over the latency window, and the toxic side of the book has a
probability `Φ(−drift / σ)` of picking off our cancelled-but-still-
live quote. The published treatment is Olding (2022) and the
slippage-control rejection in Barzykin (2026, arXiv 2603.07752).

Public API
----------
- `CancelLatencyModel.expected_loss_bps(book_state) -> float`
   Expected bps lost per round-trip from cancel-latency drift.
- `CancelLatencyModel.should_repost(observed_price_change_bps) -> bool`
   Last-look rejection: refuse to post a replacement quote if the mid
   has moved more than `last_look_threshold_bps` during the gap.
- `CancelLatencyModel.simulate_cancel(book, side, post_price, sigma_per_sec)`
   Monte-Carlo: return the probability that our cancelled quote gets
   filled before the cancel lands (the "ghost fill" rate that V2
   only partially closed per Polymarket Nov 2025).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# Φ(x): standard normal CDF in pure Python.
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class CancelLatencyModel:
    """Brownian σ√Δt cancel-drift + last-look model.

    Defaults are tuned to Polymarket post-V2 (Nov 2025):
      - block_sec=2.0    Polygon block time
      - rtt_sec=0.15     gateway round-trip (Dubach 2026 fact #6 lower bound)
      - last_look_bps=15 reject replacement quote if mid moved this much
                         during the cancel window (Barzykin 2026 default)
      - heavy_tail_p=0.05 5% of cancels see a multi-second outlier; we
                          inflate Δt by `heavy_tail_mult` for those.
    """
    block_sec: float = 2.0
    rtt_sec: float = 0.15
    last_look_bps: float = 15.0
    heavy_tail_p: float = 0.05
    heavy_tail_mult: float = 3.0
    # Whether to enable the model (default-on; flip off to compare).
    enabled: bool = True

    def effective_dt_sec(self) -> float:
        """Expected latency window. block_sec + rtt_sec, scaled up by
        heavy-tail factor weighted by `heavy_tail_p`."""
        base = self.block_sec + self.rtt_sec
        return base * (1.0 + self.heavy_tail_p * (self.heavy_tail_mult - 1.0))

    def expected_drift_bps(self, sigma_per_sec: float) -> float:
        """Brownian σ√Δt expected absolute drift in bps. Multiply by the
        ratio of drift-to-spread to get the fraction of the spread that
        the move eats."""
        dt = self.effective_dt_sec()
        return float(10_000.0 * sigma_per_sec * math.sqrt(dt))

    def expected_loss_bps(
        self,
        sigma_per_sec: float,
        spread_bps: float | None = None,
    ) -> float:
        """Expected loss in bps per round-trip from cancel-latency.

        Without an explicit spread, the loss equals the drift (we eat
        the full drift on adverse moves). With a spread, the loss is
        capped at the half-spread (we can't lose more than what's on
        offer between mid and our quote)."""
        drift = self.expected_drift_bps(sigma_per_sec)
        if spread_bps is None:
            return drift
        half = float(spread_bps) / 2.0
        return float(min(drift, half))

    def fill_probability(
        self,
        sigma_per_sec: float,
        distance_to_mid_bps: float,
    ) -> float:
        """Probability the cancelled quote still gets filled before the
        cancel lands. Φ(−distance / drift_sigma_bps) i.e. the chance
        the mid drifts further than the gap.

        `distance_to_mid_bps` is the absolute distance between our
        quote and the current mid in bps (positive)."""
        dt = self.effective_dt_sec()
        drift_sigma_bps = 10_000.0 * sigma_per_sec * math.sqrt(dt)
        if drift_sigma_bps <= 1e-9:
            return 0.0
        # Probability mid moves at least `distance` in the toxic direction:
        z = float(distance_to_mid_bps) / drift_sigma_bps
        return float(1.0 - _phi(z))

    def should_repost(self, observed_price_change_bps: float) -> bool:
        """Last-look rejection. Refuse to repost a replacement quote
        if the mid moved more than `last_look_bps` during the cancel
        window — that's evidence the venue saw informed flow and our
        replacement is about to be adversely-selected.

        Returns True iff |observed_price_change_bps| ≤ last_look_bps.
        """
        if not self.enabled:
            return True
        ok = abs(float(observed_price_change_bps)) <= self.last_look_bps
        if not ok:
            log.info(
                "cancel_latency_repost_blocked",
                observed_bps=round(observed_price_change_bps, 2),
                threshold_bps=self.last_look_bps,
            )
        return ok

    def simulate_cancel(
        self,
        sigma_per_sec: float,
        distance_to_mid_bps: float,
        n_iters: int = 1000,
        rng: random.Random | None = None,
    ) -> dict:
        """Monte-Carlo: sample `n_iters` cancel windows with optional
        heavy-tail inflation and return:
          - p_filled: empirical fill probability of a cancelled quote
          - avg_loss_bps: average bps loss conditional on fill
        """
        rng = rng or random.Random(0)
        dt_base = self.block_sec + self.rtt_sec
        n_filled = 0
        loss_acc = 0.0
        for _ in range(n_iters):
            dt = dt_base
            if rng.random() < self.heavy_tail_p:
                dt *= self.heavy_tail_mult
            drift = sigma_per_sec * math.sqrt(dt) * rng.gauss(0.0, 1.0)
            drift_bps = drift * 10_000.0
            if drift_bps >= distance_to_mid_bps:
                n_filled += 1
                loss_acc += drift_bps - distance_to_mid_bps
        p_filled = n_filled / max(1, n_iters)
        avg_loss = (loss_acc / n_filled) if n_filled else 0.0
        return {
            "p_filled": p_filled,
            "avg_loss_bps": avg_loss,
            "n_iters": n_iters,
        }
