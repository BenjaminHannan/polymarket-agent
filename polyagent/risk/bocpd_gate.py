"""Bayesian Online Changepoint Detection (Adams & MacKay 2007;
Tsaknaki, Lillo & Mazzarisi 2024) on the realized win/loss stream.

For each resolved trade we observe a Bernoulli outcome:
  x_t = 1 if pnl > 0 (a "win")
  x_t = 0 otherwise

We model the underlying win-rate p_t with a Beta(α, β) conjugate prior
and run Adams-MacKay's run-length posterior. When the posterior mass
on run-length=0 (= "changepoint at t") exceeds ``cp_threshold``, we
flip the gate into a "deleverage" state: every active strategy that
consults :meth:`size_multiplier` gets a multiplier of
``deleverage_mult`` (default 0.5) for ``deleverage_trades`` subsequent
resolutions, after which we restore the gate to 1.0.

Why Bernoulli (not Brier scalar):
- The conjugate Beta-Bernoulli model has a closed-form posterior
  predictive (= Beta-Binomial mean). No quadrature, no MCMC.
- Brier is bounded but real-valued; we'd need a Beta likelihood with
  parameter-of-parameter inference, much more code, marginally
  better detection.
- The doc's "BOCPD on Brier loss" recommendation is theoretically
  cleaner; in practice the win/loss reduction loses very little
  changepoint sensitivity at our N (~100s of trades) and is far
  simpler to verify.

Memory bound: the run-length posterior is truncated at
``max_run_length`` (default 500). Mass past the truncation is
re-normalized into the surviving cells.

References:
  Adams & MacKay (2007), arXiv:0710.3742.
  Tsaknaki, Lillo & Mazzarisi (2024), arXiv:2407.16376.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class BOCPDGate:
    hazard: float = 0.02           # constant hazard ≈ 50-trade prior on regime length;
                                   # empirically catches 50→20% shifts in ~20 trades
                                   # and 50→5% shifts in ~16 trades (see scripts/backtest_bocpd.py grid).
    alpha0: float = 2.0            # Beta prior α₀ (slightly informative)
    beta0: float = 2.0             # Beta prior β₀
    cp_threshold: float = 0.7      # P(changepoint at t) > this → flip gate
    deleverage_trades: int = 30    # how many subsequent trades to size-down
    deleverage_mult: float = 0.5   # size multiplier when gated
    max_run_length: int = 500      # truncation for memory bound

    # State
    posterior: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([2.0]))
    beta: np.ndarray = field(default_factory=lambda: np.array([2.0]))
    n_observed: int = 0

    # Detection state — modal run-length collapse is the canonical BOCPD
    # online signal (see ruptures, bocd refs). Under constant hazard,
    # P(r_t = 0 | x) collapses to ``hazard`` algebraically and is not
    # discriminative; the modal run-length argmax IS.
    modal_run_length: int = 0
    _historical_max_modal: int = 0

    # Deleverage state
    _cooldown_remaining: int = 0
    _last_cp_prob: float = 0.0
    _changepoints: list = field(default_factory=list)  # [(ts, cp_prob, n_obs)]

    def __post_init__(self):
        # Re-init posterior arrays to single-element arrays anchored at α0/β0
        # (the dataclass defaults capture closure values; this is safer).
        self.posterior = np.array([1.0], dtype=float)
        self.alpha = np.array([self.alpha0], dtype=float)
        self.beta = np.array([self.beta0], dtype=float)

    def update(self, win: bool) -> float:
        """Stream one Bernoulli observation. Returns P(changepoint at t)."""
        x = 1.0 if win else 0.0
        # Posterior predictive under each run-length: Beta-mean.
        pi = self.alpha / (self.alpha + self.beta)
        pred = np.where(x > 0.5, pi, 1.0 - pi)
        # Growth: each run-length r grows to r+1, weighted by predictive
        # likelihood × (1 − hazard).
        growth = self.posterior * pred * (1.0 - self.hazard)
        # Changepoint mass: total over all r-lengths × hazard.
        cp_mass = float((self.posterior * pred * self.hazard).sum())
        # New posterior: cp at front (r=0), then growth.
        new_post = np.concatenate([[cp_mass], growth])
        s = new_post.sum()
        if s > 0:
            new_post /= s
        else:
            new_post = np.zeros_like(new_post)
            new_post[0] = 1.0
        # Update suff stats: new r=0 cell starts at α0/β0; existing cells
        # grow with x.
        new_alpha = np.concatenate([[self.alpha0], self.alpha + x])
        new_beta = np.concatenate([[self.beta0], self.beta + (1.0 - x)])
        # Truncate
        if new_post.size > self.max_run_length:
            new_post = new_post[: self.max_run_length]
            new_alpha = new_alpha[: self.max_run_length]
            new_beta = new_beta[: self.max_run_length]
            s = new_post.sum()
            if s > 0:
                new_post /= s
        self.posterior = new_post
        self.alpha = new_alpha
        self.beta = new_beta
        self.n_observed += 1

        # Modal run-length collapse signal. When the data is consistent
        # with a long stable regime, the posterior mass concentrates on
        # large run-lengths and the argmax grows monotonically with t.
        # When a regime change happens, posterior mass shifts to short
        # run-lengths because the new observations match a freshly-
        # initialized cell better than they match the long-run cell —
        # so the argmax collapses. The collapse is the changepoint
        # signal.
        self.modal_run_length = int(np.argmax(self.posterior))
        if self.modal_run_length > self._historical_max_modal:
            self._historical_max_modal = self.modal_run_length

        # cp_prob: only triggers on a *severe* modal collapse to avoid
        # false-positives on stable-regime noise. Empirical sweep on
        # 10 stable-regime seeds at hazard=0.02 showed 3 seeds with
        # spurious modal dips when threshold was `1 - 2·ratio`. Tightened
        # to `1 - 5·ratio`, which fires only when modal < 20% of peak —
        # synthetic 50→20% shift still detected at delay~20 trades, but
        # 0 false-positives across 10 stable-regime seeds.
        if self._historical_max_modal < 30:
            cp_prob = 0.0
        else:
            ratio = self.modal_run_length / max(1, self._historical_max_modal)
            cp_prob = max(0.0, 1.0 - 5.0 * ratio)
        self._last_cp_prob = cp_prob

        # Decrement cooldown each observation
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Trigger deleverage if cp_prob crosses threshold
        if cp_prob > self.cp_threshold:
            triggered = self._cooldown_remaining < self.deleverage_trades
            if triggered:
                # Reset cooldown to full duration on every fresh trigger
                self._cooldown_remaining = self.deleverage_trades
                self._changepoints.append({
                    "ts": time.time(),
                    "cp_prob": cp_prob,
                    "n_observed": self.n_observed,
                })
                log.warning(
                    "bocpd_changepoint_detected",
                    cp_prob=round(cp_prob, 3),
                    n_observed=self.n_observed,
                    deleverage_trades=self.deleverage_trades,
                    deleverage_mult=self.deleverage_mult,
                )
        return cp_prob

    def size_multiplier(self) -> float:
        """Multiplier strategies should apply to per-trade Kelly sizing.
        1.0 normally; ``deleverage_mult`` while in cooldown."""
        return self.deleverage_mult if self._cooldown_remaining > 0 else 1.0

    def is_deleveraged(self) -> bool:
        return self._cooldown_remaining > 0

    def summary(self) -> dict:
        # Most-likely run length
        argmax = int(np.argmax(self.posterior))
        # Posterior mean win-rate weighted by run-length posterior
        pi = self.alpha / (self.alpha + self.beta)
        win_rate_mean = float((self.posterior * pi).sum())
        return {
            "n_observed": self.n_observed,
            "cp_prob_last": round(self._last_cp_prob, 3),
            "is_deleveraged": self.is_deleveraged(),
            "cooldown_remaining": self._cooldown_remaining,
            "modal_run_length": argmax,
            "win_rate_posterior_mean": round(win_rate_mean, 3),
            "n_changepoints": len(self._changepoints),
        }
