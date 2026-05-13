"""Combinatorially-Symmetric Cross-Validation (CSCV) for PBO.

Direct implementation of the doc's Problem-5 fix #3 (Bailey-Borwein-
Lopez de Prado-Zhu 2014, SSRN 2326253; Arian-Norouzi-Seco 2024). The
**Probability of Backtest Overfitting** (PBO) is the published
companion to DSR: while DSR adjusts for the number of trials, PBO
estimates the probability that the strategy with the *best in-sample*
performance is *worse than median* out-of-sample.

A PBO close to 0.5 (or above) means the in-sample winner is no better
than a coin flip — strong evidence of overfitting. PBO < 0.1 means
the best configuration genuinely generalizes.

Existing Polyagent state
------------------------
We already run PBO on the v4 cert config-grid (5 configs, PBO=0.214).
This module standardizes the routine so any future strategy variant
can be evaluated identically. It exposes a single function:

  pbo_from_returns(returns_matrix, n_splits=8) -> {pbo, dsr_winner,
                                                  bos, los_pdf}

where `returns_matrix` is shape (T, C) — T time periods × C
configurations — and the routine returns:

  - pbo: Probability of Backtest Overfitting
  - dsr_winner: DSR of the in-sample winner re-evaluated on OOS
  - bos: array of "best out-of-sample" ranks per split
  - los_pdf: empirical pdf of out-of-sample loss vs in-sample winner

Reference
---------
  Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest
  Overfitting," J. Computational Finance 2017 / SSRN 2326253, 2014.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class CSCVResult:
    pbo: float
    dsr_winner_oos: float
    median_oos_winner: float
    n_splits: int
    n_configs: int
    n_periods: int

    def summary(self) -> dict:
        return self.__dict__.copy()


def _annual_sharpe(x: np.ndarray, periods_per_year: int = 252) -> float:
    if len(x) < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(x) / sd * np.sqrt(periods_per_year))


def pbo_from_returns(
    returns_matrix: np.ndarray,
    *,
    n_splits: int = 8,
    periods_per_year: int = 252,
) -> CSCVResult:
    """Bailey-Borwein-Lopez de Prado-Zhu 2014 PBO via CSCV.

    Args:
        returns_matrix: shape (T, C). One row per time period, one
            column per backtested configuration. Cells are *returns*,
            not cumulative.
        n_splits: number of equal-size time slices (must be even).
            T must be divisible by n_splits. The CSCV procedure
            partitions into S slices, then for every way to split S
            into two equal halves it computes in-sample/out-of-sample
            ranks. Total combinations = C(S, S/2).
        periods_per_year: annualisation for Sharpe.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"expected 2-D returns_matrix, got shape {R.shape}")
    T, C = R.shape
    if C < 2:
        raise ValueError(f"need ≥ 2 configurations, got {C}")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if T < n_splits:
        raise ValueError(f"T={T} < n_splits={n_splits}")
    block = T // n_splits
    # Truncate to exact multiple of n_splits.
    R = R[: block * n_splits, :]
    slices = [R[i * block:(i + 1) * block, :] for i in range(n_splits)]

    half = n_splits // 2
    combos = list(itertools.combinations(range(n_splits), half))
    n_combos = len(combos)
    if n_combos == 0:
        raise RuntimeError("no CSCV combinations — n_splits too small")

    # For each combination: pick `half` slices as IS, the rest as OOS.
    los_ranks = np.zeros(n_combos)
    winner_oos_sharpes = np.zeros(n_combos)
    for k, is_idx in enumerate(combos):
        is_set = set(is_idx)
        oos_idx = [i for i in range(n_splits) if i not in is_set]
        is_block = np.vstack([slices[i] for i in is_idx])
        oos_block = np.vstack([slices[i] for i in oos_idx])
        is_sharpe = np.array([_annual_sharpe(is_block[:, c], periods_per_year)
                              for c in range(C)])
        oos_sharpe = np.array([_annual_sharpe(oos_block[:, c], periods_per_year)
                               for c in range(C)])
        winner = int(np.argmax(is_sharpe))
        # Rank of the IS-winner in OOS Sharpe. PBO is the prob that this
        # rank is below median.
        rank = (oos_sharpe < oos_sharpe[winner]).sum() / max(1, C - 1)
        los_ranks[k] = rank
        winner_oos_sharpes[k] = oos_sharpe[winner]
    # PBO = prob (rank < 0.5) — winner is worse than median OOS.
    pbo = float(np.mean(los_ranks < 0.5))
    log.info(
        "cscv_pbo",
        pbo=round(pbo, 3),
        n_configs=C,
        n_periods=T,
        n_splits=n_splits,
        median_winner_oos_sharpe=round(float(np.median(winner_oos_sharpes)), 3),
    )
    return CSCVResult(
        pbo=pbo,
        dsr_winner_oos=float(np.median(winner_oos_sharpes)),
        median_oos_winner=float(np.median(winner_oos_sharpes)),
        n_splits=n_splits,
        n_configs=C,
        n_periods=T,
    )


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Bailey-Lopez de Prado 2014 Deflated Sharpe Ratio.

    Computes:
        z = (S - μ_max) / σ_max  (Bonferroni-Sidak style penalty for
                                  best-of-N trials)
        DSR = Φ(z * sqrt(n_obs))

    where μ_max ≈ sqrt(2 log n_trials) (Bonferroni upper bound on the
    Sharpe of a noise-only winner over n_trials).
    """
    if n_observations < 2 or n_trials < 1:
        return 0.0
    # Approximation of E[max Sharpe of n_trials independent noise]
    gamma = 0.5772156649  # Euler-Mascheroni
    if n_trials == 1:
        mu_max = 0.0
    else:
        mu_max = ((1 - gamma) * _gauss_quantile(1 - 1.0 / n_trials)
                  + gamma * _gauss_quantile(1 - 1.0 / (n_trials * np.e)))
    var_adj = 1 - skew * sharpe + (excess_kurtosis - 1) / 4 * sharpe * sharpe
    var_adj = max(var_adj, 1e-12)
    z = (sharpe - mu_max) / np.sqrt(var_adj / (n_observations - 1))
    # Φ(z)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2))))


def _gauss_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam approximation)."""
    p = float(p)
    if p <= 0 or p >= 1:
        return 0.0
    # Rational approx, accurate to ~1e-9 in the tails.
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
               / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
           / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
