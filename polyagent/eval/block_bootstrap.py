"""Politis-Romano (1994) stationary block bootstrap for Sharpe CIs.

Direct implementation of the doc's Problem-5 fix #4. Trade-level
returns from any real strategy are *autocorrelated* (clusters of wins
and losses) — the IID Sharpe formula and standard bootstrap both
under-estimate variance when this is true.

The stationary block bootstrap (Politis-Romano 1994, JASA) resamples
blocks of random geometric length so the resampled series is itself
stationary. This preserves autocorrelation at all lags up to the
average block length.

Reference:
  - D. Politis & J. Romano, "The Stationary Bootstrap," JASA 89(428):
    1303–1313, 1994.
  - O. Ledoit & M. Wolf, "Robust performance hypothesis testing with
    the Sharpe ratio," J. Empirical Finance 2008, 15(5): 850–859.

Usage
-----
```python
from polyagent.eval.block_bootstrap import (
    stationary_bootstrap_sharpe, sharpe_ci,
)

returns = np.array([...])  # per-trade or per-period returns
sharpe_dist = stationary_bootstrap_sharpe(returns, n_boot=2000, mean_block=10)
ci_lo, ci_hi = sharpe_ci(sharpe_dist, alpha=0.05)
```
"""
from __future__ import annotations

import math

import numpy as np
import structlog

log = structlog.get_logger()


def stationary_bootstrap_sample(
    x: np.ndarray,
    mean_block: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One stationary-block-bootstrap resample of length len(x).

    At each step, with probability 1/mean_block start a fresh random
    index; otherwise advance by 1 (wrapping). Produces a series of
    geometric-length blocks averaging `mean_block`.
    """
    rng = rng or np.random.default_rng()
    n = len(x)
    if n == 0:
        return x.copy()
    p_restart = 1.0 / max(1.0, mean_block)
    out = np.empty(n, dtype=x.dtype)
    idx = int(rng.integers(0, n))
    for i in range(n):
        out[i] = x[idx]
        if rng.random() < p_restart:
            idx = int(rng.integers(0, n))
        else:
            idx = (idx + 1) % n
    return out


def _sharpe(x: np.ndarray, periods_per_year: int = 252) -> float:
    if len(x) < 2:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return mu / sd * math.sqrt(periods_per_year)


def stationary_bootstrap_sharpe(
    returns: np.ndarray | list[float],
    *,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap distribution of the annualized Sharpe via stationary
    blocks. Returns an array of length `n_boot`.

    Args:
        returns: 1-D series of per-period returns (decimals, not pct).
        n_boot: number of bootstrap iterations. 2000 is the published
            sweet spot for stable percentile CIs.
        mean_block: average block length. Rule of thumb:
            mean_block ≈ n^(1/3). For n=200, mean_block≈6; for n=2000,
            mean_block≈13.
        periods_per_year: annualization factor. 252 for daily, 52 for
            weekly, etc. For trade-level returns use 1 (Sharpe is then
            unitless per-trade).
        seed: optional RNG seed for reproducibility.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros(n_boot)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = stationary_bootstrap_sample(x, mean_block=mean_block, rng=rng)
        out[i] = _sharpe(sample, periods_per_year=periods_per_year)
    return out


def sharpe_ci(
    bootstrap_sharpe: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile confidence interval at level (1-alpha) from a
    bootstrap Sharpe distribution."""
    lo = float(np.quantile(bootstrap_sharpe, alpha / 2))
    hi = float(np.quantile(bootstrap_sharpe, 1 - alpha / 2))
    return lo, hi


def bootstrap_p_value(
    returns: np.ndarray | list[float],
    *,
    null_sharpe: float = 0.0,
    n_boot: int = 2000,
    mean_block: float = 10.0,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> float:
    """Two-sided p-value for H0: true Sharpe = null_sharpe under the
    stationary-block bootstrap distribution.

    The p-value is the fraction of bootstrap Sharpes at or beyond the
    observed Sharpe in the *opposite* direction from null_sharpe.
    """
    x = np.asarray(returns, dtype=float)
    if len(x) < 2:
        return 1.0
    observed = _sharpe(x, periods_per_year=periods_per_year)
    dist = stationary_bootstrap_sharpe(
        x, n_boot=n_boot, mean_block=mean_block,
        periods_per_year=periods_per_year, seed=seed,
    )
    # Center distribution at null_sharpe and ask: how many samples are
    # at least as extreme as observed?
    centered = dist - (dist.mean() - null_sharpe)
    if observed >= null_sharpe:
        p = float((centered >= observed).mean())
    else:
        p = float((centered <= observed).mean())
    # Two-sided: double the one-sided.
    return min(1.0, 2.0 * p)
