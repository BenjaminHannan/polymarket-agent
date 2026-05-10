"""Hierarchical Beta-binomial calibration across (category × horizon) cells.

Direct implementation of `pmwhybetter.md` Top-5 priority #3: "Hierarchical
Bayesian calibration + Bayesian Sharpe across (category × horizon)."

Pure-Python partial-pooling implementation that avoids the brms/Stan
dependency the doc suggests. The model is **empirical Bayes** style:

  Per cell c ∈ {(category × horizon)}:
      win_count_c | n_c, p_c ~ Binomial(n_c, p_c)
      p_c | μ, ν ~ Beta(μν, (1−μ)ν)

  Hyperprior (chosen as weakly informative):
      μ ~ Uniform(0, 1)          (global mean win-rate)
      ν ~ Half-Cauchy(20)        (concentration; higher ν ⇒ tighter pool)

The trick: instead of full MCMC, we fit hyperparameters by **Type-II
maximum-likelihood** (a.k.a. empirical Bayes) — maximise the marginal
likelihood of the data integrated over p_c. This is much cheaper than
MCMC and adequate for ~50 cells with ~10–500 observations each, which
is the scale Polyagent will reach in the next 1–2 years.

The Beta-Binomial marginal likelihood is closed-form:

  P(k_c | n_c, μ, ν) = C(n_c, k_c) · B(k_c + μν, n_c − k_c + (1−μ)ν)
                       / B(μν, (1−μ)ν)

(B = Beta function in log domain via lgamma).

Per-cell posterior is then a Beta(μν + k_c, (1−μ)ν + n_c − k_c). The
**calibrated probability** for that cell is the posterior mean.

Why this matters
----------------
- Cells with few samples (e.g. a new horizon for an emerging category)
  get pulled toward the global mean μ, avoiding wild over-fitting on
  3-sample cells.
- Cells with many samples dominate their own posterior, recovering the
  per-cell empirical rate.
- The shared ν means we estimate "how much to pool" from the data
  itself — fixed-strength priors like Beta(1, 1) get this wrong on
  unbalanced cells.

Mulligan QMF 2024 ("Bayesian Estimation for Sharpe Ratios under
Selection Bias") and Manokhin arXiv 2605.03816 both endorse this
partial-pooling recipe for calibration on prediction-style problems
where per-cell sample sizes vary by orders of magnitude.
"""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


def _log_beta(a: float, b: float) -> float:
    """log B(a, b) = lgamma(a) + lgamma(b) − lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_choose(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


@dataclass
class CellPosterior:
    cell_key: str
    n_obs: int
    n_wins: int
    posterior_mean: float
    posterior_low: float        # 2.5% quantile
    posterior_high: float       # 97.5% quantile
    raw_rate: float             # k_c / n_c (or 0.5 if n_c == 0)


def _beta_marginal_log_lik(
    counts: list[tuple[int, int]],
    mu: float,
    nu: float,
) -> float:
    """Log marginal likelihood of the Beta-Binomial model.

    `counts`: list of (n, k) pairs per cell.
    """
    if mu <= 0 or mu >= 1 or nu <= 0:
        return -math.inf
    alpha = mu * nu
    beta = (1 - mu) * nu
    log_b_prior = _log_beta(alpha, beta)
    total = 0.0
    for n, k in counts:
        if n <= 0:
            continue
        total += (
            _log_choose(n, k)
            + _log_beta(k + alpha, n - k + beta)
            - log_b_prior
        )
    return total


def _grid_search_hyperparams(
    counts: list[tuple[int, int]],
    n_mu: int = 21,
    n_nu: int = 24,
) -> tuple[float, float]:
    """Coarse grid search for the type-II ML (μ, ν). Cheap and stable;
    avoids gradient pathologies on degenerate cells.

    Grid:
      μ ∈ [0.05, 0.95] uniform
      ν ∈ log-spaced [1, 1000]
    """
    if not counts:
        return 0.5, 10.0
    best_ll = -math.inf
    best_mu = 0.5
    best_nu = 10.0
    mus = [0.05 + (0.95 - 0.05) * i / (n_mu - 1) for i in range(n_mu)]
    nus = [10 ** (0 + 3 * i / (n_nu - 1)) for i in range(n_nu)]  # 1 to 1000
    for mu in mus:
        for nu in nus:
            ll = _beta_marginal_log_lik(counts, mu, nu)
            if ll > best_ll:
                best_ll = ll
                best_mu = mu
                best_nu = nu
    return float(best_mu), float(best_nu)


def _beta_quantile(alpha: float, beta: float, q: float, n_iter: int = 25) -> float:
    """Bisection to find the q-quantile of Beta(alpha, beta).

    Uses the regularized incomplete beta CDF Ix(α, β) ≈ q. We integrate
    numerically via the continued-fraction Lentz method (pure Python).
    For n_iter=25 it's accurate to ~4 decimals which is plenty for
    quoting credible intervals.
    """
    def _betainc(a: float, b: float, x: float) -> float:
        """Regularized incomplete beta I_x(a, b) using continued fraction."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        # series transformation
        bt = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1 - x)
        )
        if x < (a + 1) / (a + b + 2):
            return bt * _betacf(a, b, x) / a
        return 1.0 - bt * _betacf(b, a, 1 - x) / b

    def _betacf(a: float, b: float, x: float) -> float:
        FPMIN = 1e-30
        qab = a + b
        qap = a + 1
        qam = a - 1
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, 101):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            del_ = d * c
            h *= del_
            if abs(del_ - 1.0) < 3e-7:
                return h
        return h

    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        m = (lo + hi) / 2
        if _betainc(alpha, beta, m) < q:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hierarchical_calibration (
            cell_key TEXT PRIMARY KEY,
            n_obs INTEGER NOT NULL,
            n_wins INTEGER NOT NULL,
            posterior_mean REAL NOT NULL,
            posterior_low REAL NOT NULL,
            posterior_high REAL NOT NULL,
            raw_rate REAL NOT NULL,
            mu_hat REAL NOT NULL,
            nu_hat REAL NOT NULL,
            last_updated REAL NOT NULL
        )"""
    )
    conn.commit()


def fit_and_persist(
    cell_observations: dict[str, tuple[int, int]],
    conn: sqlite3.Connection | None = None,
) -> tuple[dict[str, CellPosterior], float, float]:
    """Fit hierarchical Beta-Binomial calibration across cells.

    `cell_observations`: {cell_key: (n_obs, n_wins)}. Cells with n_obs=0
    are skipped from the hyperparameter fit but still get a posterior
    (which collapses to the global prior Beta(μν, (1−μ)ν)).

    Returns (posteriors_dict, mu_hat, nu_hat). Optionally persists to
    `hierarchical_calibration` table.
    """
    counts = [(n, k) for n, k in cell_observations.values() if n > 0]
    if not counts:
        return {}, 0.5, 10.0
    mu_hat, nu_hat = _grid_search_hyperparams(counts)
    alpha = mu_hat * nu_hat
    beta = (1 - mu_hat) * nu_hat
    out: dict[str, CellPosterior] = {}
    now = time.time()
    if conn is not None:
        ensure_table(conn)
    for cell, (n, k) in cell_observations.items():
        a_post = alpha + k
        b_post = beta + (n - k)
        mean = a_post / (a_post + b_post)
        lo = _beta_quantile(a_post, b_post, 0.025)
        hi = _beta_quantile(a_post, b_post, 0.975)
        raw = (k / n) if n > 0 else 0.5
        post = CellPosterior(
            cell_key=cell, n_obs=n, n_wins=k,
            posterior_mean=mean, posterior_low=lo,
            posterior_high=hi, raw_rate=raw,
        )
        out[cell] = post
        if conn is not None:
            conn.execute(
                """INSERT INTO hierarchical_calibration
                   (cell_key, n_obs, n_wins, posterior_mean, posterior_low,
                    posterior_high, raw_rate, mu_hat, nu_hat, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cell_key) DO UPDATE SET
                      n_obs=excluded.n_obs,
                      n_wins=excluded.n_wins,
                      posterior_mean=excluded.posterior_mean,
                      posterior_low=excluded.posterior_low,
                      posterior_high=excluded.posterior_high,
                      raw_rate=excluded.raw_rate,
                      mu_hat=excluded.mu_hat,
                      nu_hat=excluded.nu_hat,
                      last_updated=excluded.last_updated""",
                (cell, n, k, mean, lo, hi, raw, mu_hat, nu_hat, now),
            )
    if conn is not None:
        conn.commit()
    log.info(
        "hierarchical_calibration_fit",
        n_cells=len(out),
        mu_hat=round(mu_hat, 4),
        nu_hat=round(nu_hat, 2),
        total_obs=sum(n for n, _ in counts),
    )
    return out, mu_hat, nu_hat


def lookup_posterior(
    conn: sqlite3.Connection, cell_key: str
) -> CellPosterior | None:
    """O(1) cell lookup. Returns None if the cell hasn't been fit."""
    row = conn.execute(
        """SELECT cell_key, n_obs, n_wins, posterior_mean, posterior_low,
                  posterior_high, raw_rate
           FROM hierarchical_calibration WHERE cell_key=?""",
        (cell_key,),
    ).fetchone()
    if row is None:
        return None
    return CellPosterior(
        cell_key=row[0], n_obs=row[1], n_wins=row[2],
        posterior_mean=row[3], posterior_low=row[4],
        posterior_high=row[5], raw_rate=row[6],
    )
