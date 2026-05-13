"""Bayesian Sharpe with t-likelihood (Kruschke BEST, Mulligan QMF 2024).

Direct implementation of the doc's Problem-5 fix #1. At Polyagent's
current sample size (n ≈ 190 fills, growing slowly), the frequentist
Sharpe ratio and its Bailey-LdP DSR are point estimates with very
wide standard errors. A Bayesian posterior over (μ, σ, ν) with weakly
informative priors gives **credible intervals** that are honest about
the small-sample uncertainty.

The published recipe (Kruschke "BEST" 2013; Mulligan QMF 2024
"Bayesian Estimation for Sharpe Ratios under Selection Bias") uses a
Student-t likelihood (because trade-level returns are leptokurtic) and
broad priors:

  μ ~ Normal(0, σ_prior²)
  σ ~ Half-Cauchy(σ_scale)
  ν ~ Exponential(1/30)   # degrees of freedom; broad enough for both
                            normal-ish and heavy-tailed regimes
  r_i | μ, σ, ν ~ Student-t(ν, μ, σ)

Posterior is sampled via a simple Metropolis-Hastings chain (we avoid
adding a hard dependency on pymc/numpyro). For the n ≤ 1000 regime
this is plenty.

Output
------
- posterior_samples(): DataFrame-like dict with keys mu, sigma, nu,
  sharpe (= mu/sigma * sqrt(periods_per_year))
- credible_interval(): (lo, hi) at given alpha
- prob_positive(): P(Sharpe > 0 | data)
- bayes_dsr(): Bayesian counterpart to Deflated Sharpe — fraction of
  posterior with Sharpe > psr_threshold; defaults aligned with Bailey-
  Lopez de Prado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


def _log_t_pdf(x: np.ndarray, mu: float, sigma: float, nu: float) -> np.ndarray:
    """Log-pdf of Student-t with location μ, scale σ, df ν."""
    if sigma <= 0 or nu <= 0:
        return np.full_like(x, -np.inf)
    z = (x - mu) / sigma
    norm = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
            - 0.5 * math.log(nu * math.pi) - math.log(sigma))
    return norm - ((nu + 1) / 2) * np.log1p(z * z / nu)


def _log_prior(mu: float, sigma: float, nu: float,
               sigma_prior: float, sigma_scale: float) -> float:
    # Normal(0, sigma_prior^2) on μ
    lp_mu = -0.5 * (mu / sigma_prior) ** 2 - math.log(sigma_prior * math.sqrt(2 * math.pi))
    # Half-Cauchy(sigma_scale) on σ (improper outside σ > 0)
    if sigma <= 0:
        return -np.inf
    lp_sigma = math.log(2.0) - math.log(math.pi * sigma_scale) - math.log(1.0 + (sigma / sigma_scale) ** 2)
    # Exponential(1/30) on ν
    if nu <= 0:
        return -np.inf
    lp_nu = -nu / 30.0 - math.log(30.0)
    return lp_mu + lp_sigma + lp_nu


@dataclass
class BayesianSharpeResult:
    mu_samples: np.ndarray
    sigma_samples: np.ndarray
    nu_samples: np.ndarray
    sharpe_samples: np.ndarray         # annualised
    n_draws: int
    n_data: int
    annualization: int
    acceptance_rate: float

    def credible_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        return (float(np.quantile(self.sharpe_samples, alpha / 2)),
                float(np.quantile(self.sharpe_samples, 1 - alpha / 2)))

    def median(self) -> float:
        return float(np.median(self.sharpe_samples))

    def prob_positive(self) -> float:
        return float(np.mean(self.sharpe_samples > 0))

    def prob_above(self, threshold: float) -> float:
        return float(np.mean(self.sharpe_samples > threshold))

    def bayes_dsr(self, threshold: float = 0.0) -> float:
        """Bayesian counterpart to DSR — P(annualized Sharpe > threshold
        | data). At threshold=0 this equals prob_positive; at the
        Bailey-LdP "1-year of trading at PSR=0.95" threshold the
        result is the Bayesian DSR."""
        return self.prob_above(threshold)

    def summary(self) -> dict:
        ci = self.credible_interval()
        return {
            "median_sharpe": self.median(),
            "ci_95": ci,
            "prob_positive": self.prob_positive(),
            "n_draws": self.n_draws,
            "n_data": self.n_data,
            "acceptance_rate": self.acceptance_rate,
        }


def bayesian_sharpe(
    returns: np.ndarray | list[float],
    *,
    periods_per_year: int = 252,
    n_draws: int = 4000,
    n_burn: int = 1000,
    seed: int | None = None,
    sigma_prior: float = 0.5,
    sigma_scale: float = 0.1,
    propose_scale_mu: float = 0.01,
    propose_scale_sigma: float = 0.005,
    propose_scale_nu: float = 2.0,
) -> BayesianSharpeResult:
    """Run Metropolis-Hastings to sample the posterior over (μ, σ, ν)
    of the t-likelihood model, and return the annualised Sharpe
    posterior."""
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 5:
        # Too few data — return a trivial posterior centred on a flat
        # likelihood. Caller should check n_data.
        return BayesianSharpeResult(
            mu_samples=np.zeros(1),
            sigma_samples=np.ones(1),
            nu_samples=np.full(1, 10.0),
            sharpe_samples=np.zeros(1),
            n_draws=1,
            n_data=n,
            annualization=periods_per_year,
            acceptance_rate=0.0,
        )
    rng = np.random.default_rng(seed)

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) or 0.01
    nu = 10.0

    def log_post(mu_, sigma_, nu_):
        if sigma_ <= 0 or nu_ <= 0:
            return -np.inf
        lp = _log_prior(mu_, sigma_, nu_, sigma_prior, sigma_scale)
        ll = float(np.sum(_log_t_pdf(x, mu_, sigma_, nu_)))
        return lp + ll

    cur_lp = log_post(mu, sigma, nu)
    mu_chain = np.empty(n_draws)
    sigma_chain = np.empty(n_draws)
    nu_chain = np.empty(n_draws)
    n_accept = 0
    total = n_draws + n_burn

    for t in range(total):
        mu_p = mu + rng.normal(0.0, propose_scale_mu)
        sigma_p = abs(sigma + rng.normal(0.0, propose_scale_sigma))
        nu_p = max(1.5, nu + rng.normal(0.0, propose_scale_nu))
        new_lp = log_post(mu_p, sigma_p, nu_p)
        if not np.isfinite(new_lp):
            log_alpha = -np.inf
        else:
            log_alpha = new_lp - cur_lp
        if math.log(rng.random()) < log_alpha:
            mu, sigma, nu = mu_p, sigma_p, nu_p
            cur_lp = new_lp
            n_accept += 1
        if t >= n_burn:
            i = t - n_burn
            mu_chain[i] = mu
            sigma_chain[i] = sigma
            nu_chain[i] = nu

    sharpe_chain = (mu_chain / np.maximum(sigma_chain, 1e-12)
                    * math.sqrt(periods_per_year))
    result = BayesianSharpeResult(
        mu_samples=mu_chain,
        sigma_samples=sigma_chain,
        nu_samples=nu_chain,
        sharpe_samples=sharpe_chain,
        n_draws=n_draws,
        n_data=n,
        annualization=periods_per_year,
        acceptance_rate=n_accept / total,
    )
    log.info(
        "bayesian_sharpe_done",
        n=n,
        median_sharpe=round(result.median(), 3),
        ci=tuple(round(v, 3) for v in result.credible_interval()),
        acceptance_rate=round(result.acceptance_rate, 2),
    )
    return result
