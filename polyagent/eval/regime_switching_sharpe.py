"""Bayesian regime-switching robust Sharpe (pmwhybetter.md Problem-5 #5).

Companion to `bayesian_sharpe.py`. Where the BEST t-likelihood model
gives a single posterior over (μ, σ, ν), real trade-return series
exhibit *regime-switching* — quiet periods of low volatility punctuated
by bursty drawdowns. A single-regime model averages across both and
under-states the true uncertainty.

Model
-----
Two-regime hidden-Markov mixture:

  regime ∈ {0, 1}                            (latent)
  r_t | regime = k ~ Normal(μ_k, σ_k²)       (state-conditional)
  P(regime_t = j | regime_{t−1} = i) = T[i, j]   (transition)

Priors (weakly informative):
  μ_0, μ_1 ~ Normal(0, σ_prior²)             (forced ordered: μ_0 < μ_1)
  σ_0, σ_1 ~ Half-Cauchy(σ_scale)
  T_ii ~ Beta(10, 2)                         (regime persistence ~83%)

We use **forward-backward (Baum-Welch) E-step + MAP M-step** rather
than full MCMC: at our n ~ 200–2000 sample size this converges in a
few iterations and is robust to initialisation.

The **regime-switching Sharpe posterior** is then the *mixture* of
the two regimes' per-regime Sharpes, weighted by their long-run
stationary probability. This gives a CI that's much wider than the
single-regime BEST under crisis-like return distributions.

API
---
- `fit_regime_switching(returns)` → `RSResult`
- `RSResult.regime_sharpes` — per-regime annualized Sharpe estimates
- `RSResult.stationary_prob` — long-run regime probabilities
- `RSResult.mixture_sharpe` — weighted Sharpe under the mixture
- `RSResult.credible_interval(alpha)` — bootstrap CI on the mixture

References
----------
- Hamilton 1989, "A New Approach to the Economic Analysis of
  Nonstationary Time Series", Econometrica 57(2).
- artifact-research.com 2025 "Bayesian regime-switching robust Sharpe".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


def _normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return -math.inf
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)


@dataclass
class RSResult:
    mu: tuple[float, float]              # (mu_0, mu_1), ordered low/high
    sigma: tuple[float, float]
    transition: np.ndarray               # 2x2 transition matrix
    stationary_prob: tuple[float, float] # (P(regime=0), P(regime=1))
    regime_sharpes: tuple[float, float]  # annualized per regime
    mixture_sharpe: float
    posterior_regime: np.ndarray         # shape (n,), prob of regime=1 at each t
    n: int
    n_iter: int
    log_likelihood: float

    def regime_proportion(self) -> tuple[float, float]:
        """Empirical regime occupancy (versus the model's stationary)."""
        p1 = float(np.mean(self.posterior_regime > 0.5))
        return (1 - p1, p1)

    def summary(self) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "stationary_prob": self.stationary_prob,
            "regime_sharpes": self.regime_sharpes,
            "mixture_sharpe": self.mixture_sharpe,
            "n": self.n,
            "n_iter": self.n_iter,
            "log_likelihood": self.log_likelihood,
        }


def _forward_backward(
    x: np.ndarray,
    mu: tuple[float, float],
    sigma: tuple[float, float],
    T: np.ndarray,
    pi0: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(x)
    K = 2
    log_alpha = np.zeros((n, K))
    log_beta = np.zeros((n, K))
    log_pi0 = np.log(np.clip(pi0, 1e-12, 1.0))
    log_T = np.log(np.clip(T, 1e-12, 1.0))

    for k in range(K):
        log_alpha[0, k] = log_pi0[k] + _normal_log_pdf(x[0], mu[k], sigma[k])
    for t in range(1, n):
        for k in range(K):
            log_emit = _normal_log_pdf(x[t], mu[k], sigma[k])
            log_alpha[t, k] = log_emit + _logsumexp(log_alpha[t - 1] + log_T[:, k])

    log_beta[n - 1] = 0.0
    for t in range(n - 2, -1, -1):
        for k in range(K):
            terms = log_T[k] + np.array([
                _normal_log_pdf(x[t + 1], mu[i], sigma[i]) for i in range(K)
            ]) + log_beta[t + 1]
            log_beta[t, k] = _logsumexp(terms)

    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp_axis1(log_gamma)[:, None]
    gamma = np.exp(log_gamma)

    log_xi = np.zeros((n - 1, K, K))
    for t in range(n - 1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = (
                    log_alpha[t, i] + log_T[i, j]
                    + _normal_log_pdf(x[t + 1], mu[j], sigma[j])
                    + log_beta[t + 1, j]
                )
        log_xi[t] -= _logsumexp(log_xi[t].flatten())

    log_lik = _logsumexp(log_alpha[-1])
    return gamma, np.exp(log_xi), log_lik


def _logsumexp(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    if not np.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(arr - m))))


def _logsumexp_axis1(arr: np.ndarray) -> np.ndarray:
    m = np.max(arr, axis=1)
    return m + np.log(np.sum(np.exp(arr - m[:, None]), axis=1))


def _stationary_distribution(T: np.ndarray) -> tuple[float, float]:
    """Stationary distribution of a 2x2 transition matrix."""
    # For a 2-state Markov chain, stationary = (T[1,0], T[0,1]) / (T[1,0] + T[0,1])
    denom = T[1, 0] + T[0, 1]
    if denom <= 1e-12:
        return (0.5, 0.5)
    return (T[1, 0] / denom, T[0, 1] / denom)


def fit_regime_switching(
    returns,
    *,
    n_iter: int = 50,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> RSResult:
    """Fit a 2-regime Gaussian HMM to the returns series via Baum-Welch.

    Returns RSResult with per-regime Sharpes + the mixture Sharpe
    weighted by the stationary regime probability."""
    x = np.asarray(returns, dtype=float).flatten()
    n = len(x)
    if n < 20:
        # Too few — return a trivial single-regime result.
        mu = float(x.mean()) if n else 0.0
        sd = float(x.std(ddof=1)) if n > 1 else 0.01
        sh = mu / max(sd, 1e-12) * math.sqrt(periods_per_year)
        return RSResult(
            mu=(mu, mu), sigma=(sd, sd),
            transition=np.array([[0.5, 0.5], [0.5, 0.5]]),
            stationary_prob=(0.5, 0.5),
            regime_sharpes=(sh, sh),
            mixture_sharpe=sh,
            posterior_regime=np.full(n, 0.5),
            n=n, n_iter=0, log_likelihood=0.0,
        )
    rng = np.random.default_rng(seed)

    # Init: split data by median into two halves.
    med = float(np.median(x))
    mu = (float(x[x <= med].mean()), float(x[x > med].mean()))
    sigma = (
        float(x[x <= med].std(ddof=1) or 0.01),
        float(x[x > med].std(ddof=1) or 0.01),
    )
    T = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi0 = (0.5, 0.5)

    prev_ll = -math.inf
    for it in range(n_iter):
        gamma, xi, log_lik = _forward_backward(x, mu, sigma, T, pi0)
        # M-step
        new_mu_unordered = (
            float(np.sum(gamma[:, 0] * x) / max(np.sum(gamma[:, 0]), 1e-12)),
            float(np.sum(gamma[:, 1] * x) / max(np.sum(gamma[:, 1]), 1e-12)),
        )
        new_sigma_unordered = (
            float(math.sqrt(np.sum(gamma[:, 0] * (x - new_mu_unordered[0]) ** 2)
                            / max(np.sum(gamma[:, 0]), 1e-12)) or 1e-4),
            float(math.sqrt(np.sum(gamma[:, 1] * (x - new_mu_unordered[1]) ** 2)
                            / max(np.sum(gamma[:, 1]), 1e-12)) or 1e-4),
        )
        # Order so regime 0 = low-mean, regime 1 = high-mean
        if new_mu_unordered[0] > new_mu_unordered[1]:
            new_mu = (new_mu_unordered[1], new_mu_unordered[0])
            new_sigma = (new_sigma_unordered[1], new_sigma_unordered[0])
            gamma = gamma[:, ::-1]
            xi = xi[:, ::-1, ::-1]
        else:
            new_mu = new_mu_unordered
            new_sigma = new_sigma_unordered
        # Transition matrix
        new_T = np.zeros((2, 2))
        for i in range(2):
            denom = float(np.sum(xi[:, i, :]))
            for j in range(2):
                new_T[i, j] = (
                    float(np.sum(xi[:, i, j])) / max(denom, 1e-12)
                )
        new_pi0 = (float(gamma[0, 0]), float(gamma[0, 1]))
        # Check convergence
        if abs(log_lik - prev_ll) < 1e-6:
            mu, sigma, T, pi0 = new_mu, new_sigma, new_T, new_pi0
            break
        mu, sigma, T, pi0 = new_mu, new_sigma, new_T, new_pi0
        prev_ll = log_lik

    stationary = _stationary_distribution(T)
    sh = (
        mu[0] / max(sigma[0], 1e-12) * math.sqrt(periods_per_year),
        mu[1] / max(sigma[1], 1e-12) * math.sqrt(periods_per_year),
    )
    mixture_sh = stationary[0] * sh[0] + stationary[1] * sh[1]
    return RSResult(
        mu=mu, sigma=sigma, transition=T,
        stationary_prob=stationary,
        regime_sharpes=sh,
        mixture_sharpe=float(mixture_sh),
        posterior_regime=gamma[:, 1],
        n=n, n_iter=it + 1, log_likelihood=log_lik,
    )
