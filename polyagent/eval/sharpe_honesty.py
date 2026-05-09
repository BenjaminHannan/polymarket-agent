"""Sharpe-honesty formulas — PSR, DSR, MTRL.

References:
  - Bailey & López de Prado 2012, "The Sharpe Ratio Efficient Frontier",
    J. Risk; SSRN 1821643. Probabilistic Sharpe Ratio.
  - Bailey & López de Prado 2014, "The Deflated Sharpe Ratio",
    J. Portfolio Management 40(5); SSRN 2460551. DSR + MTRL.
  - López de Prado, "Advances in Financial Machine Learning" (2018).
    Optimal Number of Clusters (ONC) for effective n_independent_trials.

These are the *honesty* metrics. They answer:

  - PSR(SR_benchmark): probability that the *true* Sharpe ratio is
    at least the benchmark, given the observed sample (corrected for
    skew + excess kurtosis).
  - DSR: PSR with the benchmark inflated to the expected maximum SR
    that would arise from N_trials independent strategies under H0
    (no real skill). A positive DSR means the observed SR is unlikely
    to be the artifact of multiple-testing.
  - MTRL: minimum sample size N at which PSR reaches a target (e.g.
    0.95). If you have fewer trades than MTRL, you cannot statistically
    distinguish your SR from the benchmark.

Pure-Python, scipy-only. No third-party stats package needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_EULER_MASCHERONI = 0.5772156649015329


def _moments(returns: np.ndarray) -> tuple[float, float, float, float, int]:
    """Return (mean, std, skew, excess_kurt, n) — biased-sample style
    matching de Prado's PSR derivation. Returns NaNs for n<3 or std==0."""
    arr = np.asarray(returns, dtype=float).ravel()
    n = arr.size
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return mu, 0.0, float("nan"), float("nan"), n
    z = (arr - mu) / sd
    skew = float(np.mean(z ** 3))
    # excess kurtosis (Fisher = excess over 3.0). Bailey/de Prado use
    # *excess* kurtosis throughout.
    kurt = float(np.mean(z ** 4) - 3.0)
    return mu, sd, skew, kurt, n


def sharpe_ratio(returns: np.ndarray) -> float:
    mu, sd, _, _, n = _moments(returns)
    if n < 2 or sd <= 0 or math.isnan(mu) or math.isnan(sd):
        return float("nan")
    return mu / sd


def psr(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & de Prado 2012).

      PSR(SR*) = Φ( (SR_obs − SR*) · √(n−1) /
                   √(1 − γ₃·SR_obs + (γ₄−1)/4·SR_obs²) )

    where γ₃ = sample skew, γ₄ = sample excess kurtosis (Fisher),
    SR_obs = mean/std of the return series.

    Returns the probability in [0, 1] that the *true* SR exceeds
    sr_benchmark. Returns NaN if the sample is too small or
    degenerate.
    """
    mu, sd, skew, ex_kurt, n = _moments(returns)
    if n < 5 or math.isnan(mu) or sd <= 0 or math.isnan(skew):
        return float("nan")
    sr_obs = mu / sd
    g3 = skew
    # de Prado uses γ4 = E[z^4] with kurt-shift; here ex_kurt is
    # already (E[z^4] − 3), so we add back to get γ4-style kurt.
    g4 = ex_kurt + 3.0
    denom_sq = 1.0 - g3 * sr_obs + ((g4 - 1.0) / 4.0) * sr_obs ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_obs - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(_normal_cdf(z))


def expected_max_sr(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum SR from N independent strategies under H0
    (true SR=0, all trials i.i.d. Normal(0, sr_trials_std²)).

    Bailey & de Prado 2014 Eq. (5):
      E[max_SR] = sd_SR · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)))

    where γ = Euler-Mascheroni ≈ 0.5772.
    """
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    inv1 = _normal_ppf(1.0 - 1.0 / n_trials)
    inv2 = _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_trials_std * ((1.0 - g) * inv1 + g * inv2)


def deflated_sharpe(
    returns: np.ndarray,
    sr_trials: list[float] | np.ndarray,
    n_independent_trials: int | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & de Prado 2014).

    DSR = PSR(expected_max_sr_under_H0).

    sr_trials: cross-sectional sample of SR values from variants of the
      strategy you tested. Used to estimate the sd of trial SRs.
    n_independent_trials: effective number of independent trials. If
      None, use len(sr_trials) — but in practice variants are
      correlated, so the López de Prado ONC clustering should be used
      to deflate this. We expose it as a parameter so the harness can
      plug in its own ONC estimate.
    """
    arr = np.asarray(sr_trials, dtype=float).ravel() if sr_trials is not None else np.array([])
    n_eff = (
        int(n_independent_trials) if n_independent_trials is not None
        else max(2, len(arr))
    )
    if arr.size < 2:
        # Fall back: if we have no trial dispersion estimate, use the
        # observed series's own SR-estimation error (one-trial DSR is
        # the same as PSR vs benchmark 0).
        return psr(returns, sr_benchmark=0.0)
    sr_trials_std = float(np.std(arr, ddof=1))
    sr_star = expected_max_sr(sr_trials_std, n_eff)
    return psr(returns, sr_benchmark=sr_star)


def mtrl(
    sr_estimate: float,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    sr_benchmark: float = 0.0,
    target_psr: float = 0.95,
) -> int:
    """Minimum Track Record Length (Bailey & de Prado 2012 Eq. 5):
    smallest n such that PSR(SR*) >= target_psr given the assumed
    higher moments.

    Inverts the PSR formula:
      n_min = 1 + (1 − γ₃·SR + (γ₄−1)/4·SR²) · (Φ⁻¹(target)/(SR−SR*))²
    """
    if math.isnan(sr_estimate) or sr_estimate <= sr_benchmark:
        return 10**9  # unreachable
    g3 = skew
    g4 = excess_kurt + 3.0
    inflate = 1.0 - g3 * sr_estimate + ((g4 - 1.0) / 4.0) * sr_estimate ** 2
    if inflate <= 0:
        return 10**9
    z = _normal_ppf(target_psr)
    n_min = 1.0 + inflate * (z / (sr_estimate - sr_benchmark)) ** 2
    return max(2, int(math.ceil(n_min)))


# --- normal CDF / inverse CDF (no scipy dependency) ----------------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Acklam's inverse normal CDF (rational approximation, ~5e-9 abs err)."""
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


@dataclass
class HonestyReport:
    n: int
    mean_r: float
    std_r: float
    skew: float
    excess_kurt: float
    sr: float
    psr_zero: float
    psr_skill: float
    dsr: float
    mtrl_zero_95: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_r": round(self.mean_r, 6),
            "std_r": round(self.std_r, 6),
            "skew": round(self.skew, 4),
            "excess_kurt": round(self.excess_kurt, 4),
            "sr": round(self.sr, 4),
            "psr_zero": round(self.psr_zero, 4),
            "psr_skill": round(self.psr_skill, 4),
            "dsr": round(self.dsr, 4),
            "mtrl_zero_95": int(self.mtrl_zero_95),
        }


def report(
    returns: np.ndarray,
    sr_skill_benchmark: float = 0.5,
    sr_trials: list[float] | None = None,
    n_independent_trials: int | None = None,
) -> HonestyReport:
    """Convenience: run all four formulas at once."""
    mu, sd, sk, ek, n = _moments(returns)
    sr = mu / sd if (sd and sd > 0 and not math.isnan(mu)) else float("nan")
    return HonestyReport(
        n=n,
        mean_r=mu if not math.isnan(mu) else 0.0,
        std_r=sd if (sd and not math.isnan(sd)) else 0.0,
        skew=sk if not math.isnan(sk) else 0.0,
        excess_kurt=ek if not math.isnan(ek) else 0.0,
        sr=sr if not math.isnan(sr) else 0.0,
        psr_zero=psr(returns, sr_benchmark=0.0),
        psr_skill=psr(returns, sr_benchmark=sr_skill_benchmark),
        dsr=deflated_sharpe(returns, sr_trials or [], n_independent_trials),
        mtrl_zero_95=mtrl(sr, sk if not math.isnan(sk) else 0.0,
                          ek if not math.isnan(ek) else 0.0,
                          sr_benchmark=0.0, target_psr=0.95),
    )
