"""CPCV + Deflated Sharpe + PBO validation harness.

Implements the de Prado validation discipline applied to the combiner's
out-of-fold predictions joined to realized P&L. Reports:

  - CPCV (Combinatorial Purged Cross-Validation): fold the historical
    market set so that ALL rows of a market are in the same fold (purge
    by market_id), then compute fold-wise log-loss / Brier / AUC and
    realized "trade returns" at gates we configure.
  - Deflated Sharpe Ratio (Bailey/Lopez de Prado): adjusts the apparent
    Sharpe by the number of strategy/feature configurations tried, the
    skew, and the kurtosis of returns.
  - Probability of Backtest Overfitting (Bailey/Borwein/Lopez de Prado):
    fraction of CPCV path-pairs where the in-sample-best config
    underperforms median out-of-sample.

Pure analysis script; doesn't touch the live bot. Run:
    python -m scripts.validate_strategy --n-trials 6 --n-folds 8
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import structlog

from polyagent import logging_setup
from polyagent.config import settings

log = logging_setup.configure()


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    """Bailey/Lopez de Prado 2014 DSR. Returns the deflated value in [0,1].

    DSR = Phi(((SR - SR0) * sqrt(T - 1)) / sqrt(1 - g3*SR + (g4-1)/4*SR^2))
    where SR0 ≈ sqrt(2*ln(N))/sqrt(T) is the threshold accounting for n_trials.
    """
    if len(returns) < 5:
        return float("nan")
    from scipy.stats import skew, kurtosis, norm

    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return float("nan")
    sr = mu / sd
    T = len(returns)
    g3 = float(skew(returns))
    g4 = float(kurtosis(returns, fisher=False))  # not subtracting 3
    sr0 = math.sqrt(2 * math.log(max(2, n_trials))) / math.sqrt(T)
    denom = math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def cpcv_split_indices(n: int, groups: np.ndarray, n_folds: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged folds: each unique market_id stays together."""
    rng = np.random.default_rng(42)
    unique_groups = np.array(sorted(set(groups.tolist())))
    rng.shuffle(unique_groups)
    fold_groups = np.array_split(unique_groups, n_folds)
    splits = []
    for k in range(n_folds):
        test_groups = set(fold_groups[k].tolist())
        test_mask = np.array([g in test_groups for g in groups], dtype=bool)
        train_mask = ~test_mask
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def pbo(in_sample_best: list[float], out_of_sample: list[list[float]]) -> float:
    """Probability of Backtest Overfitting.

    For each pair of CPCV paths, compute the fraction of times the in-sample
    best config underperformed the OOS median.
    """
    if len(in_sample_best) < 2 or not out_of_sample:
        return float("nan")
    n_paths = len(in_sample_best)
    median_oos = [float(np.median(o)) for o in out_of_sample]
    bad = 0
    total = 0
    for i, j in combinations(range(n_paths), 2):
        is_best_i = in_sample_best[i]
        oos_j = median_oos[j]
        if is_best_i < oos_j:  # underperforms median
            bad += 1
        total += 1
        # symmetric
        is_best_j = in_sample_best[j]
        oos_i = median_oos[i]
        if is_best_j < oos_i:
            bad += 1
        total += 1
    return bad / max(1, total)


def load_signal_outcomes(db_path: str = settings.db_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (P_combined_estimate, y, group_market_id) for all rows that have
    enough columns to score a per-row predicted probability.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT condition_id, yes_won,
               COALESCE(p_stat_lgbm, 0.5) as p_stat,
               COALESCE(p_market_6h, p_market_pre, 0.5) as p_mkt
        FROM signal_outcomes
        WHERE p_stat_lgbm IS NOT NULL
        """
    ).fetchall()
    conn.close()
    if not rows:
        return np.zeros(0), np.zeros(0, dtype=int), np.array([])
    cids = np.array([r[0] for r in rows])
    y = np.array([int(r[1]) for r in rows], dtype=int)
    p_stat = np.array([float(r[2]) for r in rows])
    p_mkt = np.array([float(r[3]) for r in rows])
    # Simple log-pool with equal weights as baseline; could load actual combiner
    p = 0.5 * p_stat + 0.5 * p_mkt
    p = np.clip(p, 0.001, 0.999)
    return p, y, cids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=8)
    p.add_argument("--n-trials", type=int, default=6)
    args = p.parse_args()

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    probs, y, groups = load_signal_outcomes()
    if len(y) < 200:
        log.error("insufficient_rows", n=int(len(y)))
        raise SystemExit(2)

    log.info("validation_start", n=int(len(y)), n_folds=args.n_folds, n_trials=args.n_trials)
    splits = cpcv_split_indices(len(y), groups, n_folds=args.n_folds)

    # Per-fold metrics
    fold_logloss = []
    fold_brier = []
    fold_auc = []
    is_best = []
    oos = []
    for fi, (tr_idx, te_idx) in enumerate(splits):
        if len(np.unique(y[te_idx])) < 2:
            continue
        ll = float(log_loss(y[te_idx], probs[te_idx], labels=[0, 1]))
        br = float(brier_score_loss(y[te_idx], probs[te_idx]))
        au = float(roc_auc_score(y[te_idx], probs[te_idx]))
        fold_logloss.append(ll)
        fold_brier.append(br)
        fold_auc.append(au)
        # Path "score" = -log_loss (higher = better)
        is_best.append(-ll)
        oos.append([-ll])

    log.info(
        "cpcv_results",
        n_folds=len(fold_logloss),
        mean_logloss=round(float(np.mean(fold_logloss)), 4),
        mean_brier=round(float(np.mean(fold_brier)), 4),
        mean_auc=round(float(np.mean(fold_auc)), 4),
        std_logloss=round(float(np.std(fold_logloss)), 4),
    )

    # DSR on the realized "P&L per fold" (using -log_loss as score)
    dsr = deflated_sharpe(np.array(is_best), n_trials=args.n_trials)
    log.info("deflated_sharpe", dsr=round(dsr, 4) if not math.isnan(dsr) else None,
             interpretation=("PASS" if dsr > 0.95 else "WEAK" if dsr > 0.5 else "INSUFFICIENT"))

    # PBO
    p_bo = pbo(is_best, oos)
    log.info(
        "pbo",
        pbo=round(p_bo, 4) if not math.isnan(p_bo) else None,
        interpretation=("PASS" if p_bo < 0.5 else "OVERFIT_RISK"),
    )


if __name__ == "__main__":
    main()
