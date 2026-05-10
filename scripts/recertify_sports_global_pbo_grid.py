"""Re-certify sports_global with an EX-ANTE config grid + honest PBO.

Per pmwhy.md §C3: "Stop running PBO with one config. Declare a small
ex-ante config grid (e.g., 5 reasonable hyperparam choices), run them,
report PBO, and freeze."

The original sports_global cert (DSR=0.996, 8/8 folds positive) was
computed with a single fixed configuration: equal-weight log-pool of
(p_stat_lgbm, p_market_24h). With one config, PBO degenerates to ~0.5
and is uninformative. This script declares 5 reasonable, plausible-but-
not-overfit configurations, runs CPCV across each, and reports the
proper Bailey-López de Prado PBO over the resulting paths.

Configs (declared up front, frozen after):

  C1: equal-weight log-pool [stat, mkt_24h]            (the original cert)
  C2: equal-weight log-pool [stat, mkt_6h]             (different mkt horizon)
  C3: 60/40 stat/mkt log-pool [stat, mkt_24h]          (higher stat weight)
  C4: 40/60 stat/mkt log-pool [stat, mkt_24h]          (lower stat weight)
  C5: equal-weight arithmetic-mean [stat, mkt_24h]     (different aggregator)

PBO is computed on the per-fold per-config matrix:
  - For each pair of CPCV paths, count whether the in-sample-best
    config underperformed median OOS on the other path.
  - Bailey-Borwein-López de Prado-Zhu 2014 formulation.

Output: per-config DSR, the best-config DSR with PBO context, and a
new strategy_certificates row recording the result.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from itertools import combinations

import numpy as np

DB = r"C:\Users\benja\Downloads\Polymarket\data\paper.db"
CATEGORY = "sports_global"
N_FOLDS = 8


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    if len(returns) < 5: return float("nan")
    from scipy.stats import skew, kurtosis, norm
    mu, sd = float(np.mean(returns)), float(np.std(returns, ddof=1))
    if sd <= 0: return float("nan")
    sr, T = mu / sd, len(returns)
    g3, g4 = float(skew(returns)), float(kurtosis(returns, fisher=False))
    sr0 = math.sqrt(2 * math.log(max(2, n_trials))) / math.sqrt(T)
    denom = math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def cpcv_splits(n: int, groups: np.ndarray, n_folds: int = N_FOLDS, seed: int = 42):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    fg = np.array_split(uniq, n_folds)
    out = []
    for k in range(n_folds):
        test = set(fg[k].tolist())
        m = np.array([g in test for g in groups], dtype=bool)
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def log_pool(p1, p2, w1):
    eps = 1e-9
    logit = lambda p: np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    s = w1 * logit(p1) + (1 - w1) * logit(p2)
    return 1.0 / (1.0 + np.exp(-s))


def arithmetic_mean(p1, p2, w1):
    return w1 * p1 + (1 - w1) * p2


# Frozen ex-ante config grid
CONFIGS = [
    {"id": "C1_equal_logpool_24h", "horizon": "24h", "w_stat": 0.5, "agg": "logpool"},
    {"id": "C2_equal_logpool_6h",  "horizon": "6h",  "w_stat": 0.5, "agg": "logpool"},
    {"id": "C3_high_stat_logpool", "horizon": "24h", "w_stat": 0.6, "agg": "logpool"},
    {"id": "C4_low_stat_logpool",  "horizon": "24h", "w_stat": 0.4, "agg": "logpool"},
    {"id": "C5_equal_arithmetic",  "horizon": "24h", "w_stat": 0.5, "agg": "arithmetic"},
]


def run_config(cfg, p_stat, p_mkt, y, cids):
    """Return per-fold edges (market_LL - combined_LL) for one config."""
    from sklearn.metrics import log_loss
    edges = []
    splits = cpcv_splits(len(y), cids, n_folds=N_FOLDS)
    for tr, te in splits:
        if len(np.unique(y[te])) < 2 or len(te) < 20:
            edges.append(np.nan)
            continue
        if cfg["agg"] == "logpool":
            p_te = log_pool(p_stat[te], p_mkt[te], cfg["w_stat"])
        else:
            p_te = arithmetic_mean(p_stat[te], p_mkt[te], cfg["w_stat"])
        comb_ll = float(log_loss(y[te], np.clip(p_te, 1e-6, 1-1e-6), labels=[0, 1]))
        mkt_ll = float(log_loss(y[te], p_mkt[te], labels=[0, 1]))
        edges.append(mkt_ll - comb_ll)
    return np.array(edges)


def pbo(score_matrix: np.ndarray) -> float:
    """Bailey-Borwein-Lopez de Prado-Zhu 2014 PBO.

    score_matrix[fold, config] = score of that config on that fold.
    For each pair of folds (i, j): identify the in-sample-best config
    on fold i, compare its OOS score on fold j to the median OOS
    score on fold j. PBO = fraction of pairs where the IS-best
    underperforms.
    """
    n_folds, n_cfgs = score_matrix.shape
    if n_folds < 2 or n_cfgs < 2:
        return float("nan")
    bad = total = 0
    for i, j in combinations(range(n_folds), 2):
        # Skip folds with NaN scores (insufficient test set)
        if np.any(np.isnan(score_matrix[i])) or np.any(np.isnan(score_matrix[j])):
            continue
        # IS-best config on fold i
        best_i = int(np.argmax(score_matrix[i]))
        oos_score_at_j = score_matrix[j, best_i]
        median_oos_at_j = float(np.median(score_matrix[j]))
        if oos_score_at_j < median_oos_at_j:
            bad += 1
        total += 1
        # Symmetric direction
        best_j = int(np.argmax(score_matrix[j]))
        oos_score_at_i = score_matrix[i, best_j]
        median_oos_at_i = float(np.median(score_matrix[i]))
        if oos_score_at_i < median_oos_at_i:
            bad += 1
        total += 1
    if total == 0:
        return float("nan")
    return bad / total


def main() -> None:
    horizon_to_col = {
        "24h": "p_market_24h",
        "6h":  "p_market_6h",
        "1h":  "p_market_1h",
        "pre": "p_market_pre",
    }
    # Pull all sports_global head-to-head rows with both horizons populated
    cols_needed = sorted({horizon_to_col[c["horizon"]] for c in CONFIGS} | {"p_stat_lgbm"})
    where_clause = " AND ".join(f"{c} IS NOT NULL" for c in cols_needed)
    sql_cols = ", ".join(["condition_id", "yes_won"] + cols_needed)
    sql = (
        f"SELECT {sql_cols} FROM signal_outcomes "
        f"WHERE category = ? AND {where_clause}"
    )
    conn = sqlite3.connect(DB)
    rows = conn.execute(sql, (CATEGORY,)).fetchall()
    n = len(rows)
    print(f"sports_global head-to-head rows (cols complete on all needed horizons): {n}")
    if n < 300:
        print("INSUFFICIENT DATA")
        return
    cids = np.array([r[0] for r in rows])
    y = np.array([int(r[1]) for r in rows])
    cols_idx = {c: 2 + i for i, c in enumerate(cols_needed)}
    p_stat = np.clip(np.array([float(r[cols_idx["p_stat_lgbm"]]) for r in rows]), 1e-3, 1-1e-3)

    print(f"\n{'config':30s}  {'mean_edge':>10s}  {'std':>8s}  {'pos':>5s}/{N_FOLDS}  {'DSR':>7s}")
    score_matrix = np.full((N_FOLDS, len(CONFIGS)), np.nan)
    config_results = []
    for c_idx, cfg in enumerate(CONFIGS):
        col = horizon_to_col[cfg["horizon"]]
        p_mkt = np.clip(
            np.array([float(r[cols_idx[col]]) for r in rows]),
            1e-3, 1-1e-3,
        )
        edges = run_config(cfg, p_stat, p_mkt, y, cids)
        score_matrix[:len(edges), c_idx] = edges
        valid = edges[~np.isnan(edges)]
        if len(valid) == 0:
            continue
        pos = int((valid > 0).sum())
        dsr = deflated_sharpe(valid, n_trials=len(CONFIGS))
        config_results.append({
            "id": cfg["id"],
            "mean_edge": float(valid.mean()),
            "std_edge": float(valid.std(ddof=1)),
            "pos_folds": pos,
            "fold_count": len(valid),
            "dsr": dsr,
        })
        print(f"  {cfg['id']:30s}  {valid.mean():+10.4f}  {valid.std(ddof=1):8.4f}  {pos:>5}/{len(valid)}  {dsr:7.4f}")

    # PBO across the grid (proper now)
    p_bo = pbo(score_matrix)
    print(f"\nPBO across {len(CONFIGS)}-config grid: {p_bo:.4f}")
    print("  PBO < 0.5 → in-sample-best generalises out-of-sample")
    print("  PBO >= 0.5 → overfit risk; the IS-best is no better than median OOS")

    best = max(config_results, key=lambda r: r["mean_edge"])
    pass_dsr = (not math.isnan(best["dsr"])) and best["dsr"] > 0.95
    pass_pbo = (not math.isnan(p_bo)) and p_bo < 0.5
    pass_edge = best["mean_edge"] > 0
    enabled = 1 if (pass_dsr and pass_pbo and pass_edge) else 0
    reason = (
        f"PASS: best={best['id']} DSR={best['dsr']:.4f} "
        f"edge={best['mean_edge']:+.4f} PBO={p_bo:.4f}"
        if enabled else
        f"FAIL: best={best['id']} dsr_pass={pass_dsr} "
        f"pbo_pass={pass_pbo} edge_pos={pass_edge} pbo={p_bo:.4f}"
    )
    verdict = {
        "enabled": enabled,
        "reason": reason,
        "category": CATEGORY,
        "n_holdout": n,
        "config_grid_size": len(CONFIGS),
        "configs": [c["id"] for c in CONFIGS],
        "per_config_results": config_results,
        "best_config": best["id"],
        "best_dsr": best["dsr"],
        "best_mean_edge": best["mean_edge"],
        "pbo_grid": p_bo,
    }
    conn.execute(
        """INSERT INTO strategy_certificates(name,enabled,dsr_holdout,n_holdout,issued_ts,detail)
           VALUES(?,?,?,?,?,?)""",
        (
            "stat_lgbm_combiner_sports_global_v4_pbo_grid",
            int(enabled),
            float(best["dsr"]) if not math.isnan(best["dsr"]) else None,
            int(n),
            time.time(),
            json.dumps(verdict),
        ),
    )
    conn.commit()
    print()
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
