"""Sweep market-price horizons. Train a 2-expert log-pool combiner
(stat_lgbm + market_at_horizon) for each available horizon, evaluate on a
held-out split, and report metrics so we can pick the time-to-close that
generalizes best.

The "best" combiner is saved to data/combiner.joblib for the runtime
CombinedSignaler. Default selection criterion: lowest held-out log-loss.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.signals.combiner import LogPoolCombiner, fit_weights, log_pool

log = logging_setup.configure()


HORIZON_COLS = ["p_market_1h", "p_market_6h", "p_market_24h", "p_market_7d"]


def _eval_split(P_tr, y_tr, P_te, y_te, expert_names):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    combiner = fit_weights(P_tr, y_tr, expert_names)
    p_combined = np.array([log_pool(P_te[i].tolist(), combiner.weights) for i in range(len(P_te))])
    p_combined = np.clip(p_combined, 1e-6, 1 - 1e-6)

    def _metrics(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return {
            "log_loss": float(log_loss(y_te, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y_te, p)),
            "auc": float(roc_auc_score(y_te, p)) if len(set(y_te)) > 1 else float("nan"),
        }

    return {
        "weights": [round(w, 4) for w in combiner.weights],
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "expert_metrics": {name: _metrics(P_te[:, i]) for i, name in enumerate(expert_names)},
        "combined_metrics": _metrics(p_combined),
        "combiner": combiner,
    }


def _load(horizon_col: str, db_path: str) -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"""SELECT yes_won, p_stat_lgbm, {horizon_col}
            FROM signal_outcomes
            WHERE p_stat_lgbm IS NOT NULL AND {horizon_col} IS NOT NULL"""
    ).fetchall()
    conn.close()
    if not rows:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    y = np.array([int(r[0]) for r in rows], dtype=int)
    P = np.array([[float(r[1]), float(r[2])] for r in rows], dtype=float)
    return P, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(settings.db_path).parent / "combiner.joblib"))
    p.add_argument(
        "--criterion",
        choices=["log_loss", "brier", "auc"],
        default="log_loss",
        help="Metric used to pick the best horizon (lower is better for log_loss/brier; higher for auc)",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    summary: list[dict] = []
    best_score = float("inf")
    best_combiner: LogPoolCombiner | None = None
    best_horizon: str | None = None
    best_expert_names: list[str] | None = None

    for horizon in HORIZON_COLS:
        P, y = _load(horizon, settings.db_path)
        if len(y) < 100:
            log.warning("horizon_too_few_rows", horizon=horizon, n=int(len(y)))
            continue
        idx = rng.permutation(len(y))
        n_te = int(len(y) * args.test_frac)
        test_idx, train_idx = idx[:n_te], idx[n_te:]
        P_tr, P_te, y_tr, y_te = P[train_idx], P[test_idx], y[train_idx], y[test_idx]
        expert_names = ["stat_lgbm", horizon]
        res = _eval_split(P_tr, y_tr, P_te, y_te, expert_names)
        log.info(
            "horizon_result",
            horizon=horizon,
            n_train=res["n_train"],
            n_test=res["n_test"],
            weights=res["weights"],
            stat_lgbm=res["expert_metrics"]["stat_lgbm"],
            market=res["expert_metrics"][horizon],
            combined=res["combined_metrics"],
        )
        summary.append({"horizon": horizon, **res})

        score = res["combined_metrics"][args.criterion]
        # AUC: higher is better — flip
        comparable = -score if args.criterion == "auc" else score
        if comparable < best_score:
            best_score = comparable
            best_combiner = res["combiner"]
            best_horizon = horizon
            best_expert_names = expert_names

    if best_combiner is None or best_horizon is None:
        log.error("no_horizon_qualified")
        raise SystemExit(2)

    log.info(
        "best_horizon",
        horizon=best_horizon,
        criterion=args.criterion,
        score=round(-best_score if args.criterion == "auc" else best_score, 4),
        weights=[round(w, 4) for w in best_combiner.weights],
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "weights": best_combiner.weights,
            "expert_names": best_expert_names,
            "horizon": best_horizon,
        },
        args.out,
    )
    log.info("combiner_saved", path=args.out)


if __name__ == "__main__":
    main()
