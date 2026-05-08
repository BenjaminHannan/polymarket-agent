"""Fit log-pool weights over the experts in signal_outcomes.

Drops rows missing any of the chosen experts. Reports learned weights, log-loss,
and AUC vs each individual expert.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.signals.combiner import LogPoolCombiner, fit_weights, log_pool

log = logging_setup.configure()


def load(experts: list[str], db_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = {"stat_lgbm": "p_stat_lgbm", "news_match": "p_news_match", "market": "p_market_pre"}
    select = ", ".join(["yes_won"] + [cols[e] for e in experts])
    where = " AND ".join([f"{cols[e]} IS NOT NULL" for e in experts])
    conn = sqlite3.connect(db_path)
    rows = list(conn.execute(f"SELECT condition_id, {select} FROM signal_outcomes WHERE {where}"))
    conn.close()
    if not rows:
        return np.zeros((0, len(experts))), np.zeros(0, dtype=int), []
    cids = [r[0] for r in rows]
    y = np.array([int(r[1]) for r in rows], dtype=int)
    P = np.array([[float(r[2 + i]) for i in range(len(experts))] for r in rows], dtype=float)
    return P, y, cids


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = argparse.ArgumentParser()
    p.add_argument("--experts", nargs="+", default=["stat_lgbm"], choices=["stat_lgbm", "news_match", "market"])
    p.add_argument("--out", default=str(Path(settings.db_path).parent / "combiner.joblib"))
    args = p.parse_args()

    P, y, cids = load(args.experts, settings.db_path)
    if len(y) < 50:
        log.error("not_enough_rows", n=len(y), needed=50)
        raise SystemExit(2)

    combiner = fit_weights(P, y, args.expert_names if hasattr(args, "expert_names") else args.experts)
    weights = combiner.weights
    log.info(
        "combiner_fit_done",
        n=len(y),
        experts=combiner.expert_names,
        weights=[round(w, 4) for w in weights],
    )

    # Evaluate combined vs each individual expert
    combined = np.array([log_pool(P[i].tolist(), weights) for i in range(len(P))])
    ll_comb = float(log_loss(y, np.clip(combined, 1e-6, 1 - 1e-6), labels=[0, 1]))
    br_comb = float(brier_score_loss(y, combined))
    try:
        auc_comb = float(roc_auc_score(y, combined))
    except ValueError:
        auc_comb = float("nan")
    log.info(
        "combined_metrics",
        log_loss=round(ll_comb, 4),
        brier=round(br_comb, 4),
        auc=round(auc_comb, 4),
    )
    for i, name in enumerate(args.experts):
        col = P[:, i]
        ll = float(log_loss(y, np.clip(col, 1e-6, 1 - 1e-6), labels=[0, 1]))
        br = float(brier_score_loss(y, col))
        try:
            auc = float(roc_auc_score(y, col))
        except ValueError:
            auc = float("nan")
        log.info(f"expert_{name}", log_loss=round(ll, 4), brier=round(br, 4), auc=round(auc, 4))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"weights": weights, "expert_names": combiner.expert_names}, args.out)
    log.info("combiner_saved", path=args.out)


if __name__ == "__main__":
    main()
