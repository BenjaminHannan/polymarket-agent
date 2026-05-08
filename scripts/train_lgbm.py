"""Train the LightGBM base-rate classifier with K-fold CV + isotonic calibration."""

from __future__ import annotations

import argparse

import structlog

from polyagent import logging_setup
from polyagent.models.lgbm import train

log = logging_setup.configure()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="output joblib path")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--objective",
        default="lambdarank",
        choices=["binary", "lambdarank"],
        help="lambdarank (default, Poh 2021 ~3x Sharpe) or binary",
    )
    args = p.parse_args()

    res = train(out_path=args.out, n_folds=args.folds, objective=args.objective)
    log.info(
        "lgbm_trained",
        n_total=res.n_total,
        cv_logloss=round(res.cv_logloss, 4),
        cv_brier=round(res.cv_brier, 4),
        cv_auc=round(res.cv_auc, 4),
        cv_ece_raw=round(res.cv_ece_raw, 4),
        cv_ece_calibrated=round(res.cv_ece_calibrated, 4),
        base_rate=round(res.base_rate, 4),
        model_path=res.model_path,
        calibrator_path=res.calibrator_path,
    )
    log.info("top_features", **res.feature_importance)


if __name__ == "__main__":
    main()
