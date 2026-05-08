"""Train per-category log-pool combiners.

For each category with at least --min-rows labeled outcomes, fit a 2-expert
combiner on a stratified train/test split using the chosen market-price
horizon. Save a v2 bundle:

    {
      "version": 2,
      "horizon": "p_market_6h",
      "default": {"weights": [...], "expert_names": ["stat_lgbm", "p_market_6h"]},
      "by_category": {
          "sports_us": {"weights": [...], "expert_names": [...]},
          ...
      },
      "metrics": {category: {"n_train", "n_test", "log_loss", "auc"}, ...}
    }

The CombinedSignaler picks `by_category[cat]` if present, else `default`.
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
from polyagent.models.categorize import categorize
from polyagent.models.outcomes import _ensure_table
from polyagent.signals.combiner import LogPoolCombiner, fit_weights, log_pool

log = logging_setup.configure()


def _backfill_category_column(db_path: str) -> int:
    """Populate category for any rows where it's NULL."""
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)  # idempotent: adds the column if missing
    rows = conn.execute(
        "SELECT condition_id, question FROM signal_outcomes WHERE category IS NULL"
    ).fetchall()
    n = 0
    for cid, q in rows:
        cat = categorize(q or "")
        conn.execute("UPDATE signal_outcomes SET category = ? WHERE condition_id = ?", (cat, cid))
        n += 1
    conn.commit()
    conn.close()
    return n


_EXPERT_TO_COLUMN = {
    "stat_lgbm": "p_stat_lgbm",
    "news_match": "p_news_match",
    "market": "p_market_pre",
    "p_market_1h": "p_market_1h",
    "p_market_6h": "p_market_6h",
    "p_market_24h": "p_market_24h",
    "p_market_7d": "p_market_7d",
}


def _load_for(
    category: str | None,
    expert_names: list[str],
    db_path: str,
    *,
    with_ts: bool = False,
):
    cols = [_EXPERT_TO_COLUMN[e] for e in expert_names]
    select = ", ".join(["yes_won"] + cols + (["resolved_ts"] if with_ts else []))
    where_parts = [f"{c} IS NOT NULL" for c in cols]
    if category is not None:
        where_parts.append("category = ?")
    where = " AND ".join(where_parts)
    conn = sqlite3.connect(db_path)
    sql = f"SELECT {select} FROM signal_outcomes WHERE {where}"
    if with_ts:
        sql += " ORDER BY resolved_ts ASC"
    if category is None:
        rows = conn.execute(sql).fetchall()
    else:
        rows = conn.execute(sql, (category,)).fetchall()
    conn.close()
    if not rows:
        empty_p = np.zeros((0, len(expert_names)))
        empty_y = np.zeros(0, dtype=int)
        if with_ts:
            return empty_p, empty_y, np.zeros(0)
        return empty_p, empty_y
    y = np.array([int(r[0]) for r in rows], dtype=int)
    P = np.array([[float(r[i + 1]) for i in range(len(expert_names))] for r in rows], dtype=float)
    if with_ts:
        ts = np.array(
            [float(r[len(expert_names) + 1] or 0.0) for r in rows], dtype=float
        )
        return P, y, ts
    return P, y


def _eval_forward(
    P_tr: np.ndarray,
    y_tr: np.ndarray,
    P_fw: np.ndarray,
    y_fw: np.ndarray,
    expert_names: list[str],
) -> dict | None:
    """Train weights on (P_tr, y_tr); evaluate combined log-loss on (P_fw, y_fw)."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    if len(y_tr) == 0 or len(y_fw) == 0:
        return None
    if len(set(y_tr)) < 2 or len(set(y_fw)) < 2:
        return None
    combiner = fit_weights(P_tr, y_tr, expert_names)
    combined = np.array(
        [log_pool(P_fw[i].tolist(), combiner.weights) for i in range(len(P_fw))]
    )
    combined = np.clip(combined, 1e-6, 1 - 1e-6)
    return {
        "n_forward": int(len(y_fw)),
        "n_train": int(len(y_tr)),
        "log_loss": float(log_loss(y_fw, combined, labels=[0, 1])),
        "brier": float(brier_score_loss(y_fw, combined)),
        "auc": float(roc_auc_score(y_fw, combined)),
    }


def _train_one(
    P: np.ndarray, y: np.ndarray, expert_names: list[str], test_frac: float, seed: int
) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_te = max(1, int(len(y) * test_frac))
    test_idx, train_idx = idx[:n_te], idx[n_te:]
    P_tr, P_te = P[train_idx], P[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"skip": True}
    combiner = fit_weights(P_tr, y_tr, expert_names)
    combined = np.array([log_pool(P_te[i].tolist(), combiner.weights) for i in range(len(P_te))])
    combined = np.clip(combined, 1e-6, 1 - 1e-6)
    return {
        "weights": [float(w) for w in combiner.weights],
        "expert_names": expert_names,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "metrics": {
            "log_loss": float(log_loss(y_te, combined, labels=[0, 1])),
            "brier": float(brier_score_loss(y_te, combined)),
            "auc": float(roc_auc_score(y_te, combined)) if len(set(y_te)) > 1 else float("nan"),
        },
    }


def count_full_rows(experts: list[str], db_path: str = settings.db_path) -> int:
    """Count signal_outcomes rows where every expert column is non-null."""
    cols = [_EXPERT_TO_COLUMN[e] for e in experts]
    where = " AND ".join([f"{c} IS NOT NULL" for c in cols])
    conn = sqlite3.connect(db_path)
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM signal_outcomes WHERE {where}").fetchone()[0]
    except sqlite3.OperationalError:
        n = 0
    finally:
        conn.close()
    return int(n)


def _check_regression(
    new_bundle: dict,
    old_path: str,
    tolerance: float,
) -> tuple[bool, str, dict]:
    """Compare new bundle's log-loss to the old bundle's.

    Prefers `forward_metrics.log_loss` (chronological holdout — catches
    distribution shift) when both bundles have it; falls back to
    `_default.log_loss` (random split) otherwise.

    Returns (passes, reason, info). passes=True -> safe to swap.
    """
    info: dict = {}
    if not Path(old_path).exists():
        return True, "no_baseline_bundle", info
    try:
        old_bundle = joblib.load(old_path)
    except Exception as e:
        return True, f"old_load_failed:{e}", info

    old_fwd = old_bundle.get("forward_metrics") or {}
    new_fwd = new_bundle.get("forward_metrics") or {}
    if "log_loss" in old_fwd and "log_loss" in new_fwd:
        info["metric"] = "forward_log_loss"
        info["old_logloss"] = old_fwd["log_loss"]
        info["new_logloss"] = new_fwd["log_loss"]
        if new_fwd["log_loss"] > old_fwd["log_loss"] + tolerance:
            return False, "forward_logloss_regressed", info
        return True, "forward_logloss_ok", info

    # Fallback: random-split metric
    old_m = (old_bundle.get("metrics") or {}).get("_default") or {}
    new_m = (new_bundle.get("metrics") or {}).get("_default") or {}
    old_ll = old_m.get("log_loss")
    new_ll = new_m.get("log_loss")
    info["metric"] = "default_log_loss"
    info["old_logloss"] = old_ll
    info["new_logloss"] = new_ll
    if old_ll is None and new_ll is None:
        return True, "no_metrics_to_compare", info
    if old_ll is None:
        return True, "no_old_metric_first_train", info
    if new_ll is None:
        return False, "new_bundle_missing_default_metric", info
    if new_ll > old_ll + tolerance:
        return False, "logloss_regressed", info
    return True, "logloss_ok", info


def run_pipeline(
    *,
    experts: list[str],
    horizon: str,
    min_rows: int,
    test_frac: float,
    seed: int,
    out_path: str,
    db_path: str = settings.db_path,
    regression_tolerance: float = 0.0,
    allow_regression: bool = False,
    forward_holdout_k: int = 50,
) -> dict | None:
    """Programmatic entrypoint. Returns the candidate bundle (saved or rejected),
    or None if training failed (insufficient data).

    Quality gate: after training, compares the new bundle's default log-loss to
    the existing on-disk bundle's. If new > old + tolerance and
    `allow_regression` is False, the swap is blocked, the candidate is returned
    with `regression_blocked=True`, and the on-disk bundle is unchanged.

    Atomic swap on success: writes to `<out>.tmp`, then os.replace -> `<out>`.
    """
    import os

    unknown = [e for e in experts if e not in _EXPERT_TO_COLUMN]
    if unknown:
        log.error("unknown_experts", names=unknown, valid=list(_EXPERT_TO_COLUMN))
        return None

    n_filled = _backfill_category_column(db_path)
    if n_filled:
        log.info("category_backfilled", rows=n_filled)

    n_full_rows = count_full_rows(experts, db_path)

    conn = sqlite3.connect(db_path)
    cats = [r[0] for r in conn.execute(
        "SELECT category, COUNT(*) FROM signal_outcomes GROUP BY category ORDER BY 2 DESC"
    ).fetchall()]
    conn.close()

    bundle: dict = {
        "version": 2,
        "horizon": horizon,
        "default": None,
        "by_category": {},
        "metrics": {},
        "experts": experts,
        "n_full_rows": n_full_rows,
        "trained_ts": __import__("time").time(),
    }

    P, y = _load_for(None, experts, db_path)
    if len(y) >= min_rows:
        res = _train_one(P, y, experts, test_frac, seed)
        if not res.get("skip"):
            bundle["default"] = {"weights": res["weights"], "expert_names": res["expert_names"]}
            bundle["metrics"]["_default"] = {
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                **res["metrics"],
            }
            log.info(
                "default_trained",
                n=int(len(y)),
                weights=[round(w, 3) for w in res["weights"]],
                **{k: round(v, 4) for k, v in res["metrics"].items()},
            )

    # Forward-time holdout: chronologically-newest K rows held out entirely.
    # Train on the rest, evaluate combined log-pool predictions on the held-out
    # forward set. Uses ts-ordered load.
    if forward_holdout_k > 0:
        P_ts, y_ts, _ts = _load_for(None, experts, db_path, with_ts=True)
        if len(y_ts) >= min_rows + forward_holdout_k:
            P_tr, P_fw = P_ts[:-forward_holdout_k], P_ts[-forward_holdout_k:]
            y_tr, y_fw = y_ts[:-forward_holdout_k], y_ts[-forward_holdout_k:]
            fwd = _eval_forward(P_tr, y_tr, P_fw, y_fw, experts)
            if fwd is not None:
                bundle["forward_metrics"] = fwd
                log.info(
                    "forward_eval",
                    n_forward=fwd["n_forward"],
                    n_train=fwd["n_train"],
                    log_loss=round(fwd["log_loss"], 4),
                    brier=round(fwd["brier"], 4),
                    auc=round(fwd["auc"], 4),
                )
        else:
            log.info(
                "forward_eval_skipped_insufficient_rows",
                n=int(len(y_ts)),
                needed=min_rows + forward_holdout_k,
            )

    for cat in cats:
        if not cat:
            continue
        P, y = _load_for(cat, experts, db_path)
        if len(y) < min_rows:
            log.info("category_skipped_small", category=cat, n=int(len(y)))
            continue
        res = _train_one(P, y, experts, test_frac, seed)
        if res.get("skip"):
            log.info("category_skipped_imbalanced", category=cat, n=int(len(y)))
            continue
        bundle["by_category"][cat] = {
            "weights": res["weights"],
            "expert_names": res["expert_names"],
        }
        bundle["metrics"][cat] = {
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            **res["metrics"],
        }
        log.info(
            "category_trained",
            category=cat,
            n=int(len(y)),
            weights=[round(w, 3) for w in res["weights"]],
            **{k: round(v, 4) for k, v in res["metrics"].items()},
        )

    if bundle["default"] is None:
        log.error("no_default_combiner")
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path + ".tmp"
    joblib.dump(bundle, tmp_path)

    passes, reason, info = _check_regression(bundle, out_path, regression_tolerance)
    bundle["regression_check"] = {"passes": passes, "reason": reason, **info}
    if not passes and not allow_regression:
        bundle["regression_blocked"] = True
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass
        log.warning(
            "retrain_regression_blocked",
            reason=reason,
            old_logloss=info.get("old_logloss"),
            new_logloss=info.get("new_logloss"),
            tolerance=regression_tolerance,
        )
        return bundle

    bundle["regression_blocked"] = False
    os.replace(tmp_path, out_path)
    return bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", default="p_market_6h")
    p.add_argument(
        "--experts",
        nargs="+",
        default=None,
        help="Expert column names (e.g. stat_lgbm news_match p_market_6h). "
        "Defaults to [stat_lgbm, <horizon>].",
    )
    p.add_argument("--min-rows", type=int, default=150)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(settings.db_path).parent / "combiner.joblib"))
    p.add_argument("--regression-tolerance", type=float, default=0.0)
    p.add_argument("--allow-regression", action="store_true")
    p.add_argument("--forward-holdout-k", type=int, default=50)
    args = p.parse_args()

    horizon = args.horizon
    experts = args.experts or ["stat_lgbm", horizon]

    bundle = run_pipeline(
        experts=experts,
        horizon=horizon,
        min_rows=args.min_rows,
        test_frac=args.test_frac,
        seed=args.seed,
        out_path=args.out,
        regression_tolerance=args.regression_tolerance,
        allow_regression=args.allow_regression,
        forward_holdout_k=args.forward_holdout_k,
    )
    if bundle is None:
        raise SystemExit(2)
    if bundle.get("regression_blocked"):
        log.error(
            "retrain_blocked_by_quality_gate",
            old_logloss=bundle["regression_check"].get("old_logloss"),
            new_logloss=bundle["regression_check"].get("new_logloss"),
        )
        raise SystemExit(3)
    log.info(
        "combiner_saved",
        path=args.out,
        n_categories=len(bundle["by_category"]),
        horizon=horizon,
    )


if __name__ == "__main__":
    main()
