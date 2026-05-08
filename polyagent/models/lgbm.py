"""LightGBM binary classifier for P(YES | features) with isotonic calibration.

Pipeline:
  1. Load resolutions table -> question features + label.
  2. K-fold CV: train K models, gather out-of-fold raw predictions.
  3. Fit isotonic calibrator on the OOF predictions vs labels.
  4. Train one final model on all data; save model + calibrator + base rate.

At inference (Predictor):
  raw = model.predict_proba(features)[:,1]
  cal = calibrator.transform(raw)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog

from polyagent.config import settings
from polyagent.models.calibrator import (
    IsotonicCalibrator,
    CellCalibrator,
    calibration_metrics,
    horizon_bucket,
)
from polyagent.models.categorize import categorize
from polyagent.models.embedder import (
    embed_features as emb_features,
    embed_features_batch as emb_features_batch,
)
from polyagent.models.features import question_features

log = structlog.get_logger()


def _parse_iso(end_date: str | None) -> float | None:
    if not end_date:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _load_dataset(db_path: str = settings.db_path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = list(
        conn.execute("SELECT condition_id, yes_won, detail, resolved_ts FROM resolutions")
    )
    conn.close()
    records = []
    questions = []
    for r in rows:
        try:
            d = json.loads(r["detail"] or "{}")
        except json.JSONDecodeError:
            continue
        question = d.get("question") or ""
        if not question:
            continue
        feats = question_features(
            question,
            liquidity=d.get("liquidity") or 0.0,
            volume=d.get("volume") or 0.0,
            days_to_resolution=None,
        )
        feats["yes_won"] = int(r["yes_won"])
        feats["condition_id"] = r["condition_id"]
        feats["question"] = question  # kept for per-cell calibration; dropped before predict
        # Day-of-resolution group key for LambdaRank (Poh et al. 2021 found
        # ~3x Sharpe lift vs pointwise; the ranker only needs to get
        # cross-market relative-edge ordering right within each query group).
        try:
            ts = float(r["resolved_ts"] or 0.0)
        except Exception:
            ts = 0.0
        if ts > 0:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            feats["_group_day"] = dt.strftime("%Y-%m-%d")
        else:
            feats["_group_day"] = "unknown"
        records.append(feats)
        questions.append(question)
    # Add 384-dim sentence embeddings in batched GPU calls.
    if questions:
        log.info("embedding_dataset", n=len(questions))
        emb_feats = emb_features_batch(questions)
        for rec, ef in zip(records, emb_feats):
            rec.update(ef)
    return pd.DataFrame.from_records(records)


def _fit_one(X_tr, y_tr, X_val, y_val, seed: int = 42, *, group_tr=None, group_val=None, objective: str = "binary"):
    """Tuned LightGBM trainer.

    objective='binary' (default): standard pointwise classifier.
    objective='lambdarank': listwise; requires `group_tr` / `group_val` arrays
    (sizes per query). Predictions are scores, not probabilities — feed
    through isotonic/cell calibration to get probs.
    """
    import lightgbm as lgb

    if objective == "lambdarank":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=30,
            feature_fraction=0.7,
            bagging_fraction=0.85,
            bagging_freq=5,
            reg_lambda=2.0,
            reg_alpha=0.5,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(
            X_tr, y_tr,
            group=group_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val] if group_val is not None else None,
        )
        return model

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        feature_fraction=0.7,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=seed,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    return model


@dataclass
class TrainResult:
    model_path: str
    calibrator_path: str
    n_total: int
    base_rate: float
    cv_logloss: float
    cv_brier: float
    cv_auc: float
    cv_ece_raw: float
    cv_ece_calibrated: float
    feature_importance: dict[str, float]


def _cpcv_split_indices(n: int, groups: np.ndarray, n_folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged folds: every row sharing a `groups` key stays
    together (de Prado). Avoids the leak where a market's sibling
    outcomes (or its same-day correlated event) end up in train+test.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups.tolist())))
    rng.shuffle(unique_groups)
    fold_groups = np.array_split(unique_groups, n_folds)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_groups = set(fold_groups[k].tolist())
        test_mask = np.array([g in test_groups for g in groups], dtype=bool)
        train_mask = ~test_mask
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def _group_sizes(group_keys: np.ndarray) -> np.ndarray:
    """Convert a sorted array of group keys into the per-group counts that
    LightGBM's ranker expects."""
    if len(group_keys) == 0:
        return np.zeros(0, dtype=int)
    sizes = []
    cur = group_keys[0]
    n = 1
    for k in group_keys[1:]:
        if k == cur:
            n += 1
        else:
            sizes.append(n)
            cur = k
            n = 1
    sizes.append(n)
    return np.asarray(sizes, dtype=int)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train(
    *,
    db_path: str = settings.db_path,
    out_path: str | None = None,
    n_folds: int = 5,
    seed: int = 42,
    objective: str = "lambdarank",
) -> TrainResult:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if out_path is None:
        out_path = str(Path(db_path).parent / "lgbm_model.joblib")
    cal_path = str(Path(out_path).with_suffix("")) + "_calibrator.joblib"

    df = _load_dataset(db_path)
    if len(df) < 200:
        raise RuntimeError(f"need >=200 resolutions to train with CV; have {len(df)}")

    feature_cols = [
        c for c in df.columns
        if c not in ("yes_won", "condition_id", "question", "_group_day")
    ]
    X = df[feature_cols].astype(float).values
    y = df["yes_won"].astype(int).values
    base_rate = float(y.mean())
    group_keys_all = df["_group_day"].astype(str).values if "_group_day" in df.columns else None

    # If we asked for lambdarank but every row falls in a single group (or
    # group key is missing), the listwise loss collapses to nonsense — fall
    # back to binary in that case.
    if objective == "lambdarank":
        if group_keys_all is None or len(set(group_keys_all)) < 2:
            log.warning("lambdarank_groups_unusable_fallback_to_binary")
            objective = "binary"

    # K-fold OOF predictions for calibration.
    # CPCV purged by day-of-resolution group key (de Prado): keeps all
    # markets resolving on the same day in the same fold, blocking the
    # NegRisk-sibling leak that vanilla StratifiedKFold permits. Falls
    # back to StratifiedKFold when we don't have enough distinct groups.
    n_distinct_groups = len(set(group_keys_all.tolist())) if group_keys_all is not None else 0
    use_cpcv = group_keys_all is not None and n_distinct_groups >= max(2 * n_folds, 8)
    if use_cpcv:
        splits = _cpcv_split_indices(len(y), group_keys_all, n_folds=n_folds, seed=seed)
        log.info("cv_strategy", strategy="cpcv", n_distinct_groups=n_distinct_groups, n_folds=n_folds)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        log.info(
            "cv_strategy",
            strategy="stratified_kfold_fallback",
            n_distinct_groups=n_distinct_groups,
            n_folds=n_folds,
        )
    oof = np.zeros(len(y), dtype=float)
    fold_models = []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        if objective == "lambdarank":
            # Sort each fold by group key so LightGBM sees grouped queries.
            tr_order = np.argsort(group_keys_all[tr_idx], kind="stable")
            va_order = np.argsort(group_keys_all[va_idx], kind="stable")
            X_tr = X[tr_idx][tr_order]
            y_tr = y[tr_idx][tr_order]
            X_va = X[va_idx][va_order]
            y_va = y[va_idx][va_order]
            g_tr = _group_sizes(group_keys_all[tr_idx][tr_order])
            g_va = _group_sizes(group_keys_all[va_idx][va_order])
            m = _fit_one(
                X_tr, y_tr, X_va, y_va,
                seed=seed + fi,
                group_tr=g_tr, group_val=g_va,
                objective="lambdarank",
            )
            # Ranker outputs raw scores; squash to (0,1) with sigmoid so the
            # downstream isotonic/Beta calibrators see probability-shaped input.
            scores = m.predict(X_va)
            probs = _sigmoid(np.asarray(scores, dtype=float))
            # Unsort back into original va_idx order.
            inv = np.argsort(va_order)
            oof[va_idx] = probs[inv]
        else:
            m = _fit_one(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], seed=seed + fi)
            oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        fold_models.append(m)

    cv_ll = float(log_loss(y, oof, labels=[0, 1]))
    cv_brier = float(brier_score_loss(y, oof))
    try:
        cv_auc = float(roc_auc_score(y, oof))
    except ValueError:
        cv_auc = float("nan")

    # Fit calibrator on OOF predictions.
    # Per-(category × horizon) calibration per Le 2026 — addresses the 26%
    # of calibration variance explained by domain×horizon interactions.
    # Falls back to global isotonic for cells with too few samples.
    cell_cal = CellCalibrator()
    # Derive category + horizon bucket per row from the question + TTR cols
    # (we don't store category in df; recompute here)
    cats = np.array([categorize(q) for q in df.get("question", [""] * len(y)).tolist()]) if "question" in df.columns else np.array(["other"] * len(y))
    # If we don't have question text, just use 'other' for everything
    if "question" not in df.columns:
        cats = np.array(["other"] * len(y))
    horizons = np.array(["unknown"] * len(y))  # historical TTR not reliably stored
    try:
        cell_summary = cell_cal.fit(oof, y, cats, horizons, min_per_cell=80)
        log.info("cell_calibrator_fit", cells=cell_summary)
    except Exception as e:
        log.warning("cell_calibrator_fit_failed", err=str(e))

    # Also keep the old global isotonic so downstream code that still expects
    # a single calibrator can use it.
    cal = IsotonicCalibrator()
    cal.fit(oof, y)
    cal_oof = np.asarray(cal.transform(oof))
    ece_raw = calibration_metrics(oof, y)["ece"]
    ece_cal = calibration_metrics(cal_oof, y)["ece"]

    # Final model on all data (no holdout — calibration is from CV).
    if objective == "lambdarank":
        final_order = np.argsort(group_keys_all, kind="stable")
        X_all = X[final_order]
        y_all = y[final_order]
        g_all = _group_sizes(group_keys_all[final_order])
        final = _fit_one(
            X_all, y_all, X_all, y_all,
            seed=seed,
            group_tr=g_all, group_val=g_all,
            objective="lambdarank",
        )
    else:
        final = _fit_one(X, y, X, y, seed=seed)

    importances = final.feature_importances_
    fi = dict(sorted(zip(feature_cols, [float(x) for x in importances]), key=lambda kv: -kv[1]))

    bundle = {
        "model": final,
        "objective": objective,            # used by Predictor to dispatch
        "feature_cols": feature_cols,
        "base_rate": base_rate,
        "cv_metrics": {
            "logloss": cv_ll,
            "brier": cv_brier,
            "auc": cv_auc,
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "n_folds": n_folds,
            "n_total": len(y),
        },
    }
    joblib.dump(bundle, out_path)
    cal.save(cal_path)
    cell_cal_path = str(Path(out_path).with_suffix("")) + "_cell_calibrator.joblib"
    try:
        cell_cal.save(cell_cal_path)
    except Exception as e:
        log.warning("cell_calibrator_save_failed", err=str(e))

    return TrainResult(
        model_path=out_path,
        calibrator_path=cal_path,
        n_total=len(y),
        base_rate=base_rate,
        cv_logloss=cv_ll,
        cv_brier=cv_brier,
        cv_auc=cv_auc,
        cv_ece_raw=ece_raw,
        cv_ece_calibrated=ece_cal,
        feature_importance={k: v for k, v in list(fi.items())[:20]},
    )


@dataclass
class Predictor:
    model_path: str
    calibrator_path: str | None = None
    _bundle: dict | None = None
    _calibrator: IsotonicCalibrator | None = None
    _cell_calibrator: CellCalibrator | None = None

    def load(self) -> None:
        self._bundle = joblib.load(self.model_path)
        cal_path = self.calibrator_path or (
            str(Path(self.model_path).with_suffix("")) + "_calibrator.joblib"
        )
        if Path(cal_path).exists():
            self._calibrator = IsotonicCalibrator.load(cal_path)
        else:
            log.warning("calibrator_missing", path=cal_path)
            self._calibrator = None
        # Optional per-(category × horizon) calibrator. If present, takes
        # precedence over the global isotonic for the calibrated output.
        cell_path = str(Path(self.model_path).with_suffix("")) + "_cell_calibrator.joblib"
        if Path(cell_path).exists():
            try:
                self._cell_calibrator = CellCalibrator.load(cell_path)
                log.info("cell_calibrator_loaded", n_cells=len(self._cell_calibrator.cells))
            except Exception as e:
                log.warning("cell_calibrator_load_failed", err=str(e))
                self._cell_calibrator = None

    def predict(
        self,
        question: str,
        *,
        liquidity: float = 0.0,
        volume: float = 0.0,
        days_to_resolution: float | None = None,
    ) -> dict:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        feats = question_features(
            question,
            liquidity=liquidity,
            volume=volume,
            days_to_resolution=days_to_resolution,
        )
        feats.update(emb_features(question))
        cols = self._bundle["feature_cols"]
        row = pd.DataFrame([{c: feats.get(c, 0.0) for c in cols}], columns=cols)
        obj = self._bundle.get("objective", "binary")
        if obj == "lambdarank":
            score = float(self._bundle["model"].predict(row)[0])
            raw = float(_sigmoid(np.asarray([score]))[0])
        else:
            raw = float(self._bundle["model"].predict_proba(row)[0, 1])
        cat = categorize(question)
        horiz = horizon_bucket(days_to_resolution)
        p_low = p_high = None
        if self._cell_calibrator is not None:
            cal_p, p_low, p_high = self._cell_calibrator.transform_with_interval(raw, cat, horiz)
            cal = float(cal_p)
            p_low = float(p_low)
            p_high = float(p_high)
        elif self._calibrator is not None:
            cal = float(self._calibrator.transform(raw))
        else:
            cal = raw
        out = {"raw": raw, "calibrated": cal, "category": cat, "horizon": horiz}
        if p_low is not None:
            out["calibrated_low"] = p_low
            out["calibrated_high"] = p_high
        return out

    def predict_batch(
        self,
        questions: list,
    ) -> list[dict]:
        """Batched predict. ~10x faster than calling predict() in a loop.

        `questions` is a list of (question, liquidity, volume) or
        (question, liquidity, volume, days_to_resolution) tuples.
        """
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        if not questions:
            return []
        cols = self._bundle["feature_cols"]
        rows = []
        qtexts = []
        for t in questions:
            if len(t) == 4:
                q, liq, vol, ttr = t
            else:
                q, liq, vol = t
                ttr = None
            feats = question_features(q, liquidity=liq, volume=vol, days_to_resolution=ttr)
            rows.append(feats)
            qtexts.append(q)
        emb_feats = emb_features_batch(qtexts)
        for rec, ef in zip(rows, emb_feats):
            rec.update(ef)
        df = pd.DataFrame(
            [{c: rec.get(c, 0.0) for c in cols} for rec in rows], columns=cols
        )
        obj = self._bundle.get("objective", "binary")
        if obj == "lambdarank":
            scores = np.asarray(self._bundle["model"].predict(df), dtype=float)
            raws = _sigmoid(scores)
        else:
            raws = self._bundle["model"].predict_proba(df)[:, 1]
        # Per-(cat, horiz) calibration
        cats = []
        horizs = []
        for t in questions:
            if len(t) == 4:
                q, _, _, ttr = t
            else:
                q, _, _ = t
                ttr = None
            cats.append(categorize(q))
            horizs.append(horizon_bucket(ttr))
        cals: list[float] = []
        lows: list[float | None] = []
        highs: list[float | None] = []
        if self._cell_calibrator is not None:
            for i in range(len(raws)):
                p, lo, hi = self._cell_calibrator.transform_with_interval(
                    float(raws[i]), cats[i], horizs[i]
                )
                cals.append(float(p))
                lows.append(float(lo))
                highs.append(float(hi))
        elif self._calibrator is not None:
            cals = [float(self._calibrator.transform(float(r))) for r in raws]
            lows = [None] * len(raws)
            highs = [None] * len(raws)
        else:
            cals = [float(r) for r in raws]
            lows = [None] * len(raws)
            highs = [None] * len(raws)
        out = []
        for i in range(len(raws)):
            d = {
                "raw": float(raws[i]),
                "calibrated": cals[i],
                "category": cats[i],
                "horizon": horizs[i],
            }
            if lows[i] is not None:
                d["calibrated_low"] = lows[i]
                d["calibrated_high"] = highs[i]
            out.append(d)
        return out
