"""Compare LGBM trained with volume features vs trade-count / net-flow
features on the certified sports_global slice.

The Sirolly Nov 2025 finding (SSRN 5714122) is that wash-trade share
peaked at 60% Dec 2024 and sat at ~20% Oct 2025 — with **sports as the
worst-affected category**. Our certified slice is sports_global. The
existing LGBM training pipeline includes `volume` and `log_volume`
features which are exactly the metrics inflated by wash cycling.

This script answers the empirical question: "if we swap the volume
features for wash-robust trade-count + net-flow features, does the
sports_global Brier get better or worse?"

Methodology
-----------
1. Load resolutions + question/liquidity/volume from `resolutions.detail`.
2. Join to `historical_trades` on condition_id, compute per-resolution:
   - `trade_count_24h_pre`: # trades in 24h before resolution
   - `net_flow_24h_pre`:    signed sum of BUY - SELL volume
   - `unique_wallets_24h_pre`: distinct wallet count
   - `top_wallet_share_pre`:  fraction of volume from single most-active wallet
3. Categorize each resolution by question; restrict to sports_global.
4. Split 80/20 train/test by timestamp (chronological hold-out, NOT
   random — random leaks future).
5. Train two LightGBMs:
   - v2: existing features including `volume`, `log_volume`
   - v3: same features minus `volume`/`log_volume` plus the 4 flow features
6. Evaluate both on the 20% held-out set: Brier, log-loss, AUC.
7. Print verdict.

Expected outcomes
-----------------
- **v3 ≥ v2**: clean evidence the volume features were noise.
  Recommendation: swap in production.
- **v3 ≈ v2**: ambiguous. Swap is safe but unlikely to move realized
  P&L by itself.
- **v3 << v2**: volume features carry real signal we shouldn't strip.
  Hold off; the wash contamination is below the model's signal floor
  on this particular slice.

Usage
-----
```
.venv/Scripts/python.exe -m scripts.eval_decontaminated_features
```
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.models.features import question_features
from polyagent.models.categorize import categorize

logging_setup.configure()
log = structlog.get_logger()


# ── Flow features from historical_trades ───────────────────────────────
def compute_flow_features(
    conn: sqlite3.Connection,
    condition_ids: list[str],
    *,
    window_sec: float = 86400.0,
) -> dict[str, dict]:
    """For each condition_id, compute the 4 flow features over the
    `window_sec` preceding the *latest trade* on that market (proxy
    for "immediately before resolution").

    Returns {condition_id: {trade_count, net_flow, unique_wallets,
                            top_wallet_share}}.
    """
    if not condition_ids:
        return {}
    placeholders = ",".join(["?"] * len(condition_ids))
    # Use the latest trade timestamp per condition as the right edge
    # (closest proxy for resolution moment given we don't have a
    # resolved_ts in historical_trades).
    rows = conn.execute(
        f"""SELECT condition_id, wallet, side, size, ts
            FROM historical_trades
            WHERE condition_id IN ({placeholders})""",
        condition_ids,
    ).fetchall()
    if not rows:
        return {}
    # Bucket per condition
    by_cond: dict[str, list[tuple]] = defaultdict(list)
    for cid, wallet, side, size, ts in rows:
        try:
            by_cond[cid].append((wallet, (side or "").upper(),
                                 float(size or 0), float(ts or 0)))
        except (TypeError, ValueError):
            continue
    out: dict[str, dict] = {}
    for cid, trades in by_cond.items():
        if not trades:
            continue
        latest_ts = max(t[3] for t in trades)
        cutoff = latest_ts - window_sec
        window = [t for t in trades if t[3] >= cutoff]
        if not window:
            window = trades  # fallback to all
        wallet_vol: dict[str, float] = defaultdict(float)
        buy_vol = 0.0
        sell_vol = 0.0
        for wallet, side, size, _ts in window:
            wallet_vol[wallet] += size
            if side == "BUY":
                buy_vol += size
            elif side == "SELL":
                sell_vol += size
        total_vol = buy_vol + sell_vol
        top_wallet_share = (max(wallet_vol.values()) / total_vol) if total_vol > 0 else 0.0
        out[cid] = {
            "trade_count_24h_pre": float(len(window)),
            "net_flow_24h_pre": float(buy_vol - sell_vol),
            "unique_wallets_24h_pre": float(len(wallet_vol)),
            "top_wallet_share_pre": float(top_wallet_share),
        }
    log.info("flow_features_computed", n_conditions=len(out))
    return out


# ── Dataset assembly ───────────────────────────────────────────────────
def load_eval_dataset(
    db_path: str,
    *,
    category_filter: str = "sports_global",
) -> pd.DataFrame:
    """Build the eval dataset: resolutions joined to flow features,
    restricted to `category_filter`."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT condition_id, yes_won, detail, resolved_ts FROM resolutions"
    ).fetchall()
    records = []
    by_cond_details = {}
    for cid, yes_won, detail_str, resolved_ts in rows:
        try:
            d = json.loads(detail_str or "{}")
        except json.JSONDecodeError:
            continue
        question = d.get("question") or ""
        if not question:
            continue
        cat = categorize(question)
        if cat != category_filter:
            continue
        by_cond_details[cid] = {
            "yes_won": int(yes_won),
            "question": question,
            "liquidity": d.get("liquidity") or 0.0,
            "volume": d.get("volume") or 0.0,
            "resolved_ts": float(resolved_ts or 0.0),
        }
    log.info(
        "load_eval_dataset",
        category=category_filter,
        n_in_category=len(by_cond_details),
        total_resolutions=len(rows),
    )
    # Compute flow features for the in-category condition_ids
    flow = compute_flow_features(conn, list(by_cond_details.keys()))
    conn.close()
    for cid, d in by_cond_details.items():
        feats_v2 = question_features(
            d["question"], liquidity=d["liquidity"], volume=d["volume"],
        )
        rec = dict(feats_v2)
        rec["condition_id"] = cid
        rec["yes_won"] = d["yes_won"]
        rec["resolved_ts"] = d["resolved_ts"]
        f = flow.get(cid, {})
        rec["trade_count_24h_pre"] = f.get("trade_count_24h_pre", 0.0)
        rec["net_flow_24h_pre"] = f.get("net_flow_24h_pre", 0.0)
        rec["unique_wallets_24h_pre"] = f.get("unique_wallets_24h_pre", 0.0)
        rec["top_wallet_share_pre"] = f.get("top_wallet_share_pre", 0.0)
        # log scaling for trade_count and unique_wallets
        rec["log_trade_count_24h_pre"] = math.log1p(rec["trade_count_24h_pre"])
        rec["log_unique_wallets_24h_pre"] = math.log1p(rec["unique_wallets_24h_pre"])
        records.append(rec)
    return pd.DataFrame.from_records(records)


# ── Train + eval ───────────────────────────────────────────────────────
def train_and_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """Train LGBM on chronological split. Returns Brier, log-loss, AUC,
    ECE on held-out 20%."""
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    if "resolved_ts" not in df.columns:
        raise ValueError("need resolved_ts for chronological split")
    df_sorted = df.sort_values("resolved_ts").reset_index(drop=True)
    split = int(len(df_sorted) * train_frac)
    train_df = df_sorted.iloc[:split]
    test_df = df_sorted.iloc[split:]
    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df["yes_won"].astype(int).values
    X_test = test_df[feature_cols].astype(float).values
    y_test = test_df["yes_won"].astype(int).values
    if len(X_test) < 5 or len(set(y_test.tolist())) < 2:
        return {"error": "test set too small or all one class"}
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, num_leaves=31, min_child_samples=10,
        objective="binary", random_state=seed, verbosity=-1,
    )
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    p_test = np.clip(p_test, 1e-4, 1 - 1e-4)
    # ECE 10-bin
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_test >= lo) & (p_test < hi) if i < 9 else (p_test >= lo) & (p_test <= hi)
        if mask.sum() == 0:
            continue
        bin_p = float(p_test[mask].mean())
        bin_y = float(y_test[mask].mean())
        ece += mask.sum() / len(p_test) * abs(bin_p - bin_y)
    importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )[:10]
    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "log_loss": float(log_loss(y_test, p_test)),
        "auc": float(roc_auc_score(y_test, p_test)) if len(set(y_test.tolist())) == 2 else None,
        "ece": float(ece),
        "test_base_rate": float(y_test.mean()),
        "top_features": importance,
    }


# ── Entry ──────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=settings.db_path)
    p.add_argument("--category", default="sports_global")
    p.add_argument("--train_frac", type=float, default=0.8)
    args = p.parse_args()

    df = load_eval_dataset(args.db, category_filter=args.category)
    if len(df) < 50:
        print(f"too few rows ({len(df)}) in {args.category}; aborting")
        return 1

    # Feature set definitions
    # Exclude condition_id, yes_won, resolved_ts from features
    all_cols = [c for c in df.columns
                if c not in ("condition_id", "yes_won", "resolved_ts")]
    flow_cols = [
        "trade_count_24h_pre", "net_flow_24h_pre",
        "unique_wallets_24h_pre", "top_wallet_share_pre",
        "log_trade_count_24h_pre", "log_unique_wallets_24h_pre",
    ]
    volume_cols = ["volume", "log_volume"]

    # v2 = existing features (includes volume/log_volume, excludes flow features)
    v2_cols = [c for c in all_cols if c not in flow_cols]
    # v3 = same features minus volume/log_volume plus flow features
    v3_cols = [c for c in all_cols if c not in volume_cols]

    print(f"\n=== {args.category} feature-swap evaluation ===")
    print(f"n_rows: {len(df)}, n_features v2={len(v2_cols)}, v3={len(v3_cols)}")
    print(f"flow features added: {flow_cols}")
    print(f"volume features removed in v3: {volume_cols}\n")

    # v4 = both volume AND flow features (does flow add incremental info?)
    v4_cols = list(all_cols)

    res_v2 = train_and_eval(df, v2_cols, train_frac=args.train_frac)
    res_v3 = train_and_eval(df, v3_cols, train_frac=args.train_frac)
    res_v4 = train_and_eval(df, v4_cols, train_frac=args.train_frac)

    if "error" in res_v2 or "error" in res_v3 or "error" in res_v4:
        print(f"errors: v2={res_v2.get('error')} v3={res_v3.get('error')} v4={res_v4.get('error')}")
        return 1

    def fmt(r):
        return (f"  Brier={r['brier']:.4f}  log_loss={r['log_loss']:.4f}  "
                f"AUC={r['auc']:.4f}  ECE={r['ece']:.4f}  "
                f"n_train={r['n_train']} n_test={r['n_test']} "
                f"base_rate={r['test_base_rate']:.3f}")
    print("v2 (volume features, no flow):")
    print(fmt(res_v2))
    print(f"  top features: {[k for k, _ in res_v2['top_features'][:5]]}")
    print("\nv3 (flow features, volume removed):")
    print(fmt(res_v3))
    print(f"  top features: {[k for k, _ in res_v3['top_features'][:5]]}")
    print("\nv4 (both volume AND flow):")
    print(fmt(res_v4))
    print(f"  top features: {[k for k, _ in res_v4['top_features'][:5]]}")

    print("\ndeltas vs v2 baseline:")
    for label, r in [("v3", res_v3), ("v4", res_v4)]:
        d_brier = r["brier"] - res_v2["brier"]
        d_logloss = r["log_loss"] - res_v2["log_loss"]
        d_auc = (r["auc"] or 0) - (res_v2["auc"] or 0)
        print(f"  {label}: dBrier={d_brier:+.4f}  dlogloss={d_logloss:+.4f}  dAUC={d_auc:+.4f}")

    # Verdict logic
    if res_v3["auc"] < res_v2["auc"] - 0.05:
        verdict_v3 = "REJECT v3 -- volume feature carries discrimination we just stripped"
    elif res_v3["brier"] < res_v2["brier"] - 0.001:
        verdict_v3 = "ADOPT v3 -- meaningful Brier improvement"
    else:
        verdict_v3 = "AMBIGUOUS v3 -- swap is roughly Brier-neutral"
    if res_v4["brier"] < res_v2["brier"] - 0.001 and res_v4["auc"] >= res_v2["auc"] - 0.02:
        verdict_v4 = "ADOPT v4 -- flow features add incremental info on top of volume"
    elif res_v4["brier"] > res_v2["brier"] + 0.002:
        verdict_v4 = "REJECT v4 -- flow features add noise"
    else:
        verdict_v4 = "AMBIGUOUS v4 -- flow features within-noise"
    print(f"\nverdict v3 (replace volume with flow): {verdict_v3}")
    print(f"verdict v4 (add flow alongside volume): {verdict_v4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
