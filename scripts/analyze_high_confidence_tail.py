"""High-confidence-tail analysis on signal_outcomes + model_failures.

Question: does the model's accuracy / log-loss / vs-market gap vary
sharply across confidence buckets, or is it ~uniformly bad?

Framing (from the senior-quant review):
  - If the model is much MORE accurate (or much LESS wrong-vs-market)
    in its high-confidence tail, then a selective-abstention gate
    (Venn-Abers width, conformal prediction) materially improves
    Sharpe — because we keep the high-confidence wins and discard
    the noise.
  - If the model is uniformly wrong across confidence buckets, then
    selective abstention only improves Brier / calibration plots,
    not realized P&L. Trading PnL benefits little because the
    high-confidence trades aren't differentially better.

Method: bucket every (p_model, p_market, yes_won) triple by
confidence = |p_model - 0.5| × 2 ∈ [0, 1], then within each bucket
compute:
  - directional accuracy of model
  - directional accuracy of market
  - mean log-loss of model
  - mean log-loss of market
  - delta = model_LL − market_LL  (positive = model worse)

Read the resulting table:
  - if model_acc rises with confidence, selective abstention helps PnL
  - if model_acc is FLAT or falls, the high-confidence tail is just
    overconfidence and Venn-Abers buys you Brier only
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np

DB = Path(r"C:\Users\benja\Downloads\Polymarket\data\paper.db")


def _ll(p: float, y: int) -> float:
    p = max(1e-3, min(1 - 1e-3, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main() -> None:
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        """SELECT p_stat_lgbm,
                  COALESCE(p_market_24h, p_market_6h, p_market_1h, p_market_pre) AS p_market,
                  yes_won,
                  category
           FROM signal_outcomes
           WHERE p_stat_lgbm IS NOT NULL"""
    ).fetchall()
    conn.close()
    print(f"rows with p_model populated: {len(rows)}")

    p_m = np.array([r[0] for r in rows], dtype=float)
    p_mkt_raw = [r[1] for r in rows]
    have_mkt = np.array([x is not None for x in p_mkt_raw])
    p_mkt = np.array([x if x is not None else 0.5 for x in p_mkt_raw], dtype=float)
    y = np.array([int(r[2]) for r in rows], dtype=int)
    cats = [r[3] or "uncat" for r in rows]
    print(f"rows with market price too: {int(have_mkt.sum())}")
    base_rate = y.mean()
    print(f"overall base rate yes_won: {base_rate:.4f}")
    print()

    # Confidence ∈ [0, 1] = |p - 0.5| × 2
    confidence = np.abs(p_m - 0.5) * 2.0

    print("=" * 92)
    print("Q1: model directional accuracy by confidence bucket (whole sample, n=10,215)")
    print("=" * 92)
    print(f"{'bucket':>14}  {'n':>6}  {'mean_p':>8}  {'realized_yes':>14}  {'model_dir_acc':>14}  {'naive_majority':>15}")
    edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.1f},{hi:.1f})    0    --     --        --              --")
            continue
        # model_dir_pred: 1 if p_model > 0.5 else 0
        model_pred = (p_m[mask] > 0.5).astype(int)
        model_acc = float((model_pred == y[mask]).mean())
        naive_acc = max(base_rate, 1 - base_rate)  # base-rate baseline
        print(f"  [{lo:.2f},{hi:.2f})  {n:>6}  {p_m[mask].mean():.4f}    {y[mask].mean():.4f}          {model_acc:.4f}          {naive_acc:.4f}")

    print()
    print("=" * 92)
    print("Q2: model log-loss vs market log-loss, by confidence bucket (head-to-head subsample)")
    print("=" * 92)
    print(f"{'bucket':>14}  {'n':>5}  {'model_LL':>9}  {'market_LL':>9}  {'delta':>9}  {'mod_acc':>8}  {'mkt_acc':>8}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = have_mkt & (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n < 30:
            print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  --insufficient--")
            continue
        ll_m = float(np.mean([_ll(p, int(yy)) for p, yy in zip(p_m[mask], y[mask])]))
        ll_k = float(np.mean([_ll(p, int(yy)) for p, yy in zip(p_mkt[mask], y[mask])]))
        delta = ll_m - ll_k
        mod_acc = float(((p_m[mask] > 0.5).astype(int) == y[mask]).mean())
        mkt_acc = float(((p_mkt[mask] > 0.5).astype(int) == y[mask]).mean())
        flag = " <-- model BETTER" if delta < 0 else ""
        print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  {ll_m:>8.4f}  {ll_k:>8.4f}  {delta:>+8.4f}  {mod_acc:>7.4f}  {mkt_acc:>7.4f}{flag}")

    # === sports_global slice — the certified one ===
    print()
    print("=" * 92)
    print("Q3: SAME analysis restricted to sports_global (certified slice; n_h2h=626)")
    print("=" * 92)
    print(f"{'bucket':>14}  {'n':>5}  {'model_LL':>9}  {'market_LL':>9}  {'delta':>9}  {'mod_acc':>8}  {'mkt_acc':>8}")
    sg_mask = np.array([c == "sports_global" for c in cats])
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = sg_mask & have_mkt & (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n < 20:
            print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  --insufficient--")
            continue
        ll_m = float(np.mean([_ll(p, int(yy)) for p, yy in zip(p_m[mask], y[mask])]))
        ll_k = float(np.mean([_ll(p, int(yy)) for p, yy in zip(p_mkt[mask], y[mask])]))
        delta = ll_m - ll_k
        mod_acc = float(((p_m[mask] > 0.5).astype(int) == y[mask]).mean())
        mkt_acc = float(((p_mkt[mask] > 0.5).astype(int) == y[mask]).mean())
        flag = " <-- model BETTER" if delta < 0 else ""
        print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  {ll_m:>8.4f}  {ll_k:>8.4f}  {delta:>+8.4f}  {mod_acc:>7.4f}  {mkt_acc:>7.4f}{flag}")

    # === Sharpe-relevant: PnL of "trade only the high-confidence tail" simulation ===
    print()
    print("=" * 92)
    print("Q4: simulated naive PnL — buy model-favored side at market_24h, by confidence bucket")
    print("    (whole-sample head-to-head; PnL = (1 if right else 0) − side_price; n_min=30)")
    print("=" * 92)
    print(f"{'bucket':>14}  {'n':>5}  {'mean_pnl':>9}  {'sum_pnl':>10}  {'sharpe':>8}  {'win_rate':>9}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = have_mkt & (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n < 30:
            print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  --insufficient--")
            continue
        side_yes = p_m[mask] > 0.5
        right = (side_yes & (y[mask] == 1)) | (~side_yes & (y[mask] == 0))
        side_price = np.where(side_yes, p_mkt[mask], 1.0 - p_mkt[mask])
        pnl_per = np.where(right, 1.0 - side_price, -side_price)
        mp = float(pnl_per.mean())
        sp = float(pnl_per.std() + 1e-9)
        print(f"  [{lo:.2f},{hi:.2f})  {n:>5}  {mp:>+8.4f}  {pnl_per.sum():>+9.2f}  {mp/sp:>+7.3f}  {right.mean():>8.4f}")

    # === failure_type breakdown by confidence bucket ===
    print()
    print("=" * 92)
    print("Q5: model_failures distribution by confidence bucket (full sample)")
    print("=" * 92)
    conn = sqlite3.connect(str(DB))
    fail_rows = conn.execute(
        """SELECT condition_id, p_model, failure_type, severity
           FROM model_failures
           WHERE p_model IS NOT NULL"""
    ).fetchall()
    conn.close()
    fp = np.array([r[1] for r in fail_rows], dtype=float)
    f_conf = np.abs(fp - 0.5) * 2.0
    f_types = [r[2] for r in fail_rows]
    print(f"  total failure rows with p_model: {len(fail_rows)}")
    distinct_types = sorted(set(f_types))
    for ft in distinct_types:
        mask = np.array([t == ft for t in f_types])
        if mask.sum() == 0:
            continue
        c = f_conf[mask]
        print(f"  {ft:35s} n={int(mask.sum()):>4}  conf median={np.median(c):.3f}  conf mean={c.mean():.3f}")


if __name__ == "__main__":
    main()
