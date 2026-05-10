"""Reconcile live `fills` against the queue-aware shadow fills.

Direct implementation of `pmwhybetter.md` Top-5 priority #1 closing
step: *"Validate by reconciling against your live ~190 fills."* The
queue-aware fill simulator (`polyagent/risk/queue_aware_fills.py`)
writes `fills_shadow_queue` rows alongside every real broker fill.
This script reads both, joins on tx hash + token + ts, and produces
a haircut summary:

  - **Top-of-book naive P&L**     — what the broker recorded
  - **Walked VWAP P&L**           — top-of-book walked through depth
  - **Pessimistic queue-aware**   — Brownian σ√Δt + queue position +
                                    cancel-latency drift

The doc projects 20–40% inflation of paper P&L vs. realistic fills.
The reconciliation tells us empirically what the haircut is for *our*
fills, not just the literature's averages.

References
----------
  - hftbacktest semantics: `power_prob_queue_model=3` is the
    post-2024 default. Our queue-aware simulator approximates this.
  - Akey, Gregoire, Harvie & Martineau 2026 (SSRN 6443103): the
    9.3 pp loss-probability reduction per σ maker-share quantification.

Usage
-----
```
.venv/Scripts/python.exe -m scripts.reconcile_queue_aware
.venv/Scripts/python.exe -m scripts.reconcile_queue_aware --strategy combined_trader
.venv/Scripts/python.exe -m scripts.reconcile_queue_aware --since 1735689600
```
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict

from polyagent.config import settings


def reconcile(
    db_path: str,
    *,
    strategy: str | None = None,
    since_ts: float | None = None,
) -> dict:
    """Compute per-strategy reconciliation between real and shadow fills.

    Returns a dict with structure:
      {
        "n_fills": int,
        "by_strategy": {
            "<name>": {
                "n": int,
                "naive_pnl_usd": float,
                "walked_pnl_usd": float,
                "queue_aware_pnl_usd": float,
                "naive_vs_walked_pct": float,
                "naive_vs_queue_aware_pct": float,
            }
        }
      }
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=30000")
    where: list[str] = ["1=1"]
    params: list = []
    if strategy:
        where.append("f.strategy = ?")
        params.append(strategy)
    if since_ts:
        where.append("f.ts >= ?")
        params.append(since_ts)

    # First, check if shadow tables exist; if not, return zeroed stats.
    cols = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    has_shadow = "fills_shadow_queue" in cols

    if has_shadow:
        sql = f"""SELECT f.strategy, f.ts, f.side, f.size, f.price,
                         f.token_id,
                         COALESCE(s.walked_vwap_price, f.price) AS walked,
                         COALESCE(s.queue_aware_price, f.price) AS queue_aware
                  FROM fills f
                  LEFT JOIN fills_shadow_queue s
                    ON s.token_id = f.token_id
                       AND s.ts = f.ts
                       AND s.side = f.side
                  WHERE {" AND ".join(where)}
                  ORDER BY f.ts"""
    else:
        sql = f"""SELECT f.strategy, f.ts, f.side, f.size, f.price,
                         f.token_id,
                         f.price AS walked,
                         f.price AS queue_aware
                  FROM fills f
                  WHERE {" AND ".join(where)}
                  ORDER BY f.ts"""
    rows = conn.execute(sql, params).fetchall()

    out_by_strat: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "naive_pnl_usd": 0.0,
        "walked_pnl_usd": 0.0,
        "queue_aware_pnl_usd": 0.0,
    })
    for strat, ts, side, size, price, token_id, walked, queue_aware in rows:
        s = float(size or 0)
        if s <= 0:
            continue
        # Cost contribution per side: BUY adds cost; SELL releases it.
        # For the haircut calc we approximate "executable P&L if filled at
        # X" as: shares × (1 − price) [BUY of YES] − cost.
        # Here we only care about price-realization deltas, so attribute
        # naive vs walked vs queue-aware as the *price haircut*:
        haircut_walked = (walked - price) * s if side == "BUY" else (price - walked) * s
        haircut_qa = (queue_aware - price) * s if side == "BUY" else (price - queue_aware) * s
        bucket = out_by_strat[strat]
        bucket["n"] += 1
        bucket["naive_pnl_usd"] += 0.0  # baseline by construction
        bucket["walked_pnl_usd"] += -haircut_walked
        bucket["queue_aware_pnl_usd"] += -haircut_qa
    # Compute percent haircuts
    for strat, b in out_by_strat.items():
        if b["n"] > 0:
            b["naive_vs_walked_pct"] = (
                100.0 * b["walked_pnl_usd"] / max(1.0, b["n"])
            )
            b["naive_vs_queue_aware_pct"] = (
                100.0 * b["queue_aware_pnl_usd"] / max(1.0, b["n"])
            )
        else:
            b["naive_vs_walked_pct"] = 0.0
            b["naive_vs_queue_aware_pct"] = 0.0
    return {
        "n_fills": len(rows),
        "has_shadow_table": has_shadow,
        "by_strategy": dict(out_by_strat),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=None,
                   help="restrict to one strategy (default: all)")
    p.add_argument("--since", type=float, default=None,
                   help="UTC unix timestamp lower bound")
    p.add_argument("--db", default=None,
                   help="paper.db path (default: settings.db_path)")
    args = p.parse_args()

    db_path = args.db or settings.db_path
    res = reconcile(db_path, strategy=args.strategy, since_ts=args.since)
    print(f"n_fills: {res['n_fills']}  shadow_table: {res['has_shadow_table']}")
    if not res["by_strategy"]:
        print("(no fills under filter)")
        return 0
    print(f"\n{'strategy':<26s} {'n':>5s} {'walked_pnl_usd':>15s} "
          f"{'queue_pnl_usd':>14s} {'walked_pct':>12s} {'queue_pct':>12s}")
    for strat, b in sorted(res["by_strategy"].items()):
        print(
            f"  {strat:<24s} {b['n']:>5} "
            f"{b['walked_pnl_usd']:>15.2f} {b['queue_aware_pnl_usd']:>14.2f} "
            f"{b['naive_vs_walked_pct']:>11.2f}% {b['naive_vs_queue_aware_pct']:>11.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
