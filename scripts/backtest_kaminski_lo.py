"""Falsifiability harness for §10 (Kaminski-Lo gated stops).

Computes the AR(1) coefficient φ and the daily Sharpe ratio on the
resolved-trade return series, and reports whether the canonical
Kaminski-Lo condition (φ ≥ SR_daily) is met. If φ < SR_daily, the
existing fixed-percentage stop has been *removing mean* and disabling
it should be pure +Sharpe (mostly via mean lift, per the doc's
"+0.05 to +0.25 mostly removes mean drag" estimate).

Unlike the BOCPD harness, this one has a clear statistical answer
even on small samples — we're not asking BOCPD to detect a
distribution shift, we're estimating two scalars and comparing them.

Run:
    python -m scripts.backtest_kaminski_lo
"""

from __future__ import annotations

import argparse

import numpy as np

from polyagent.config import settings
from polyagent.risk.exit_policy import (
    ar1_coefficient,
    daily_sharpe,
    fetch_resolved_returns,
)


def run(db_path: str) -> int:
    rs, ts = fetch_resolved_returns(db_path)
    print(f"Resolved-trade returns from {db_path}: n={rs.size}")
    if rs.size < 5:
        print("Insufficient resolved-trade data. Cannot estimate phi or SR_daily.")
        print("Default to keeping existing stop-loss; revisit when n>=30.")
        return 2
    phi = ar1_coefficient(rs)
    sr_per_trade = float(np.mean(rs) / np.std(rs, ddof=1)) if np.std(rs, ddof=1) > 0 else float("nan")
    sr_daily = daily_sharpe(rs, ts)
    print()
    print(f"  mean per-trade return : {np.mean(rs):+.4f}")
    print(f"  std  per-trade return : {np.std(rs, ddof=1):.4f}")
    print(f"  per-trade SR          : {sr_per_trade:+.4f}")
    print(f"  daily SR              : {sr_daily:+.4f}")
    print(f"  AR(1) phi             : {phi:+.4f}")
    print()
    if np.isnan(phi) or np.isnan(sr_daily):
        print("phi or SR_daily not estimable — keep stops enabled.")
        return 2
    threshold_met = phi >= sr_daily
    print(f"Kaminski-Lo condition (phi >= SR_daily): {threshold_met}")
    print()
    if threshold_met:
        print("VERDICT: existing price stops add Sharpe under the K-L theorem.")
        print("Action: keep stops enabled.")
        return 0
    print("VERDICT: phi < SR_daily. The existing fixed-percentage stop is")
    print("REMOVING MEAN per Kaminski & Lo (2014). Disabling it should")
    print("be pure +Sharpe mostly via removed mean drag (no proportional")
    print("variance reduction since returns are mean-reverting / slightly")
    print("anti-correlated, not momentum-y).")
    print()
    print("Action: ENABLE_KAMINSKI_LO_GATE=1 will disable the price stop")
    print("automatically when this condition holds.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=settings.db_path)
    args = p.parse_args()
    return run(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
