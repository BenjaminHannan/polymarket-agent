"""Weekly performance report — runnable standalone, prints to stdout.

Builds the three artefacts called out in PROJECT.md "Where to look next":

  1. Per-strategy realized Sharpe over time (daily-bucket returns).
  2. Hit rate by category (fills joined to resolutions).
  3. Calibration data: combined_signal p_combined vs realized YES rate.

Reads from `data/paper.db` only. No live dependencies, no network calls.
Honest about small-n: prints raw counts alongside every ratio and refuses
to compute Sharpe with fewer than 3 daily returns. Per Bailey-López de
Prado MinBTL, ~1,500 OOS forward trades are needed to *certify* a
non-zero Sharpe; this report gives empirical visibility, not certification.

Run:
    .venv/Scripts/python.exe -m polyagent.eval.weekly_report
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from polyagent.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DAY = 86400.0


def _day_bucket(ts: float) -> int:
    """Floor-divide an epoch timestamp into UTC day index."""
    return int(ts // _DAY)


def _sharpe(returns: list[float]) -> tuple[float, float, int]:
    """Plain Sharpe: mean / std × sqrt(252). Returns (sharpe, mean, n).

    Uses ddof=1. Returns NaN sharpe when n<3 or std==0 so callers can
    distinguish "no data" from "zero excess return". Annualised at 252
    trading days even though Polymarket is 7-day; the 252 convention
    keeps the number comparable to traditional Sharpe quotes — multiply
    by sqrt(252/365) ≈ 0.83 for a calendar-day equivalent.
    """
    n = len(returns)
    if n < 3:
        return float("nan"), (sum(returns) / n if n else 0.0), n
    mu = statistics.fmean(returns)
    sd = statistics.stdev(returns)
    if sd <= 0:
        return float("nan"), mu, n
    return (mu / sd) * math.sqrt(252.0), mu, n


def _categorize(question: str) -> str:
    """Lazy import the project categorizer so this script doesn't
    pull the whole stack just to count buckets."""
    try:
        from polyagent.models.categorize import categorize
        return categorize(question or "")
    except Exception:
        return "unknown"


@dataclass
class _Pos:
    """Per-token rolling FIFO-ish position used to attribute realized P&L
    to specific BUYs. We track total notional/size and replay SELLs / final
    resolutions against the average cost — same arithmetic the broker uses.
    The result is a per-strategy realized P&L stream that's directly
    sharpe-able by trade date."""
    size: float = 0.0
    cost: float = 0.0   # cumulative cost basis (price * size)
    # Earliest BUY ts for the still-open piece — we date the trade by it
    # so that realized P&L lands on the entry day, not the resolution day.
    # This is what makes Sharpe-by-strategy honest: a strategy that bought
    # 30 days ago and resolved today should not look like it earned the
    # return *today*. Equity managers do it the same way.
    first_buy_ts: float = 0.0
    strategy: str = ""


# ---------------------------------------------------------------------------
# Section 1 — per-strategy daily Sharpe
# ---------------------------------------------------------------------------

def per_strategy_sharpe(con: sqlite3.Connection) -> dict:
    """Walk `fills` in chronological order; for each token, hold a running
    Position keyed by strategy. SELLs realize P&L back to the BUY's
    strategy. Settlements (rows in `resolutions`) realize all remaining
    open size at the resolved payout (1 for the winning leg, 0 for the
    loser). The result is a per-strategy stream of realized P&L,
    timestamped to the *entry* date.

    Returns dict[strategy] -> {trades, total_pnl, mean_pnl, sharpe,
                               daily_returns_n, daily_sharpe}.
    """
    # Resolutions keyed by (token_id) -> (yes_won, condition_id)
    cur = con.execute(
        "SELECT condition_id, yes_token_id, no_token_id, yes_won, resolved_ts "
        "FROM resolutions"
    )
    yes_won_for: dict[str, tuple[bool, str, float]] = {}
    no_won_for: dict[str, tuple[bool, str, float]] = {}
    for cid, yes_tok, no_tok, yes_won, rts in cur:
        yes_won_for[yes_tok] = (bool(yes_won), cid, float(rts or 0.0))
        no_won_for[no_tok] = (not bool(yes_won), cid, float(rts or 0.0))

    # Position keyed by (token_id, strategy) — a token bought by combined_trader
    # and the same token bought by passive_poster are separate ledger lines.
    positions: dict[tuple[str, str], _Pos] = {}

    # Each per-trade realized P&L row: (entry_ts, strategy, pnl, notional)
    realized: list[tuple[float, str, float, float]] = []

    cur = con.execute(
        "SELECT ts, strategy, condition_id, token_id, side, price, size, notional "
        "FROM fills ORDER BY ts ASC, id ASC"
    )
    for ts, strat, _cid, tok, side, price, size, notional in cur:
        key = (tok, strat)
        pos = positions.get(key)
        if pos is None:
            pos = _Pos(strategy=strat)
            positions[key] = pos
        side = (side or "").upper()
        size = float(size or 0.0)
        price = float(price or 0.0)
        if size <= 0:
            continue
        if side == "BUY":
            if pos.size <= 0:
                pos.first_buy_ts = float(ts)
            pos.size += size
            pos.cost += size * price
        elif side == "SELL":
            # Realize at the SELL price vs running avg cost on this strategy.
            if pos.size > 0:
                sell_size = min(size, pos.size)
                avg = pos.cost / pos.size
                pnl = (price - avg) * sell_size
                realized.append((pos.first_buy_ts or float(ts), strat, pnl, sell_size * price))
                # Reduce position proportionally
                pos.cost *= (pos.size - sell_size) / pos.size
                pos.size -= sell_size
                if pos.size <= 1e-9:
                    pos.size = 0.0
                    pos.cost = 0.0
                    pos.first_buy_ts = 0.0

    # Settle any open positions at the resolved payout.
    for (tok, strat), pos in positions.items():
        if pos.size <= 1e-9:
            continue
        win_info = yes_won_for.get(tok) or no_won_for.get(tok)
        if win_info is None:
            continue  # not resolved yet; skip (unrealized — not Sharpe-able)
        won, _cid, _rts = win_info
        # For NO-token: outcome flipped in `no_won_for` above (won = not yes_won),
        # so `won=True` means this token's leg paid 1.0.
        payout = 1.0 if won else 0.0
        avg = pos.cost / pos.size if pos.size > 0 else 0.0
        pnl = (payout - avg) * pos.size
        realized.append((pos.first_buy_ts, strat, pnl, pos.size * avg))

    # Aggregate per strategy.
    out: dict[str, dict] = {}
    by_strat: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for ts, strat, pnl, notional in realized:
        by_strat[strat].append((ts, pnl, notional))

    for strat, rows in by_strat.items():
        total_pnl = sum(r[1] for r in rows)
        total_notional = sum(r[2] for r in rows)
        mean = total_pnl / max(1, len(rows))
        # Per-trade return (pnl / notional) — guard zero notional.
        per_trade_returns = [
            (r[1] / r[2]) if r[2] > 0 else 0.0 for r in rows
        ]
        sharpe_trade, _, n_trade = _sharpe(per_trade_returns)
        # Daily-bucket: sum P&L by day, then sharpe.
        by_day: dict[int, float] = defaultdict(float)
        for ts, pnl, _n in rows:
            by_day[_day_bucket(ts)] += pnl
        daily_pnls = [by_day[k] for k in sorted(by_day)]
        # Daily return: pnl normalised by starting NAV (rough but stable).
        nav0 = settings.starting_nav or 10000.0
        daily_returns = [p / nav0 for p in daily_pnls]
        sharpe_day, _, n_day = _sharpe(daily_returns)
        out[strat] = {
            "trades": len(rows),
            "total_pnl": total_pnl,
            "total_notional": total_notional,
            "mean_pnl_per_trade": mean,
            "sharpe_per_trade": sharpe_trade,
            "n_trade": n_trade,
            "daily_returns_n": n_day,
            "sharpe_daily": sharpe_day,
        }
    return out


# ---------------------------------------------------------------------------
# Section 2 — hit rate by category
# ---------------------------------------------------------------------------

def hit_rate_by_category(con: sqlite3.Connection) -> dict:
    """For each fill on a resolved market, classify whether the bet won.

    A BUY on a YES-token wins iff yes_won == True (the token paid $1).
    A BUY on a NO-token wins iff yes_won == False.

    Groups by category derived from the resolution row's stored question
    (or signal_outcomes.question when present). Returns dict[category]
    -> {n, wins, hit_rate, total_pnl_avg_cost}.
    """
    # Build token -> (yes_won, condition_id, question) lookup
    cur = con.execute(
        "SELECT r.condition_id, r.yes_token_id, r.no_token_id, r.yes_won, "
        "       COALESCE(so.question, json_extract(r.detail,'$.question'),'') "
        "FROM resolutions r LEFT JOIN signal_outcomes so "
        "  ON so.condition_id = r.condition_id"
    )
    token_meta: dict[str, tuple[bool, str, str]] = {}
    for cid, yes_tok, no_tok, yes_won, question in cur:
        token_meta[yes_tok] = (bool(yes_won), cid, question or "")
        token_meta[no_tok] = (not bool(yes_won), cid, question or "")

    by_cat: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "wins": 0, "total_cost": 0.0, "total_payout": 0.0,
    })
    cur = con.execute(
        "SELECT token_id, side, price, size FROM fills WHERE side='BUY'"
    )
    for tok, side, price, size in cur:
        meta = token_meta.get(tok)
        if meta is None:
            continue
        won, _cid, question = meta
        cat = _categorize(question)
        size = float(size or 0.0)
        price = float(price or 0.0)
        bucket = by_cat[cat]
        bucket["n"] += 1
        bucket["wins"] += 1 if won else 0
        bucket["total_cost"] += price * size
        bucket["total_payout"] += (1.0 if won else 0.0) * size

    out = {}
    for cat, b in by_cat.items():
        hit = b["wins"] / b["n"] if b["n"] else 0.0
        pnl = b["total_payout"] - b["total_cost"]
        roi = pnl / b["total_cost"] if b["total_cost"] > 0 else 0.0
        out[cat] = {
            "n": b["n"],
            "wins": b["wins"],
            "hit_rate": hit,
            "total_cost": b["total_cost"],
            "total_pnl": pnl,
            "roi": roi,
        }
    return out


# ---------------------------------------------------------------------------
# Section 3 — combined_signal calibration: p_combined vs realized YES rate
# ---------------------------------------------------------------------------

def combined_calibration(con: sqlite3.Connection, bins: int = 10) -> dict:
    """For every `signals` row with strategy='combined' and a resolved
    market, bucket by p_combined and report the realized YES rate.

    A well-calibrated forecaster has realized_yes_rate ≈ bin midpoint
    across the row. Systematic overshoot in any bin is miscalibration.
    The previous calibration audit (see PROJECT.md May-10 session) showed
    +0.14 log-loss worse than market overall — this report makes that
    finding visible to anyone running it post-deployment.

    Returns dict with keys:
      buckets: list of {lo, hi, n, mean_p_combined, mean_p_market,
                        realized_yes_rate, model_brier_contrib,
                        market_brier_contrib}
      overall: {n, model_brier, market_brier, delta}
    """
    bucket_edges = [i / bins for i in range(bins + 1)]
    rows = [
        {"lo": bucket_edges[i], "hi": bucket_edges[i + 1],
         "n": 0, "sum_p_comb": 0.0, "sum_p_mkt": 0.0, "wins": 0,
         "sse_model": 0.0, "sse_market": 0.0}
        for i in range(bins)
    ]
    yes_won_map: dict[str, bool] = {}
    for cid, yes_won in con.execute(
        "SELECT condition_id, yes_won FROM resolutions"
    ):
        yes_won_map[cid] = bool(yes_won)
    total = 0
    sse_model_total = 0.0
    sse_market_total = 0.0

    cur = con.execute(
        "SELECT condition_id, direction, detail FROM signals "
        "WHERE strategy='combined' AND detail IS NOT NULL"
    )
    for cid, _direction, detail_json in cur:
        if cid not in yes_won_map:
            continue
        try:
            detail = json.loads(detail_json)
        except (TypeError, ValueError):
            continue
        p_comb = detail.get("p_combined")
        p_mkt = detail.get("p_market")
        if p_comb is None or p_mkt is None:
            continue
        try:
            p_comb = float(p_comb)
            p_mkt = float(p_mkt)
        except (TypeError, ValueError):
            continue
        yes = 1.0 if yes_won_map[cid] else 0.0
        # Bucket on p_combined (the model's stance, what's interesting for
        # calibration). Clamp to last bucket on p == 1.0.
        idx = min(bins - 1, int(p_comb * bins))
        b = rows[idx]
        b["n"] += 1
        b["sum_p_comb"] += p_comb
        b["sum_p_mkt"] += p_mkt
        b["wins"] += int(yes)
        b["sse_model"] += (p_comb - yes) ** 2
        b["sse_market"] += (p_mkt - yes) ** 2
        total += 1
        sse_model_total += (p_comb - yes) ** 2
        sse_market_total += (p_mkt - yes) ** 2

    buckets = []
    for b in rows:
        n = b["n"]
        buckets.append({
            "lo": b["lo"],
            "hi": b["hi"],
            "n": n,
            "mean_p_combined": (b["sum_p_comb"] / n) if n else None,
            "mean_p_market": (b["sum_p_mkt"] / n) if n else None,
            "realized_yes_rate": (b["wins"] / n) if n else None,
            "model_brier": (b["sse_model"] / n) if n else None,
            "market_brier": (b["sse_market"] / n) if n else None,
        })
    overall = {
        "n": total,
        "model_brier": (sse_model_total / total) if total else None,
        "market_brier": (sse_market_total / total) if total else None,
        "delta_brier_model_minus_market": (
            (sse_model_total - sse_market_total) / total if total else None
        ),
    }
    return {"buckets": buckets, "overall": overall}


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def _fmt(x, n=3) -> str:
    if x is None:
        return "--"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{n}f}"
    return str(x)


def print_report(con: sqlite3.Connection) -> None:
    print("=" * 78)
    print("POLYAGENT WEEKLY PERFORMANCE REPORT")
    print("=" * 78)
    print(f"DB: {settings.db_path}")

    # NAV history snapshot — quick sanity line.
    row = con.execute(
        "SELECT MIN(ts), MAX(ts), COUNT(*) FROM nav_history"
    ).fetchone() or (None, None, 0)
    if row[2]:
        print(
            f"nav_history rows: {row[2]} (first {row[0]:.0f} -> last {row[1]:.0f}, "
            f"span {(row[1] - row[0]) / _DAY:.1f}d)"
        )
    else:
        print("nav_history rows: 0 — bot has not snapshotted yet.")

    n_fills = con.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    n_res = con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
    n_sig = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"fills={n_fills}  resolutions={n_res}  signals={n_sig}")
    print()

    # --- Section 1 ---
    print("-" * 78)
    print("[1] PER-STRATEGY REALIZED SHARPE")
    print("-" * 78)
    per_strat = per_strategy_sharpe(con)
    if not per_strat:
        print("  (no realized fills yet -- nothing to attribute)")
    else:
        header = f"  {'strategy':<24} {'trades':>7} {'pnl':>10} {'mean':>8} {'sh_trade':>9} {'sh_daily':>9} {'n_days':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for strat, m in sorted(per_strat.items(), key=lambda kv: -kv[1]["total_pnl"]):
            print(
                f"  {strat:<24} {m['trades']:>7d} "
                f"{m['total_pnl']:>10.2f} "
                f"{m['mean_pnl_per_trade']:>8.3f} "
                f"{_fmt(m['sharpe_per_trade']):>9} "
                f"{_fmt(m['sharpe_daily']):>9} "
                f"{m['daily_returns_n']:>7d}"
            )
        print("  note: sh_trade = per-trade return Sharpe; sh_daily = day-bucketed.")
        print("  Bailey-LdP MinBTL says detecting Sharpe 0.3–0.5 needs ~1500-3000")
        print("  resolved trades. Below that bar, every number above is variance,")
        print("  not skill.")
    print()

    # --- Section 2 ---
    print("-" * 78)
    print("[2] HIT RATE BY CATEGORY (fills joined to resolutions)")
    print("-" * 78)
    by_cat = hit_rate_by_category(con)
    if not by_cat:
        print("  (no resolved fills yet)")
    else:
        header = f"  {'category':<18} {'n':>5} {'wins':>5} {'hit%':>6} {'cost':>10} {'pnl':>10} {'roi%':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        total_n = 0
        total_cost = 0.0
        total_pnl = 0.0
        for cat, m in sorted(by_cat.items(), key=lambda kv: -kv[1]["n"]):
            print(
                f"  {cat:<18} {m['n']:>5d} {m['wins']:>5d} "
                f"{m['hit_rate'] * 100:>5.1f}% "
                f"{m['total_cost']:>10.2f} {m['total_pnl']:>10.2f} "
                f"{m['roi'] * 100:>6.2f}%"
            )
            total_n += m["n"]
            total_cost += m["total_cost"]
            total_pnl += m["total_pnl"]
        roi = (total_pnl / total_cost * 100) if total_cost else 0.0
        print("  " + "-" * (len(header) - 2))
        print(f"  {'TOTAL':<18} {total_n:>5d} {'':>5} {'':>6} "
              f"{total_cost:>10.2f} {total_pnl:>10.2f} {roi:>6.2f}%")
    print()

    # --- Section 3 ---
    print("-" * 78)
    print("[3] COMBINED-SIGNAL CALIBRATION (p_combined vs realized YES rate)")
    print("-" * 78)
    calib = combined_calibration(con, bins=10)
    overall = calib["overall"]
    if not overall["n"]:
        print("  (no resolved combined signals yet)")
    else:
        header = f"  {'bucket':<14} {'n':>6} {'mean_p_comb':>12} {'mean_p_mkt':>12} {'yes_rate':>10} {'brier_M':>8} {'brier_mkt':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for b in calib["buckets"]:
            label = f"[{b['lo']:.1f},{b['hi']:.1f})"
            print(
                f"  {label:<14} {b['n']:>6d} "
                f"{_fmt(b['mean_p_combined']):>12} "
                f"{_fmt(b['mean_p_market']):>12} "
                f"{_fmt(b['realized_yes_rate']):>10} "
                f"{_fmt(b['model_brier']):>8} "
                f"{_fmt(b['market_brier']):>10}"
            )
        print()
        print(
            f"  OVERALL n={overall['n']}  "
            f"model_brier={_fmt(overall['model_brier'])}  "
            f"market_brier={_fmt(overall['market_brier'])}  "
            f"delta={_fmt(overall['delta_brier_model_minus_market'])}"
        )
        if overall["delta_brier_model_minus_market"] is not None and (
            overall["delta_brier_model_minus_market"] > 0
        ):
            print(
                "  delta>0 means the model is *worse-calibrated* than the market "
                "(higher Brier). This matches the May-10 audit finding."
            )
    print()
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=settings.db_path,
                    help=f"path to paper.db (default: {settings.db_path})")
    args = ap.parse_args(argv)
    if not Path(args.db).exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        return 2
    con = sqlite3.connect(args.db)
    try:
        print_report(con)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
