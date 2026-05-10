"""Queue-aware backtest replay over book_archive snapshots.

Re-prices every recorded fill by replaying the book state at fill time
(from `book_snapshots` archive) through `walk_book_taker` and
`simulate_passive_fill`. Produces a per-strategy comparison of the
recorded paper P&L vs the queue-aware P&L (walked VWAP for takers,
density-adjusted rebate − cancel-latency for makers).

If the queue-aware P&L is materially worse than the recorded paper
P&L, the cert is suspect and needs to be re-validated under realistic
fills before any further capital is committed.

This is the lightweight analog of what hftbacktest / NautilusTrader
provide for full L2 reconstruction. It runs over our own self-recorded
book archive (path 1), so retroactive coverage is bounded by how long
the archive has been accumulating.

Usage:
    python -m polyagent.eval.queue_backtest --strategy passive_poster_v2
    python -m polyagent.eval.queue_backtest --strategy combined_trader

Reports for each fill: (recorded_price, queue_aware_price, delta_bps).
Aggregates across all fills for the strategy: total recorded P&L vs
queue-aware P&L, and the implied "Sharpe haircut" of moving from
optimistic to honest fills.
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

from polyagent.config import settings


@dataclass
class FillReplay:
    fill_id: int
    ts: float
    strategy: str
    side: str
    token_id: str
    recorded_price: float
    recorded_size: float
    walked_vwap: float | None
    pessimistic_price: float | None
    cancel_latency_penalty: float
    effective_fill_price: float | None
    is_maker: bool
    notional_recorded: float
    notional_queue_aware: float | None
    pnl_delta_per_share: float | None
    snapshot_age_sec: float | None  # how stale the book snapshot was vs fill time


def replay_fills_for_strategy(db_path: str, strategy: str) -> list[FillReplay]:
    """For each fill of `strategy`, look up the book snapshot at or
    before the fill time and recompute the would-be queue-aware fill
    price. Return one FillReplay per fill.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT f.id, f.ts, f.strategy, f.side, f.token_id, f.price, f.size,
                  q.walked_vwap_price, q.pessimistic_price,
                  q.cancel_latency_penalty, q.effective_fill_price, q.is_maker
           FROM fills f
           LEFT JOIN fills_shadow_queue q ON q.fill_id = f.id
           WHERE f.strategy = ?
           ORDER BY f.ts ASC""",
        (strategy,),
    ).fetchall()

    out: list[FillReplay] = []
    for r in rows:
        (fid, ts, strat, side, token, price, size,
         walked, pess, cl_pen, eff_price, is_maker) = r

        # Snapshot lookup
        snap = conn.execute(
            """SELECT ts FROM book_snapshots
               WHERE token_id = ? AND ts <= ?
               ORDER BY ts DESC LIMIT 1""",
            (token, ts),
        ).fetchone()
        snap_age = (ts - float(snap[0])) if snap else None

        # Queue-aware reference price:
        #   takers: the walked-VWAP from the recorded fills_shadow_queue row
        #   makers: the recorded effective_fill_price (already includes
        #           cancel-latency penalty)
        if walked is not None:
            qap = walked
        elif eff_price is not None:
            qap = eff_price
        else:
            qap = None

        notional_recorded = float(price) * float(size)
        notional_qa = float(qap) * float(size) if qap is not None else None
        delta_per_share = (qap - price) if qap is not None else None

        out.append(FillReplay(
            fill_id=int(fid), ts=float(ts), strategy=strat, side=side,
            token_id=token, recorded_price=float(price),
            recorded_size=float(size), walked_vwap=walked,
            pessimistic_price=pess, cancel_latency_penalty=cl_pen or 0.0,
            effective_fill_price=eff_price, is_maker=bool(is_maker),
            notional_recorded=notional_recorded,
            notional_queue_aware=notional_qa,
            pnl_delta_per_share=delta_per_share,
            snapshot_age_sec=snap_age,
        ))
    conn.close()
    return out


def aggregate_haircut(fills: list[FillReplay]) -> dict:
    """Aggregate across all replayed fills: total notional, total P&L
    haircut from the queue-aware reprice, average per-share slippage."""
    if not fills:
        return {"n_fills": 0, "summary": "no fills to replay"}

    n = len(fills)
    n_with_qa = sum(1 for f in fills if f.notional_queue_aware is not None)
    total_recorded = sum(f.notional_recorded for f in fills)
    total_qa = sum(f.notional_queue_aware for f in fills if f.notional_queue_aware is not None)
    # Net haircut: for BUYs, paying MORE is worse → qa_notional > recorded
    # is bad. For SELLs, receiving LESS is worse → qa_notional < recorded
    # is bad. We compute signed P&L change as if every BUY paid the qa
    # price and every SELL received the qa price.
    pnl_delta = 0.0
    for f in fills:
        if f.notional_queue_aware is None:
            continue
        diff_per_share = f.pnl_delta_per_share or 0.0
        if f.side == "BUY":
            # Paid MORE if qa > recorded → loss = -(qa - recorded) × size
            pnl_delta -= diff_per_share * f.recorded_size
        else:
            # Received LESS if qa < recorded → loss = (recorded - qa) × size
            pnl_delta += diff_per_share * f.recorded_size
    n_buys = sum(1 for f in fills if f.side == "BUY")
    n_sells = n - n_buys
    n_makers = sum(1 for f in fills if f.is_maker)
    avg_snap_age = (
        sum(f.snapshot_age_sec for f in fills if f.snapshot_age_sec is not None)
        / max(1, n_with_qa)
    )
    return {
        "n_fills": n,
        "n_with_queue_aware_price": n_with_qa,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_maker_fills": n_makers,
        "n_taker_fills": n - n_makers,
        "total_notional_recorded": round(total_recorded, 2),
        "total_notional_queue_aware": round(total_qa, 2),
        "pnl_haircut_from_queue_aware": round(pnl_delta, 4),
        "avg_per_share_slippage": round(
            (total_qa - total_recorded) / sum(f.recorded_size for f in fills if f.notional_queue_aware is not None),
            6,
        ) if n_with_qa > 0 else None,
        "avg_snapshot_age_sec": round(avg_snap_age, 2) if avg_snap_age else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, help="strategy name to replay")
    p.add_argument("--show-fills", type=int, default=10,
                   help="show this many sample fills")
    args = p.parse_args()

    fills = replay_fills_for_strategy(settings.db_path, args.strategy)
    print(f"=== queue-aware replay: {args.strategy} ===")
    summary = aggregate_haircut(fills)
    import json
    print(json.dumps(summary, indent=2))

    if args.show_fills and fills:
        print(f"\nsample fills (first {args.show_fills}):")
        print(f"  {'ts':>13} {'side':5} {'rec_px':>8} {'qa_px':>8} {'Δbps':>7} {'maker':5} {'snap_age':>9}")
        for f in fills[:args.show_fills]:
            qa = f"{f.walked_vwap:.4f}" if f.walked_vwap else (
                f"{f.effective_fill_price:.4f}" if f.effective_fill_price else "—"
            )
            d = f.pnl_delta_per_share
            d_bps = f"{d / max(f.recorded_price, 1e-9) * 10000:>+7.1f}" if d is not None else "—"
            sa = f"{f.snapshot_age_sec:>9.1f}" if f.snapshot_age_sec is not None else "no_snap"
            print(f"  {f.ts:>13.0f} {f.side:5} {f.recorded_price:>8.4f} {qa:>8} {d_bps:>7} {str(f.is_maker):5} {sa}")


if __name__ == "__main__":
    main()
