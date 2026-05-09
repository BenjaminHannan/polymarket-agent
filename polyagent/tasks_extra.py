"""Extra periodic tasks: stop-loss, status digest, signal pruning.

Each is a small async function meant to run under the supervisor."""

from __future__ import annotations

import asyncio
import time

import aiosqlite
import structlog

from polyagent.config import settings
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker
from polyagent.risk.exit_policy import KaminskiLoStopGate, NearResolutionLockIn

log = structlog.get_logger()


async def stop_loss_loop(
    broker: PaperBroker,
    book_store: BookStore,
    threshold_pct: float = 0.40,    # sell if mark drops 40%+ from entry
    min_loss_usd: float = 5.0,      # don't bother on tiny positions
    interval_sec: float = 60.0,
    longshot_price: float = 0.10,   # below this, raise threshold to avoid noise
    longshot_threshold_pct: float = 0.70,
    kaminski_lo_gate: KaminskiLoStopGate | None = None,
    near_resolution: NearResolutionLockIn | None = None,
    held_tracker=None,              # HeldMarketTracker for TTR lookups
) -> None:
    """Periodically sweep open positions. The exit policy is the §10
    redesign:

      1. Near-resolution lock-in: if our position is materially in
         profit and TTR is short, close to lock the gain.
      2. Kaminski-Lo gated price stop: only fire the standard
         drop>=threshold stop if phi >= SR_daily on the realized
         resolved-trade returns. Otherwise the stop has been removing
         mean per K-L 2014 and we skip it.
      3. Longshot deleverage (unconditional): for tokens entered below
         ``longshot_price``, fire at ``longshot_threshold_pct`` drop
         regardless. This is a "we made a clear mistake on a longshot"
         exit, not a Sharpe question.

    Caps small wounds: if the unrealized loss is < min_loss_usd, don't bother.
    """
    log.info(
        "stop_loss_start",
        threshold_pct=threshold_pct,
        interval_sec=interval_sec,
        kaminski_lo_enabled=kaminski_lo_gate is not None,
        near_resolution_enabled=near_resolution is not None,
    )
    while True:
        await asyncio.sleep(interval_sec)
        try:
            # Refresh K-L estimates hourly (cached internally).
            if kaminski_lo_gate is not None:
                try:
                    kaminski_lo_gate.maybe_refresh()
                except Exception as e:
                    log.warning("kaminski_lo_refresh_loop_error", err=str(e))
            kl_stops_enabled = (
                kaminski_lo_gate.stops_enabled() if kaminski_lo_gate is not None else True
            )
            tokens_to_sell: list[tuple[str, str, float, str]] = []
            for token_id, pos in list(broker.positions.items()):
                if pos.size <= 0 or pos.avg_cost <= 0:
                    continue
                book = book_store.books.get(token_id)
                if book is None:
                    continue
                bid = book.best_bid()
                if bid is None:
                    continue
                bid_price = bid[0]

                # 1. Near-resolution lock-in for winners.
                if near_resolution is not None:
                    ttr_hours = None
                    if held_tracker is not None:
                        m = held_tracker.by_token.get(token_id)
                        if m is not None and m.end_date_iso:
                            from polyagent.gamma import days_to_resolution as _ttr
                            d = _ttr(m.end_date_iso)
                            if d is not None:
                                ttr_hours = d * 24.0
                    do_exit, reason = near_resolution.should_exit(
                        avg_cost=pos.avg_cost,
                        size=pos.size,
                        bid=bid_price,
                        hours_to_resolution=ttr_hours,
                    )
                    if do_exit:
                        tokens_to_sell.append((token_id, "near_resolution", pos.size, reason or ""))
                        log.info(
                            "near_resolution_lock_in",
                            token_id=token_id[:14],
                            avg_cost=round(pos.avg_cost, 4),
                            bid=round(bid_price, 4),
                            ttr_hours=round(ttr_hours, 2) if ttr_hours is not None else None,
                        )
                        continue

                drop = (pos.avg_cost - bid_price) / pos.avg_cost

                # 2. Longshot deleverage: unconditional, even if K-L
                #    says general stops should be off. A $0.05 token
                #    that has dropped 70% is a clear-mistake exit, not
                #    a Sharpe-tunable stop.
                if pos.avg_cost < longshot_price:
                    if drop < longshot_threshold_pct:
                        continue
                    unrealized = (bid_price - pos.avg_cost) * pos.size
                    if unrealized > -min_loss_usd:
                        continue
                    tokens_to_sell.append((token_id, "stop_loss", pos.size, "longshot_deleverage"))
                    log.warning(
                        "stop_loss_triggered",
                        token_id=token_id[:14],
                        avg_cost=round(pos.avg_cost, 4),
                        bid=round(bid_price, 4),
                        drop_pct=round(drop * 100, 2),
                        unrealized=round(unrealized, 2),
                        kind="longshot_deleverage",
                    )
                    continue

                # 3. Standard price stop — gated on Kaminski-Lo.
                if not kl_stops_enabled:
                    # phi < SR_daily; the stop is removing mean per K-L 2014.
                    continue
                if drop < threshold_pct:
                    continue
                unrealized = (bid_price - pos.avg_cost) * pos.size
                if unrealized > -min_loss_usd:
                    continue
                tokens_to_sell.append((token_id, "stop_loss", pos.size, "kaminski_lo_passed"))
                log.warning(
                    "stop_loss_triggered",
                    token_id=token_id[:14],
                    avg_cost=round(pos.avg_cost, 4),
                    bid=round(bid_price, 4),
                    drop_pct=round(drop * 100, 2),
                    unrealized=round(unrealized, 2),
                    kind="kaminski_lo_passed",
                )
            for token_id, strategy, sz, reason in tokens_to_sell:
                # condition_id we don't have here; pass empty — broker's submit
                # only uses it for the SQL row.
                await broker.submit(
                    strategy=strategy,
                    condition_id="",
                    token_id=token_id,
                    side="SELL",
                    max_size=sz,
                    max_price=None,
                    reason=reason or strategy,
                )
        except Exception as e:
            log.warning("stop_loss_loop_error", err=str(e))


async def status_digest_loop(
    broker: PaperBroker,
    book_store: BookStore,
    interval_sec: float = 300.0,
) -> None:
    """One-line digest log every interval_sec. Compact summary of NAV,
    drawdown, fills, throttle state, and book health."""
    log.info("status_digest_start", interval_sec=interval_sec)
    while True:
        await asyncio.sleep(interval_sec)
        try:
            s = broker.summary()
            books_quoted = sum(
                1 for b in book_store.books.values() if b.best_bid() and b.best_ask()
            )
            log.info(
                "digest",
                nav_liq=s["nav_liquidation"],
                pnl_pct=s["pnl_pct_liquidation"],
                cash=s["cash"],
                positions=s["open_positions"],
                fills=s["fills"],
                realized=s["realized_pnl"],
                unrealized=s["unrealized_pnl_liquidation"],
                drawdown_pct=s["drawdown_pct"],
                hwm=s["hwm"],
                max_pos_pct=s["max_pos_pct"],
                killed=s["killed"],
                books_quoted=books_quoted,
            )
        except Exception as e:
            log.warning("status_digest_error", err=str(e))


async def signal_prune_loop(
    db_path: str,
    keep_days: int = 30,
    interval_sec: float = 86400.0,
) -> None:
    """Daily prune of `signals` table older than `keep_days`. Keeps the DB lean.

    Resolutions, signal_outcomes, and news are NOT pruned — those are training
    data."""
    log.info("signal_prune_start", keep_days=keep_days, interval_sec=interval_sec)
    while True:
        await asyncio.sleep(interval_sec)
        try:
            cutoff = time.time() - keep_days * 86400
            db = await aiosqlite.connect(db_path, timeout=30.0)
            await db.execute("PRAGMA busy_timeout=10000")
            cur = await db.execute("DELETE FROM signals WHERE ts < ?", (cutoff,))
            n_deleted = cur.rowcount or 0
            await db.commit()
            await db.close()
            log.info("signal_prune_done", n_deleted=n_deleted)
        except Exception as e:
            log.warning("signal_prune_error", err=str(e))
