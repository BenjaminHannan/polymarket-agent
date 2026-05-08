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

log = structlog.get_logger()


async def stop_loss_loop(
    broker: PaperBroker,
    book_store: BookStore,
    threshold_pct: float = 0.40,    # sell if mark drops 40%+ from entry
    min_loss_usd: float = 5.0,      # don't bother on tiny positions
    interval_sec: float = 60.0,
    longshot_price: float = 0.10,   # below this, raise threshold to avoid noise
    longshot_threshold_pct: float = 0.70,
) -> None:
    """Periodically sweep open positions and dump any whose mark has fallen
    >= threshold_pct from avg_cost. Sells at best_bid; logs the action.

    Caps small wounds: if the unrealized loss is < min_loss_usd, don't bother.
    Longshot fix: a 40% drop on a $0.05 token is just 2c of book noise (1c
    tick); raise the threshold to ``longshot_threshold_pct`` for entries
    below ``longshot_price`` so we don't trip on quote-noise alone.
    """
    log.info("stop_loss_start", threshold_pct=threshold_pct, interval_sec=interval_sec)
    while True:
        await asyncio.sleep(interval_sec)
        try:
            tokens_to_sell: list[tuple[str, str, float]] = []
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
                drop = (pos.avg_cost - bid_price) / pos.avg_cost
                # Choose the right threshold for this token's price regime.
                eff_threshold = (
                    longshot_threshold_pct if pos.avg_cost < longshot_price
                    else threshold_pct
                )
                if drop < eff_threshold:
                    continue
                unrealized = (bid_price - pos.avg_cost) * pos.size
                if unrealized > -min_loss_usd:
                    continue
                tokens_to_sell.append((token_id, "stop_loss", pos.size))
                log.warning(
                    "stop_loss_triggered",
                    token_id=token_id[:14],
                    avg_cost=round(pos.avg_cost, 4),
                    bid=round(bid_price, 4),
                    drop_pct=round(drop * 100, 2),
                    unrealized=round(unrealized, 2),
                )
            for token_id, _, sz in tokens_to_sell:
                # condition_id we don't have here; pass empty — broker's submit
                # only uses it for the SQL row.
                await broker.submit(
                    strategy="stop_loss",
                    condition_id="",
                    token_id=token_id,
                    side="SELL",
                    max_size=sz,
                    max_price=None,
                    reason=f"stop_loss drop>={threshold_pct:.0%}",
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
