"""Paper broker: simulates fills against the live top-of-book.

Treats every paper order as a marketable take at the current best opposite-side
price. Tracks cash, positions per token_id, and realized/unrealized P&L. All
state is mirrored to SQLite for inspection between runs.

Also handles paper-mode settlement: when a market resolves, each held outcome
token is paid 0 or 1 USDC and removed from the position book, with the entire
realized P&L logged to the `resolutions` table for downstream labeling.

Concurrency: all mutating operations (submit, settle_market, snapshot_nav)
are guarded by a single asyncio.Lock so concurrent strategy invocations
can't race on cash / positions.

State recovery: on open(), the broker replays all historical fills and
resolutions from SQLite to reconstruct in-memory cash + positions. So a
crash + restart resumes from where it left off, not from the starting NAV.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import aiosqlite
import structlog

from polyagent.config import settings
from polyagent.orderbook import BookStore
from polyagent.queue_model import pessimistic_fill_price
from polyagent.risk.drawdown import DrawdownTracker
from polyagent.risk.kill_switch import is_killed
from polyagent.risk.latency_model import LatencyTracker

log = structlog.get_logger()


@dataclass
class Position:
    token_id: str
    size: float = 0.0
    avg_cost: float = 0.0  # average paid per share, $0..$1


@dataclass
class PaperBroker:
    book_store: BookStore
    nav_start: float = settings.starting_nav
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fills: int = 0
    db: aiosqlite.Connection | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    settled_conditions: set[str] = field(default_factory=set)
    drawdown: DrawdownTracker = field(default_factory=DrawdownTracker.load)
    latency: LatencyTracker = field(default_factory=LatencyTracker)
    # Pessimistic-NAV deployment gate state. Refreshed periodically by
    # _refresh_pessimistic_gate(); BUYs are blocked when slippage burn over
    # the recent window crosses pessimistic_block_pct of starting NAV.
    _pess_gate_blocked: bool = False
    _pess_gate_last_check: float = 0.0
    # Engage protection sooner — diagnostic showed 22% slippage burn on
    # the first 100 fills, well above the 5% block threshold.
    _pess_gate_window_fills: int = 100
    _pess_gate_block_pct: float = 0.05  # 5% of starting NAV
    # Tokens that recently took a stop-loss. Strategies that read this set
    # via `was_recently_stopped()` won't re-buy them — the model's
    # "edge" persisted through the loss is exactly the same hallucination
    # that produced the loss in the first place.
    recently_stopped: dict[str, float] = field(default_factory=dict)
    stop_loss_blacklist_sec: float = 86400.0   # 24h
    # Hard per-token fill cap. Any strategy is allowed at most
    # ``max_buys_per_token_window`` BUYs per token within
    # ``buys_per_token_window_sec``. Prevents the cycling-fill pattern
    # where a single token accumulates 20+ fills as the model keeps
    # emitting the same edge claim. Counter is reset by stop-losses
    # (so we can re-enter after a clean exit) but otherwise bounded.
    _buys_per_token: dict[str, list[float]] = field(default_factory=dict)
    max_buys_per_token_window: int = 2
    buys_per_token_window_sec: float = 86400.0

    def __post_init__(self) -> None:
        self.cash = self.nav_start

    async def open(self) -> None:
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(settings.db_path, timeout=30.0)
        # Concurrency-friendly settings: WAL lets readers + writers coexist;
        # NORMAL sync trades a tiny durability window for ~10x write throughput;
        # busy_timeout backs off automatically when another connection holds the lock.
        await self.db.execute("PRAGMA journal_mode=WAL")
        await self.db.execute("PRAGMA synchronous=NORMAL")
        await self.db.execute("PRAGMA busy_timeout=10000")
        await self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                strategy TEXT,
                condition_id TEXT,
                token_id TEXT,
                side TEXT,
                price REAL,
                size REAL,
                notional REAL,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS fills_ts ON fills(ts);

            CREATE TABLE IF NOT EXISTS resolutions (
                condition_id TEXT PRIMARY KEY,
                resolved_ts REAL,
                yes_won INTEGER,
                yes_token_id TEXT,
                no_token_id TEXT,
                yes_size REAL,
                no_size REAL,
                yes_avg_cost REAL,
                no_avg_cost REAL,
                yes_payout REAL,
                no_payout REAL,
                pnl REAL,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS resolutions_ts ON resolutions(resolved_ts);

            -- Shadow ledger: every paper fill recorded with both the
            -- optimistic VWAP price (what we record) and a pessimistic
            -- price that adds full half-spread + queue-loss penalty.
            -- Used to compute a worst-case realized P&L; if pessimistic
            -- P&L is positive, the strategy is robust to execution costs.
            CREATE TABLE IF NOT EXISTS fills_shadow (
                fill_id INTEGER PRIMARY KEY,           -- FK to fills.id
                vwap_price REAL,
                pessimistic_price REAL,
                half_spread REAL,
                size REAL,
                slippage_estimate REAL
            );
            CREATE INDEX IF NOT EXISTS fills_shadow_fid ON fills_shadow(fill_id);

            -- Queue-aware shadow ledger (pmwhy.md §B2). Each fill is
            -- re-priced under three honest models so we can compare
            -- cumulative slippage by strategy: top-of-book optimistic
            -- (what fills_shadow.vwap_price already records), the
            -- multi-level walked-VWAP, and the closed-form
            -- pessimistic with cancel-latency drift from queue_model.
            -- The slippage_bps_walked column is the key metric for
            -- re-validating certs under realistic taker fills.
            CREATE TABLE IF NOT EXISTS fills_shadow_queue (
                fill_id INTEGER PRIMARY KEY,
                top_of_book_price REAL,
                walked_vwap_price REAL,
                pessimistic_price REAL,
                size REAL,
                levels_walked INTEGER,
                partial INTEGER,
                slippage_bps_walked REAL,
                slippage_bps_pess REAL,
                is_maker INTEGER DEFAULT 0,
                taker_fee_paid REAL DEFAULT 0,
                maker_rebate_credited REAL DEFAULT 0,
                cancel_latency_penalty REAL DEFAULT 0,
                effective_fill_price REAL
            );
            CREATE INDEX IF NOT EXISTS fills_shadow_queue_fid ON fills_shadow_queue(fill_id);
            """
        )
        await self.db.commit()
        await self._migrate_nav_history()
        await self._migrate_fills_shadow_queue()
        # Eagerly create the round_trip_legs and book_snapshots tables
        # so dashboard / inspection queries can target them before the
        # first fill / periodic snapshot lands.
        try:
            import sqlite3 as _sql
            from polyagent.risk.round_trips import ensure_table as _rt_ensure
            from polyagent.risk.book_archive import ensure_table as _ba_ensure
            _c = _sql.connect(settings.db_path)
            try:
                _rt_ensure(_c)
                _ba_ensure(_c)
            finally:
                _c.close()
        except Exception as e:
            log.warning("auxiliary_table_init_failed", err=str(e))
        await self._recover_state()

    async def _migrate_fills_shadow_queue(self) -> None:
        """Add fee/rebate/maker columns to fills_shadow_queue if upgrading
        from a pre-fee schema. Idempotent: if the columns already exist
        the ADD COLUMN raises and we swallow."""
        if self.db is None:
            return
        async with self.db.execute("PRAGMA table_info(fills_shadow_queue)") as cur:
            cols = {r[1] for r in await cur.fetchall()}
        for col, ddl in [
            ("is_maker", "ALTER TABLE fills_shadow_queue ADD COLUMN is_maker INTEGER DEFAULT 0"),
            ("taker_fee_paid", "ALTER TABLE fills_shadow_queue ADD COLUMN taker_fee_paid REAL DEFAULT 0"),
            ("maker_rebate_credited", "ALTER TABLE fills_shadow_queue ADD COLUMN maker_rebate_credited REAL DEFAULT 0"),
            ("cancel_latency_penalty", "ALTER TABLE fills_shadow_queue ADD COLUMN cancel_latency_penalty REAL DEFAULT 0"),
            ("effective_fill_price", "ALTER TABLE fills_shadow_queue ADD COLUMN effective_fill_price REAL"),
        ]:
            if col not in cols:
                try:
                    await self.db.execute(ddl)
                except Exception as e:
                    log.warning("fills_shadow_queue_migrate_failed", col=col, err=str(e))
        await self.db.commit()

    async def _migrate_nav_history(self) -> None:
        """Older versions had `ts` as PRIMARY KEY, which loses snapshots when
        two land in the same second. Migrate to autoincrement `id` PK
        preserving all rows."""
        if self.db is None:
            return
        async with self.db.execute("PRAGMA table_info(nav_history)") as cur:
            cols = [(r[1], int(r[5] or 0)) for r in await cur.fetchall()]
        col_names = {c for c, _ in cols}
        # Old schema = (ts REAL PK, cash, position_value, nav). New schema has 'id'.
        if "id" in col_names:
            return
        if not col_names:
            # Table didn't exist; create fresh.
            await self.db.execute(
                "CREATE TABLE nav_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts REAL, cash REAL, position_value REAL, nav REAL)"
            )
            await self.db.execute("CREATE INDEX nav_history_ts ON nav_history(ts)")
            await self.db.commit()
            return
        log.info("migrating_nav_history_schema")
        await self.db.executescript(
            """
            ALTER TABLE nav_history RENAME TO nav_history_old;
            CREATE TABLE nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                cash REAL,
                position_value REAL,
                nav REAL
            );
            CREATE INDEX nav_history_ts ON nav_history(ts);
            INSERT INTO nav_history(ts, cash, position_value, nav)
              SELECT ts, cash, position_value, nav FROM nav_history_old;
            DROP TABLE nav_history_old;
            """
        )
        await self.db.commit()

    async def _recover_state(self) -> None:
        """Replay fills + resolutions from SQLite to rebuild in-memory state.

        The replay applies the same arithmetic as submit() and settle_market()
        but doesn't write to the DB. After this runs, cash + positions match
        what the broker would have had if it never restarted.
        """
        if self.db is None:
            return

        # Build a set of settled condition_ids first; they pre-zero the relevant tokens.
        async with self.db.execute(
            "SELECT condition_id, yes_token_id, no_token_id, yes_won, "
            "yes_size, no_size, yes_avg_cost, no_avg_cost, resolved_ts FROM resolutions"
        ) as cur:
            settled: list[dict] = [
                {
                    "cid": r[0],
                    "yes_token": r[1],
                    "no_token": r[2],
                    "yes_won": bool(r[3]),
                    "yes_size": float(r[4] or 0),
                    "no_size": float(r[5] or 0),
                    "yes_cost": float(r[6] or 0),
                    "no_cost": float(r[7] or 0),
                    "ts": float(r[8] or 0),
                }
                async for r in cur
            ]
        for s in settled:
            self.settled_conditions.add(s["cid"])

        # Replay all fills in time order. For fills on conditions that later resolved,
        # their positions get zeroed out by the resolution replay; until then they
        # contribute to size + avg_cost.
        async with self.db.execute(
            "SELECT condition_id, token_id, side, price, size, notional, ts "
            "FROM fills ORDER BY ts ASC, id ASC"
        ) as cur:
            async for cid, token_id, side, price, size, notional, _ts in cur:
                size = float(size or 0)
                price = float(price or 0)
                notional = float(notional or 0)
                if size <= 0:
                    continue
                if (side or "").upper() == "BUY":
                    self.cash -= notional
                    pos = self._pos(token_id)
                    new_size = pos.size + size
                    if new_size > 0:
                        pos.avg_cost = (
                            (pos.avg_cost * pos.size) + (price * size)
                        ) / new_size
                    pos.size = new_size
                elif (side or "").upper() == "SELL":
                    self.cash += notional
                    pos = self._pos(token_id)
                    pos.size -= size
                    if pos.size <= 1e-9:
                        pos.size = 0.0
                        pos.avg_cost = 0.0
                self.fills += 1

        # Now apply resolutions: pay $1 to winning legs, zero positions.
        for s in settled:
            yes_pos = self.positions.get(s["yes_token"])
            no_pos = self.positions.get(s["no_token"])
            yes_payout = 1.0 if s["yes_won"] else 0.0
            no_payout = 0.0 if s["yes_won"] else 1.0
            if yes_pos and yes_pos.size > 0:
                self.cash += yes_pos.size * yes_payout
                self.realized_pnl += (yes_payout - yes_pos.avg_cost) * yes_pos.size
                yes_pos.size = 0.0
                yes_pos.avg_cost = 0.0
            if no_pos and no_pos.size > 0:
                self.cash += no_pos.size * no_payout
                self.realized_pnl += (no_payout - no_pos.avg_cost) * no_pos.size
                no_pos.size = 0.0
                no_pos.avg_cost = 0.0

        n_open = sum(1 for p in self.positions.values() if p.size > 0)
        log.info(
            "broker_state_recovered",
            cash=round(self.cash, 2),
            realized_pnl=round(self.realized_pnl, 2),
            open_positions=n_open,
            historical_fills=self.fills,
            settled_markets=len(settled),
        )

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    def _pos(self, token_id: str) -> Position:
        p = self.positions.get(token_id)
        if p is None:
            p = Position(token_id=token_id)
            self.positions[token_id] = p
        return p

    def _mark(self, p: Position, *, prefer: str) -> float:
        """Pick a price to mark a position at.
        prefer="bid" -> liquidation value (what we'd receive selling now)
        prefer="mid" -> neutral fair value
        prefer="ask" -> replacement cost (what it'd cost to rebuild the position)
        Falls through gracefully if the preferred side isn't quoted.
        """
        book = self.book_store.books.get(p.token_id)
        if book is None:
            return p.avg_cost
        bid = book.best_bid()
        ask = book.best_ask()
        if prefer == "bid":
            if bid is not None:
                return bid[0]
            mid = book.mid()
            if mid is not None:
                return mid
            return book.last_trade_price if book.last_trade_price is not None else p.avg_cost
        if prefer == "ask":
            if ask is not None:
                return ask[0]
            mid = book.mid()
            if mid is not None:
                return mid
            return book.last_trade_price if book.last_trade_price is not None else p.avg_cost
        # mid (default)
        mid = book.mid()
        if mid is not None:
            return mid
        if bid is not None:
            return bid[0]
        if ask is not None:
            return ask[0]
        if book.last_trade_price is not None:
            return book.last_trade_price
        return p.avg_cost

    def position_value(self, mark: str = "mid") -> float:
        """Total $ value of open positions.

        mark="mid" (default): neutral fair-value mark.
        mark="bid": liquidation value — what we'd actually realize if we
            sold every position right now at the best bid. This is the
            honest mark for "how much have I made."
        mark="ask": replacement cost.
        """
        total = 0.0
        for p in self.positions.values():
            if p.size <= 0:
                continue
            total += p.size * self._mark(p, prefer=mark)
        return total

    def unrealized_pnl(self, mark: str = "bid") -> float:
        """Open-position P&L vs entry cost. mark="bid" matches liquidation."""
        total = 0.0
        for p in self.positions.values():
            if p.size <= 0:
                continue
            total += (self._mark(p, prefer=mark) - p.avg_cost) * p.size
        return total

    def nav(self, mark: str = "mid") -> float:
        return self.cash + self.position_value(mark=mark)

    async def submit(
        self,
        *,
        strategy: str,
        condition_id: str,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        max_size: float,
        max_price: float | None = None,
        reason: str = "",
        is_maker: bool = False,
        category: str | None = None,
    ) -> float:
        """Simulate a marketable order. Returns filled size (number of shares).

        Concurrency-safe: serializes via self._lock so concurrent strategies
        can't race on cash / positions.

        is_maker — when True, fee accounting credits a maker rebate and
        applies a cancel-latency penalty to the recorded effective price
        (Polygon block ≈ 2 s of adverse drift). Default False (taker).
        category — passed in by callers that know it (e.g. passive_poster_v2);
        used by the per-category fee curve in polyagent.risk.fees.
        """
        async with self._lock:
            return await self._submit_locked(
                strategy=strategy,
                condition_id=condition_id,
                token_id=token_id,
                side=side,
                max_size=max_size,
                max_price=max_price,
                reason=reason,
                is_maker=is_maker,
                category=category,
            )

    async def _submit_locked(
        self,
        *,
        strategy: str,
        condition_id: str,
        token_id: str,
        side: str,
        max_size: float,
        max_price: float | None,
        reason: str,
        is_maker: bool = False,
        category: str | None = None,
    ) -> float:
        side = side.upper()
        # Kill switch: refuse all new BUYs if data/.STOP exists. SELLs go through
        # so we can still close positions during a halt.
        if is_killed() and side == "BUY":
            log.info("kill_switch_blocked", strategy=strategy, condition_id=condition_id)
            return 0.0
        # Pessimistic-NAV deployment gate: block new BUYs when shadow-ledger
        # slippage burn over the recent window exceeds the configured pct of
        # starting NAV. SELLs are always allowed so we can still de-risk.
        if self._pess_gate_blocked and side == "BUY":
            log.info(
                "pessimistic_gate_blocked",
                strategy=strategy,
                condition_id=condition_id,
            )
            return 0.0
        # Reject trades on already-settled markets — they'd just sit at zero.
        if condition_id in self.settled_conditions:
            return 0.0
        book = self.book_store.books.get(token_id)
        if book is None:
            return 0.0

        # VWAP-aware fill: walk the book up to max_size, capped by max_price.
        # Real fills don't only hit depth-1; they consume multiple levels and
        # the average fill price is worse than top-of-book. This matches what
        # would actually happen on a CLOB take.
        if side == "BUY":
            levels = sorted(book.asks.items())  # ascending price
        elif side == "SELL":
            levels = sorted(book.bids.items(), reverse=True)  # descending price
        else:
            return 0.0

        remaining = max_size
        total_size = 0.0
        total_notional = 0.0
        for px, sz in levels:
            if remaining <= 0:
                break
            if max_price is not None:
                if side == "BUY" and px > max_price + 1e-12:
                    break
                if side == "SELL" and px < max_price - 1e-12:
                    break
            take = min(sz, remaining)
            if take <= 0:
                continue
            total_size += take
            total_notional += take * px
            remaining -= take

        if total_size <= 0:
            return 0.0

        size = total_size
        notional = total_notional
        # Effective VWAP price (used for accounting; book is consumed
        # multi-level so a single "price" isn't a real thing here).
        price = notional / size

        if side == "BUY":
            if notional > self.cash + 1e-6:
                size = max(0.0, self.cash / price)
                notional = size * price
                if size <= 0:
                    return 0.0
            self.cash -= notional
            pos = self._pos(token_id)
            new_size = pos.size + size
            if new_size > 0:
                pos.avg_cost = ((pos.avg_cost * pos.size) + (price * size)) / new_size
            pos.size = new_size
        else:  # SELL
            pos = self._pos(token_id)
            if pos.size <= 0:
                return 0.0
            size = min(size, pos.size)
            notional = size * price
            self.cash += notional
            self.realized_pnl += (price - pos.avg_cost) * size
            pos.size -= size
            if pos.size <= 1e-9:
                pos.size = 0.0
                pos.avg_cost = 0.0

        self.fills += 1
        ts = time.time()
        # Stop-loss memo: any token that takes a stop_loss SELL goes into
        # the recently-stopped set so other strategies refuse to re-enter
        # it for ``stop_loss_blacklist_sec``. Same-token, same-condition
        # too — see was_recently_stopped().
        if strategy == "stop_loss" and side == "SELL":
            self.recently_stopped[token_id] = ts
            if condition_id:
                self.recently_stopped[condition_id] = ts
        # Per-token BUY counter for the hard fill-cap. Resets only via
        # window expiry, NOT on stop-loss — a stopped token has bigger
        # problems than the fill cap (it's blacklisted by recently_stopped).
        if side == "BUY":
            self._buys_per_token.setdefault(token_id, []).append(ts)
        # Compute pessimistic (worst-case) execution price for shadow ledger
        # using the Cont/Kukanov-style queue-loss + slippage model.
        pessimistic_price = price
        half_spread = 0.0
        slippage = 0.0
        try:
            ba = book.best_ask()
            bb = book.best_bid()
            if ba is not None and bb is not None:
                half_spread = (ba[0] - bb[0]) / 2.0
                depth_at_top = ba[1] if side == "BUY" else bb[1]
                rv = book.realized_vol(300) if hasattr(book, "realized_vol") else None
                pp = pessimistic_fill_price(
                    side=side,
                    best_bid=bb[0],
                    best_ask=ba[0],
                    book_size_consumed=size,
                    book_depth_at_top=depth_at_top,
                    queue_loss_bps=50.0,
                    realized_vol=rv,
                )
                if pp is not None:
                    pessimistic_price = pp
                    if side == "BUY":
                        slippage = max(0.0, pp - ba[0])
                    else:
                        slippage = max(0.0, bb[0] - pp)
        except Exception:
            pessimistic_price = price

        # Record book-age sample for latency p99 (per-source: WSS book stream)
        if book.last_update_ts is not None:
            self.latency.record(time.time() - book.last_update_ts, source="wss_book")

        if self.db is not None:
            cur = await self.db.execute(
                "INSERT INTO fills(ts, strategy, condition_id, token_id, side, price, size, notional, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, strategy, condition_id, token_id, side, price, size, notional, reason),
            )
            fill_id = cur.lastrowid
            await self.db.execute(
                "INSERT INTO fills_shadow(fill_id, vwap_price, pessimistic_price, half_spread, size, slippage_estimate) VALUES (?,?,?,?,?,?)",
                (fill_id, price, pessimistic_price, half_spread, size, slippage),
            )
            # Queue-aware shadow ledger (pmwhy.md §B2): re-walk the book
            # honestly to record what the multi-level VWAP would have
            # been at this moment, alongside the closed-form pessimistic.
            # This lets us re-validate certs under realistic taker fills.
            #
            # Also applies maker rebate / taker fee accounting and a
            # cancel-latency penalty on maker fills (Polygon ~2 s block:
            # a resting limit can't dodge adverse mid moves faster than
            # the next block, so paper-mode P&L is honestly discounted
            # by σ × √block_sec on every maker fill).
            try:
                from polyagent.risk.queue_aware_fills import compare_fill_models
                from polyagent.risk.fees import compute_fees, cancel_latency_penalty
                cmp = compare_fill_models(book, side, size)
                # Fees / rebate accounting
                fees = compute_fees(
                    notional=notional,
                    category=category,
                    is_maker=is_maker,
                )
                # Cancel-latency penalty (maker only — takers fill instantly)
                rv = book.realized_vol(300) if hasattr(book, "realized_vol") else None
                if is_maker:
                    eff_price = cancel_latency_penalty(price, side, rv)
                    cl_penalty = abs(eff_price - price) * size
                else:
                    eff_price = price
                    cl_penalty = 0.0
                if cmp.get("available"):
                    await self.db.execute(
                        """INSERT INTO fills_shadow_queue
                           (fill_id, top_of_book_price, walked_vwap_price,
                            pessimistic_price, size, levels_walked, partial,
                            slippage_bps_walked, slippage_bps_pess,
                            is_maker, taker_fee_paid, maker_rebate_credited,
                            cancel_latency_penalty, effective_fill_price)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            fill_id,
                            cmp["top_of_book"],
                            cmp["walked_vwap"],
                            cmp["pessimistic"],
                            cmp["filled_size"],
                            cmp["levels_walked"],
                            int(bool(cmp["partial"])),
                            cmp["slippage_bps_walked"],
                            cmp["slippage_bps_pess"],
                            int(is_maker),
                            fees.taker_fee_paid,
                            fees.maker_rebate_credited,
                            cl_penalty,
                            eff_price,
                        ),
                    )
                else:
                    # Even if compare_fill_models couldn't return a walked
                    # VWAP (empty book), still record fee/rebate so paper
                    # P&L is honest.
                    await self.db.execute(
                        """INSERT INTO fills_shadow_queue
                           (fill_id, top_of_book_price, walked_vwap_price,
                            pessimistic_price, size, levels_walked, partial,
                            slippage_bps_walked, slippage_bps_pess,
                            is_maker, taker_fee_paid, maker_rebate_credited,
                            cancel_latency_penalty, effective_fill_price)
                           VALUES (?, NULL, NULL, NULL, ?, NULL, NULL, NULL, NULL,
                                   ?, ?, ?, ?, ?)""",
                        (
                            fill_id, size,
                            int(is_maker),
                            fees.taker_fee_paid,
                            fees.maker_rebate_credited,
                            cl_penalty,
                            eff_price,
                        ),
                    )
                # Apply fee/rebate to cash so live broker state reflects
                # paper P&L honestly. Taker fees DEBIT cash; maker
                # rebates CREDIT it. Maker rebate is paper-mode bookkeeping;
                # a real-money build would receive USDC daily.
                if fees.taker_fee_paid > 0:
                    self.cash -= fees.taker_fee_paid
                if fees.maker_rebate_credited > 0:
                    self.cash += fees.maker_rebate_credited
                # Book-snapshot archive (path 1 — self-record L2 going
                # forward). Every fill captures the book at fill-time so
                # downstream cert validation can replay queue position
                # under realistic fills. Default off behind ENABLE_BOOK_ARCHIVE.
                try:
                    if os.getenv("ENABLE_BOOK_ARCHIVE", "0") == "1":
                        from polyagent.risk.book_archive import snapshot as _book_snapshot
                        import sqlite3 as _sql
                        _ba_conn = _sql.connect(settings.db_path, timeout=10.0)
                        try:
                            _book_snapshot(_ba_conn, token_id, book, trigger="fill", ts=ts)
                        finally:
                            _ba_conn.close()
                except Exception as e:
                    log.warning("book_archive_fill_snap_failed", err=str(e))

                # Round-trip P&L attribution: pair this fill against
                # earlier opposite-side fills via FIFO matching. Lets us
                # measure realized round-trip P&L per strategy without
                # waiting for market resolution.
                try:
                    from polyagent.risk.round_trips import FillContext, record_fill
                    rt_ctx = FillContext(
                        fill_id=fill_id,
                        strategy=strategy,
                        condition_id=condition_id,
                        token_id=token_id,
                        side=side,
                        price=eff_price,
                        size=size,
                        ts=ts,
                        fees_paid=fees.taker_fee_paid,
                        rebate_credited=fees.maker_rebate_credited,
                    )
                    # Reuse the same aiosqlite db handle by calling its
                    # underlying sync API (round_trips uses sync sqlite).
                    # Easier path: open a short-lived sync connection.
                    import sqlite3 as _sql
                    rt_conn = _sql.connect(settings.db_path, timeout=10.0)
                    try:
                        record_fill(rt_conn, rt_ctx)
                    finally:
                        rt_conn.close()
                except Exception as e:
                    log.warning("round_trip_record_failed", err=str(e))
            except Exception as e:
                log.warning("queue_shadow_write_failed", err=str(e))
            await self.db.commit()

        log.info(
            "paper_fill",
            strategy=strategy,
            condition_id=condition_id,
            token_id=token_id[:12] + "..." if len(token_id) > 12 else token_id,
            side=side,
            price=round(price, 4),
            size=round(size, 2),
            notional=round(notional, 2),
            cash=round(self.cash, 2),
            reason=reason,
        )
        return size

    async def settle_market(
        self,
        *,
        condition_id: str,
        yes_token_id: str,
        no_token_id: str,
        yes_won: bool,
        question: str = "",
        extra: dict | None = None,
    ) -> dict | None:
        """Settle a resolved market. Pays 1.0 to the winning outcome's holders, 0 to the loser.

        Removes both leg positions, books realized P&L into cash, persists a
        labeled row to the `resolutions` table. Idempotent + concurrency-safe:
        re-calling for the same condition_id returns None without double-settling.

        Atomicity: uses INSERT OR IGNORE then checks rowcount, so a concurrent
        settle for the same condition is a true no-op.
        """
        async with self._lock:
            return await self._settle_locked(
                condition_id=condition_id,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_won=yes_won,
                question=question,
                extra=extra,
            )

    async def _settle_locked(
        self,
        *,
        condition_id: str,
        yes_token_id: str,
        no_token_id: str,
        yes_won: bool,
        question: str,
        extra: dict | None,
    ) -> dict | None:
        if condition_id in self.settled_conditions:
            return None
        if self.db is None:
            return None

        yes_pos = self.positions.get(yes_token_id)
        no_pos = self.positions.get(no_token_id)
        yes_size = yes_pos.size if yes_pos else 0.0
        no_size = no_pos.size if no_pos else 0.0
        yes_cost = yes_pos.avg_cost if yes_pos else 0.0
        no_cost = no_pos.avg_cost if no_pos else 0.0

        yes_payout = 1.0 if yes_won else 0.0
        no_payout = 0.0 if yes_won else 1.0

        ts = time.time()
        # Try to write the resolution row first; INSERT OR IGNORE wins the race.
        cur = await self.db.execute(
            """INSERT OR IGNORE INTO resolutions(
                condition_id, resolved_ts, yes_won,
                yes_token_id, no_token_id,
                yes_size, no_size, yes_avg_cost, no_avg_cost,
                yes_payout, no_payout, pnl, detail
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                condition_id,
                ts,
                1 if yes_won else 0,
                yes_token_id,
                no_token_id,
                yes_size,
                no_size,
                yes_cost,
                no_cost,
                yes_payout,
                no_payout,
                0.0,  # pnl recomputed below
                json.dumps({"question": question, **(extra or {})}),
            ),
        )
        if cur.rowcount == 0:
            # Another process already settled this market. Don't double-pay.
            await self.db.commit()
            self.settled_conditions.add(condition_id)
            return None

        # We won the race: now actually book the cash flows.
        pnl = 0.0
        if yes_size > 0 or no_size > 0:
            yes_proceeds = yes_size * yes_payout
            no_proceeds = no_size * no_payout
            pnl = (yes_proceeds + no_proceeds) - (yes_size * yes_cost + no_size * no_cost)
            self.cash += yes_proceeds + no_proceeds
            self.realized_pnl += pnl
            if yes_pos and yes_size > 0:
                yes_pos.size = 0.0
                yes_pos.avg_cost = 0.0
            if no_pos and no_size > 0:
                no_pos.size = 0.0
                no_pos.avg_cost = 0.0

        # Persist the actual realized pnl now that we've computed it.
        await self.db.execute(
            "UPDATE resolutions SET pnl = ? WHERE condition_id = ?",
            (pnl, condition_id),
        )
        await self.db.commit()
        self.settled_conditions.add(condition_id)

        log.info(
            "market_settled",
            condition_id=condition_id,
            question=question[:90],
            yes_won=yes_won,
            yes_size=round(yes_size, 2),
            no_size=round(no_size, 2),
            yes_avg_cost=round(yes_cost, 4),
            no_avg_cost=round(no_cost, 4),
            pnl=round(pnl, 2),
            cash_after=round(self.cash, 2),
        )

        return {
            "condition_id": condition_id,
            "yes_won": yes_won,
            "pnl": pnl,
            "yes_size": yes_size,
            "no_size": no_size,
        }

    def was_recently_stopped(self, token_id: str) -> bool:
        """True if the token took a stop-loss within ``stop_loss_blacklist_sec``."""
        ts = self.recently_stopped.get(token_id)
        if ts is None:
            return False
        if time.time() - ts > self.stop_loss_blacklist_sec:
            return False
        return True

    def buys_in_window(self, token_id: str) -> int:
        """How many BUYs on this token are within the rolling fill-cap
        window? Strategies should refuse new BUYs when this is
        ``>= max_buys_per_token_window``."""
        lst = self._buys_per_token.get(token_id)
        if not lst:
            return 0
        now = time.time()
        cutoff = now - self.buys_per_token_window_sec
        # Trim in place so memory stays bounded.
        kept = [t for t in lst if t >= cutoff]
        if len(kept) != len(lst):
            self._buys_per_token[token_id] = kept
        return len(kept)

    def is_token_buy_capped(self, token_id: str) -> bool:
        return self.buys_in_window(token_id) >= self.max_buys_per_token_window

    async def snapshot_nav(self) -> None:
        if self.db is None:
            return
        async with self._lock:
            ts = time.time()
            pv_mid = self.position_value(mark="mid")
            nav_mid = self.cash + pv_mid
            self.drawdown.update(nav_mid)
            await self.db.execute(
                "INSERT INTO nav_history(ts, cash, position_value, nav) VALUES (?,?,?,?)",
                (ts, self.cash, pv_mid, nav_mid),
            )
            await self.db.commit()
        # Refresh pessimistic gate roughly every snapshot. Cheap query.
        try:
            await self._refresh_pessimistic_gate()
        except Exception as e:
            log.warning("pessimistic_gate_refresh_failed", err=str(e))

    async def _refresh_pessimistic_gate(self) -> None:
        """Compute total estimated slippage burn over the most recent
        ``_pess_gate_window_fills`` fills; if it exceeds
        ``_pess_gate_block_pct`` of starting NAV, block new BUYs.

        Idea: every fill row in fills_shadow has slippage_estimate (the
        pessimistic-vs-VWAP difference per share, multiplied by size in
        the model). If, summed over the recent window, that's burning a
        material fraction of NAV, the strategy is paying too much for
        execution and we should pause new entries.
        """
        if self.db is None:
            return
        now = time.time()
        # Throttle to once per ~30s.
        if now - self._pess_gate_last_check < 30:
            return
        self._pess_gate_last_check = now
        async with self.db.execute(
            "SELECT COUNT(*) FROM fills_shadow"
        ) as cur:
            row = await cur.fetchone()
        n_total = int((row or [0])[0])
        if n_total < self._pess_gate_window_fills:
            self._pess_gate_blocked = False
            return
        async with self.db.execute(
            "SELECT COALESCE(SUM(slippage_estimate * size), 0.0) FROM ("
            "  SELECT slippage_estimate, size FROM fills_shadow "
            "  ORDER BY fill_id DESC LIMIT ?"
            ")",
            (self._pess_gate_window_fills,),
        ) as cur:
            row = await cur.fetchone()
        slippage_burn = float((row or [0.0])[0] or 0.0)
        burn_pct = slippage_burn / max(1.0, self.nav_start)
        was_blocked = self._pess_gate_blocked
        self._pess_gate_blocked = burn_pct >= self._pess_gate_block_pct
        if self._pess_gate_blocked != was_blocked:
            log.warning(
                "pessimistic_gate_changed",
                blocked=self._pess_gate_blocked,
                slippage_burn=round(slippage_burn, 2),
                burn_pct=round(burn_pct * 100, 3),
                window=self._pess_gate_window_fills,
                threshold_pct=round(self._pess_gate_block_pct * 100, 3),
            )

    def summary(self) -> dict:
        pv_bid = self.position_value(mark="bid")
        pv_mid = self.position_value(mark="mid")
        nav_bid = self.cash + pv_bid
        nav_mid = self.cash + pv_mid
        # Track drawdown vs the all-time high-water mark.
        self.drawdown.update(max(nav_mid, self.drawdown.hwm))
        dd = self.drawdown.drawdown(nav_mid)
        # Concentration: largest single position as % of NAV.
        max_pos_value = 0.0
        for p in self.positions.values():
            if p.size <= 0:
                continue
            v = abs(p.size * self._mark(p, prefer="mid"))
            if v > max_pos_value:
                max_pos_value = v
        max_pos_pct = (max_pos_value / nav_mid) if nav_mid > 0 else 0.0
        return {
            "cash": round(self.cash, 2),
            "position_value": round(pv_mid, 2),       # legacy: neutral mid
            "liquidation_value": round(pv_bid, 2),    # what we'd actually get
            "nav": round(nav_mid, 2),                 # legacy: mid-based
            "nav_liquidation": round(nav_bid, 2),     # honest: bid-based
            "pnl_total": round(nav_mid - self.nav_start, 2),
            "pnl_total_liquidation": round(nav_bid - self.nav_start, 2),
            "pnl_pct": round((nav_mid - self.nav_start) / self.nav_start * 100, 3),
            "pnl_pct_liquidation": round(
                (nav_bid - self.nav_start) / self.nav_start * 100, 3
            ),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl_liquidation": round(self.unrealized_pnl(mark="bid"), 2),
            "open_positions": sum(1 for p in self.positions.values() if p.size > 0),
            "fills": self.fills,
            "drawdown_pct": round(dd * 100, 3),
            "hwm": round(self.drawdown.hwm, 2),
            "max_pos_pct": round(max_pos_pct * 100, 2),
            "killed": is_killed(),
        }
