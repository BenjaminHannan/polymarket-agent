"""Resolution watcher.

Periodically polls Gamma for the markets in which we hold paper positions,
detects `closed=true` with a clean win/lose `outcomePrices` array, and tells
the paper broker to settle. We deliberately don't trust 0.5/0.5 splits or
missing outcomePrices — those mean the market is in dispute or canceled, and
we wait.

Note: this is paper-only. Real-money mode would hit the CTF Adapter's
`redeemPositions` on Polygon via Alchemy, not this poller.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.gamma import Market, _parse_json_field, _to_market
from polyagent.models.lgbm import Predictor
from polyagent.models.outcomes import materialize_outcome_async
from polyagent.paper_broker import PaperBroker

log = structlog.get_logger()

POLL_SEC = 120  # Gamma /markets allows 300/10s. Polling held markets every 2 min is plenty.


@dataclass
class HeldMarketTracker:
    """Maps token_id -> Market across both legs, persisting beyond the initial
    market load so positions can still be settled if a market drops out of the
    top-N volume cohort while we hold it."""

    by_token: dict[str, Market] = field(default_factory=dict)
    by_condition: dict[str, Market] = field(default_factory=dict)

    def add(self, market: Market) -> None:
        self.by_token[market.yes_token_id] = market
        self.by_token[market.no_token_id] = market
        self.by_condition[market.condition_id] = market

    def add_many(self, markets: list[Market]) -> None:
        for m in markets:
            self.add(m)

    def held_condition_ids(self, broker: PaperBroker) -> set[str]:
        out: set[str] = set()
        for tid, p in broker.positions.items():
            if p.size <= 0:
                continue
            m = self.by_token.get(tid)
            if m is not None:
                out.add(m.condition_id)
        return out


def _outcome_prices(market_json: dict) -> tuple[float, float] | None:
    raw = _parse_json_field(market_json.get("outcomePrices"))
    if not raw or len(raw) != 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


async def _fetch_market(session: aiohttp.ClientSession, condition_id: str) -> dict | None:
    """Fetch a single market by condition_id. Falls back to the list endpoint
    filtered by condition_ids."""
    url = f"{settings.gamma_url}/markets"
    params = {"condition_ids": condition_id, "closed": "true", "limit": "1"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                return None
            data = await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    for m in rows:
        if (m.get("conditionId") or m.get("condition_id")) == condition_id:
            return m
    # Some markets may show up only with closed=false in the search; retry without filter.
    params2 = {"condition_ids": condition_id, "limit": "1"}
    try:
        async with session.get(url, params=params2, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                return None
            data = await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    for m in rows:
        if (m.get("conditionId") or m.get("condition_id")) == condition_id:
            return m
    return None


async def _check_one(
    session: aiohttp.ClientSession,
    market: Market,
    broker: PaperBroker,
    predictor: Predictor | None = None,
    bocpd_gate=None,  # polyagent.risk.bocpd_gate.BOCPDGate | None
) -> bool:
    """Return True if we settled this market (or there was nothing to do)."""
    j = await _fetch_market(session, market.condition_id)
    if j is None:
        return False
    closed = bool(j.get("closed"))
    accepting = bool(j.get("acceptingOrders", True))
    if not closed:
        return False
    prices = _outcome_prices(j)
    if prices is None:
        log.warning(
            "resolution_skipped_no_outcome_prices",
            condition_id=market.condition_id,
            question=market.question[:80],
        )
        return False
    p_yes_idx = next(
        (i for i, o in enumerate(_parse_json_field(j.get("outcomes")) or []) if str(o).strip().lower() == "yes"),
        0,
    )
    p_no_idx = 1 - p_yes_idx
    yes_price = prices[p_yes_idx]
    no_price = prices[p_no_idx]
    # Require a clean 1/0 resolution; anything else means dispute or cancel.
    if not ((yes_price >= 0.99 and no_price <= 0.01) or (no_price >= 0.99 and yes_price <= 0.01)):
        log.warning(
            "resolution_skipped_inconclusive",
            condition_id=market.condition_id,
            yes_price=yes_price,
            no_price=no_price,
            accepting=accepting,
        )
        return False
    yes_won = yes_price >= 0.99
    res = await broker.settle_market(
        condition_id=market.condition_id,
        yes_token_id=market.yes_token_id,
        no_token_id=market.no_token_id,
        yes_won=yes_won,
        question=market.question,
        extra={"yes_price": yes_price, "no_price": no_price},
    )
    # BOCPD gate: feed the win/loss outcome of every settled trade into
    # the changepoint detector. Only count as an observation if we
    # actually had a position (i.e. realized a non-zero pnl); pure
    # observational settlements (we never bought) are not edge data.
    if bocpd_gate is not None and res is not None:
        had_position = (res.get("yes_size", 0) or 0) > 0 or (res.get("no_size", 0) or 0) > 0
        if had_position:
            try:
                bocpd_gate.update(win=float(res.get("pnl", 0.0)) > 0)
            except Exception as e:
                log.warning("bocpd_update_error", err=str(e))
    # Materialize a labeled training row from the signals that fired on this market.
    try:
        await materialize_outcome_async(
            condition_id=market.condition_id,
            question=market.question,
            yes_won=yes_won,
            liquidity=market.liquidity,
            volume=market.volume_24h,
            predictor=predictor,
            half_life_sec=settings.news_match_half_life_sec or None,
        )
    except Exception as e:
        log.warning("materialize_outcome_error", condition_id=market.condition_id, err=str(e))

    # Failure tracking: classify the just-materialised row and write to
    # model_failures so the dashboard + retraining can audit where the
    # model missed. Wrapped in try/except — failure-tracking errors must
    # never break the settlement path.
    try:
        import sqlite3 as _sql
        from polyagent.models.failure_tracker import record_failures
        conn = _sql.connect(settings.db_path)
        try:
            row = conn.execute(
                """SELECT p_stat_lgbm,
                          COALESCE(p_market_24h, p_market_6h, p_market_1h, p_market_pre) AS p_market,
                          category
                   FROM signal_outcomes WHERE condition_id = ?""",
                (market.condition_id,),
            ).fetchone()
            if row is not None:
                p_stat, p_mkt, cat = row
                notional_traded = float(
                    (res.get("yes_size", 0) or 0) * (res.get("yes_avg_cost", 0) or 0)
                    + (res.get("no_size", 0) or 0) * (res.get("no_avg_cost", 0) or 0)
                ) if isinstance(res, dict) else 0.0
                realized_pnl = float(res.get("pnl", 0.0) or 0.0) if isinstance(res, dict) else 0.0
                record_failures(
                    conn,
                    condition_id=market.condition_id,
                    resolved_ts=time.time(),
                    yes_won=int(yes_won),
                    p_model=float(p_stat) if p_stat is not None else None,
                    p_market=float(p_mkt) if p_mkt is not None else None,
                    category=cat,
                    question=market.question,
                    notional_traded=notional_traded,
                    realized_pnl=realized_pnl,
                )
        finally:
            conn.close()
    except Exception as e:
        log.warning("failure_record_error", condition_id=market.condition_id, err=str(e))
    return res is not None


async def run(
    broker: PaperBroker,
    tracker: HeldMarketTracker,
    poll_sec: int = POLL_SEC,
    predictor: Predictor | None = None,
    bocpd_gate=None,
) -> None:
    log.info(
        "resolution_watcher_start",
        poll_sec=poll_sec,
        with_predictor=predictor is not None,
        with_bocpd=bocpd_gate is not None,
    )
    async with aiohttp.ClientSession() as session:
        while True:
            held = tracker.held_condition_ids(broker)
            if held:
                log.debug("resolution_check", held=len(held))
                for cid in held:
                    m = tracker.by_condition.get(cid)
                    if m is None:
                        continue
                    try:
                        await _check_one(
                            session, m, broker,
                            predictor=predictor, bocpd_gate=bocpd_gate,
                        )
                    except Exception as e:
                        log.warning("resolution_check_error", condition_id=cid, err=str(e))
            await asyncio.sleep(poll_sec)
