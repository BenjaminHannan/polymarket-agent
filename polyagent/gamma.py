"""Polymarket Gamma API client (read-only, no auth)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiohttp
import structlog

from polyagent.config import settings

log = structlog.get_logger()


@dataclass
class Market:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_iso: str | None
    liquidity: float
    volume_24h: float
    accepting_orders: bool
    category: str | None
    # NegRisk + event grouping (used by combinatorial_arb and lambdarank
    # query grouping). Both can be None for stand-alone markets.
    neg_risk: bool = False
    event_id: str | None = None
    event_slug: str | None = None
    n_outcomes_in_event: int | None = None


def _parse_json_field(raw: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def days_to_resolution(end_date_iso: str | None) -> float | None:
    if not end_date_iso:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        delta = dt.timestamp() - __import__("time").time()
        return max(0.0, delta / 86400.0)
    except Exception:
        return None


def _to_market(m: dict) -> Market | None:
    token_ids = _parse_json_field(m.get("clobTokenIds"))
    outcomes = _parse_json_field(m.get("outcomes"))

    if not token_ids or len(token_ids) != 2:
        return None
    if not outcomes or len(outcomes) != 2:
        return None

    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() == "yes"), 0)
    no_idx = 1 - yes_idx

    try:
        liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0)
    except (TypeError, ValueError):
        liquidity = 0.0
    try:
        volume_24h = float(m.get("volume24hr") or m.get("volume24Hr") or 0)
    except (TypeError, ValueError):
        volume_24h = 0.0

    # NegRisk / event grouping. Gamma exposes negRisk on the market and
    # an `events` array on the market; we pull the event id from the
    # first event entry. Both fields are best-effort and may be missing.
    neg_risk = bool(m.get("negRisk") or False)
    event_id: str | None = None
    event_slug: str | None = None
    events_arr = m.get("events") or []
    if isinstance(events_arr, list) and events_arr:
        first = events_arr[0]
        if isinstance(first, dict):
            event_id = str(first.get("id") or first.get("event_id") or "") or None
            event_slug = first.get("slug") or first.get("ticker")
    elif m.get("eventId") or m.get("event_id"):
        event_id = str(m.get("eventId") or m.get("event_id"))

    return Market(
        condition_id=m.get("conditionId") or m.get("condition_id") or "",
        question=m.get("question", ""),
        yes_token_id=str(token_ids[yes_idx]),
        no_token_id=str(token_ids[no_idx]),
        end_date_iso=m.get("endDate") or m.get("end_date_iso"),
        liquidity=liquidity,
        volume_24h=volume_24h,
        accepting_orders=bool(m.get("acceptingOrders", True)),
        category=m.get("category"),
        neg_risk=neg_risk,
        event_id=event_id,
        event_slug=event_slug,
    )


async def fetch_active_markets(limit: int = 100, min_liquidity: float = 0.0) -> list[Market]:
    """Pull active, accepting-orders markets from Gamma, sorted by liquidity desc."""
    url = f"{settings.gamma_url}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": str(min(limit * 4, 500)),
        "order": "volume24hr",
        "ascending": "false",
    }

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as r:
            r.raise_for_status()
            data = await r.json()

    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    markets: list[Market] = []
    for m in rows:
        parsed = _to_market(m)
        if parsed is None:
            continue
        if not parsed.accepting_orders:
            continue
        if parsed.liquidity < min_liquidity:
            continue
        if not parsed.condition_id or not parsed.yes_token_id:
            continue
        markets.append(parsed)
        if len(markets) >= limit:
            break

    log.info("gamma_markets_fetched", count=len(markets), min_liquidity=min_liquidity)
    return markets
