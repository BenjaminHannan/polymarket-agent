"""Smart-money registry via Polymarket's public Goldsky subgraph.

Builds a rolling registry of high-PnL maker wallets (top-K by realized
profit over the last 30 days). Used by adverse-selection: when a known
smart-money wallet is on the OTHER side of one of our proposed trades,
we either skip the trade or post deeper inside the spread.

Solidus Labs (April 2026): 0.55% of profitable maker wallets capture
50% of maker gains. Detecting that small set of wallets and avoiding
trades against them is the doc-cited adverse-selection edge.

Endpoint: Goldsky's public Polymarket subgraph (no auth required for
free tier). Falls back gracefully if the endpoint is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import structlog

from polyagent.config import settings

log = structlog.get_logger()

# Public Polymarket subgraph (Goldsky-hosted; URL has changed in the past, so
# this is an env-overridable best-guess.)
SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/project_clrhmyxsvvuao01un9dxh9ct1/"
    "subgraphs/orderbook-subgraph/0.0.7/gn"
)

_PERSIST_PATH = Path(settings.db_path).parent / "smart_money.json"


@dataclass
class SmartMoneyRegistry:
    refresh_sec: float = 6 * 3600  # 6h refresh
    top_k: int = 200
    min_profit_usd: float = 1000.0
    smart_wallets: set[str] = field(default_factory=set)
    last_refresh_ts: float = 0.0
    enabled: bool = True
    _failed_attempts: int = 0

    @classmethod
    def load(cls) -> "SmartMoneyRegistry":
        r = cls()
        if _PERSIST_PATH.exists():
            try:
                d = json.loads(_PERSIST_PATH.read_text())
                r.smart_wallets = set(d.get("wallets") or [])
                r.last_refresh_ts = float(d.get("ts", 0.0))
            except Exception as e:
                log.warning("smart_money_load_error", err=str(e))
        return r

    def save(self) -> None:
        try:
            _PERSIST_PATH.write_text(
                json.dumps(
                    {
                        "wallets": sorted(self.smart_wallets),
                        "ts": self.last_refresh_ts,
                    }
                )
            )
        except Exception as e:
            log.warning("smart_money_save_error", err=str(e))

    def is_smart(self, wallet: str) -> bool:
        return wallet.lower() in self.smart_wallets

    async def refresh(self) -> dict:
        if not self.enabled:
            return {"skipped": True}
        # Goldsky GraphQL query to find top-N profitable wallets (this is a
        # best-effort schema; the public subgraph schema can drift, in which
        # case this falls back to no-op).
        query = """
        query TopWallets($limit: Int!) {
          users(first: $limit, orderBy: profit, orderDirection: desc) {
            id
            profit
          }
        }
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SUBGRAPH_URL,
                    json={"query": query, "variables": {"limit": self.top_k}},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as r:
                    if r.status != 200:
                        self._failed_attempts += 1
                        log.warning("smart_money_http", status=r.status, attempts=self._failed_attempts)
                        return {"ok": False, "status": r.status}
                    data = await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self._failed_attempts += 1
            log.warning("smart_money_error", err=str(e), attempts=self._failed_attempts)
            return {"ok": False, "err": str(e)}

        users = (data.get("data") or {}).get("users") or []
        new_wallets = set()
        for u in users:
            try:
                profit = float(u.get("profit") or 0)
            except (TypeError, ValueError):
                continue
            if profit >= self.min_profit_usd:
                new_wallets.add(str(u.get("id", "")).lower())

        if not new_wallets:
            return {"ok": False, "reason": "no_users_returned"}

        self.smart_wallets = new_wallets
        self.last_refresh_ts = time.time()
        self._failed_attempts = 0
        self.save()
        log.info("smart_money_refresh", n_wallets=len(self.smart_wallets), top_k=self.top_k)
        return {"ok": True, "n_wallets": len(self.smart_wallets)}

    async def run(self) -> None:
        log.info("smart_money_start", refresh_sec=self.refresh_sec, top_k=self.top_k)
        # Initial refresh; if it fails 3 times in a row we disable to stop log spam
        while True:
            try:
                await self.refresh()
                if self._failed_attempts >= 3:
                    log.warning("smart_money_disabled_after_failures")
                    self.enabled = False
                    return
            except Exception as e:
                log.warning("smart_money_run_error", err=str(e))
            await asyncio.sleep(self.refresh_sec)
