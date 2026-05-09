"""Alchemy Polygon RPC client (read-only helpers).

Used today only as a connectivity probe (eth_blockNumber, eth_chainId) so
the key is exercised. When real trading is added, this module hosts the
on-chain reads (CTF position balances, redemption status) and tx submission
goes through py-clob-client / web3.
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

from polyagent.config import settings
from polyagent.news_store import NewsEvent

log = structlog.get_logger()


async def _rpc(session: aiohttp.ClientSession, method: str, params: list | None = None) -> dict | None:
    if not settings.alchemy_rpc_url:
        return None
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
    try:
        async with session.post(
            settings.alchemy_rpc_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                log.warning("alchemy_http", method=method, status=r.status)
                return None
            return await r.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        log.warning("alchemy_error", method=method, err=str(e))
        return None


async def get_block_number() -> int | None:
    async with aiohttp.ClientSession() as s:
        resp = await _rpc(s, "eth_blockNumber")
    if not resp or "result" not in resp:
        return None
    try:
        return int(resp["result"], 16)
    except (TypeError, ValueError):
        return None


async def run(queue: asyncio.Queue, poll_sec: int = 600) -> None:
    """Heartbeat probe: emit a single 'rpc_alive' event each block-tick interval.

    Doesn't trade off this. Just keeps the key warm and confirms the RPC works.
    """
    if not settings.alchemy_rpc_url:
        log.warning("alchemy_disabled_no_url")
        await asyncio.Event().wait()
        return
    log.info("alchemy_start", poll_sec=poll_sec)
    last_block = -1
    async with aiohttp.ClientSession() as session:
        while True:
            chain = await _rpc(session, "eth_chainId")
            block = await _rpc(session, "eth_blockNumber")
            if chain and block:
                try:
                    chain_id = int(chain["result"], 16)
                    block_num = int(block["result"], 16)
                except (KeyError, TypeError, ValueError):
                    chain_id, block_num = None, None
                if block_num is not None and block_num != last_block:
                    last_block = block_num
                    log.debug("alchemy_block", chain_id=chain_id, block=block_num)
            await asyncio.sleep(poll_sec)
