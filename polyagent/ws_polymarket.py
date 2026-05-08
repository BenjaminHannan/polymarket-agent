"""Polymarket public market WSS client (no auth)."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable, Iterable, Optional

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from polyagent.config import settings

log = structlog.get_logger()

# Server appears to accept large subscription lists; chunk to be safe.
_CHUNK = 200


def _chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# Optional callback the caller can pass to be notified when a chunk
# disconnects, so any cached book state for those asset_ids can be invalidated.
DisconnectCallback = Callable[[list[str]], None]


async def stream_market_chunk(
    asset_ids: list[str],
    queue: asyncio.Queue,
    on_disconnect: Optional[DisconnectCallback] = None,
) -> None:
    """Subscribe to a chunk of asset_ids and push parsed messages onto the queue.

    On disconnect, the optional `on_disconnect(asset_ids)` callback is invoked
    so the caller can clear stale book state. The server replays current book
    snapshots on resubscribe, so books will refill within seconds.
    """
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                settings.market_ws_url,
                ping_interval=10,
                ping_timeout=20,
                max_size=2**22,
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "assets_ids": asset_ids,
                            "type": "market",
                        }
                    )
                )
                log.info("ws_subscribed", n_assets=len(asset_ids))
                backoff = 1.0
                async for raw in ws:
                    if not raw or raw == "PONG":
                        continue
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    # Server may send a list of events or a single event.
                    if isinstance(parsed, list):
                        for evt in parsed:
                            try:
                                queue.put_nowait(evt)
                            except asyncio.QueueFull:
                                # Drop the oldest event to make room — better
                                # than blocking the WSS reader (which would
                                # eventually starve the socket).
                                try:
                                    _ = queue.get_nowait()
                                    queue.task_done()
                                    queue.put_nowait(evt)
                                    log.warning("queue_full_dropped_oldest")
                                except (asyncio.QueueEmpty, asyncio.QueueFull):
                                    pass
                    elif isinstance(parsed, dict):
                        try:
                            queue.put_nowait(parsed)
                        except asyncio.QueueFull:
                            try:
                                _ = queue.get_nowait()
                                queue.task_done()
                                queue.put_nowait(parsed)
                                log.warning("queue_full_dropped_oldest")
                            except (asyncio.QueueEmpty, asyncio.QueueFull):
                                pass
        except (ConnectionClosed, WebSocketException, OSError, asyncio.TimeoutError) as e:
            log.warning("ws_disconnect", err=str(e), backoff=backoff)
            if on_disconnect is not None:
                try:
                    on_disconnect(asset_ids)
                except Exception as cb_err:
                    log.warning("on_disconnect_cb_error", err=str(cb_err))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def stream_markets(
    asset_ids: list[str],
    queue: asyncio.Queue,
    on_disconnect: Optional[DisconnectCallback] = None,
) -> None:
    """Spawn one WSS task per chunk of asset_ids."""
    if not asset_ids:
        log.warning("stream_markets_empty")
        return
    tasks = [
        asyncio.create_task(stream_market_chunk(chunk, queue, on_disconnect))
        for chunk in _chunked(asset_ids, _CHUNK)
    ]
    await asyncio.gather(*tasks)


async def drain(queue: asyncio.Queue) -> AsyncIterator[dict]:
    """Async generator helper that pulls events off the queue forever."""
    while True:
        evt = await queue.get()
        yield evt
