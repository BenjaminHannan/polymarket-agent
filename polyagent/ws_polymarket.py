"""Polymarket public market WSS client (no auth).

Two pmwhy.md §D fixes baked in:

  1. **Polymarket RTDS literal "PING" keepalive** — the spec requires a
     literal "PING" string sent every ~5s, NOT a WebSocket-level frame
     ping. The server replies with the literal "PONG" string and uses
     this as the connection-liveness signal. Without it, Polymarket
     can drop the connection silently.

  2. **Stall-timeout watchdog** — connections sometimes go silent
     (TCP alive, ping/pong fine, but upstream stops emitting events).
     We track `last_message_ts` and force-reconnect if no message has
     arrived in `STALL_TIMEOUT` seconds. Without this, the bot can
     run for tens of minutes against stale books.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Callable, Iterable, Optional

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from polyagent.config import settings

log = structlog.get_logger()

# Server appears to accept large subscription lists; chunk to be safe.
_CHUNK = 200

# Polymarket RTDS keepalive: literal "PING"/"PONG" strings every ~5s
RTDS_PING_INTERVAL_SEC = 5.0

# Stall watchdog: if no message in this many seconds, force reconnect
STALL_TIMEOUT_SEC = 60.0


def _chunked(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# Optional callback the caller can pass to be notified when a chunk
# disconnects, so any cached book state for those asset_ids can be invalidated.
DisconnectCallback = Callable[[list[str]], None]


async def _enqueue_or_drop(queue: asyncio.Queue, evt) -> None:
    """Put on queue, dropping oldest on QueueFull (don't starve the socket)."""
    try:
        queue.put_nowait(evt)
    except asyncio.QueueFull:
        try:
            _ = queue.get_nowait()
            queue.task_done()
            queue.put_nowait(evt)
            log.warning("queue_full_dropped_oldest")
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


async def _send_rtds_pings(ws, stop_event: asyncio.Event) -> None:
    """Send the literal "PING" string every RTDS_PING_INTERVAL_SEC seconds.

    Polymarket's RTDS protocol requires this app-level keepalive in
    addition to (or instead of) WebSocket-level ping/pong frames.
    Without it the server may drop the connection.
    """
    try:
        while not stop_event.is_set():
            try:
                await ws.send("PING")
            except (ConnectionClosed, WebSocketException) as e:
                log.warning("ws_ping_send_error", err=str(e))
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=RTDS_PING_INTERVAL_SEC)
            except asyncio.TimeoutError:
                continue  # interval elapsed, send another ping
    except asyncio.CancelledError:
        pass


async def stream_market_chunk(
    asset_ids: list[str],
    queue: asyncio.Queue,
    on_disconnect: Optional[DisconnectCallback] = None,
) -> None:
    """Subscribe to a chunk of asset_ids and push parsed messages onto the queue.

    On disconnect, the optional `on_disconnect(asset_ids)` callback is invoked
    so the caller can clear stale book state. The server replays current book
    snapshots on resubscribe, so books will refill within seconds.

    Two reliability features layered on top of the base subscription:
      1. App-level RTDS PING/PONG every RTDS_PING_INTERVAL_SEC sec.
      2. Stall-timeout watchdog: per-recv timeout = STALL_TIMEOUT_SEC;
         if no message in that window, raise + reconnect.
    """
    backoff = 1.0
    while True:
        ping_stop = asyncio.Event()
        ping_task: Optional[asyncio.Task] = None
        try:
            # ping_interval=None disables the WS-frame ping (we use the
            # app-level RTDS PING/PONG string protocol instead).
            async with websockets.connect(
                settings.market_ws_url,
                ping_interval=None,
                ping_timeout=None,
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
                # Spin up the app-level keepalive sender
                ping_task = asyncio.create_task(_send_rtds_pings(ws, ping_stop))
                last_msg_ts = time.time()
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=STALL_TIMEOUT_SEC)
                    except asyncio.TimeoutError:
                        idle = time.time() - last_msg_ts
                        log.warning(
                            "ws_stall_timeout_reconnecting",
                            idle_sec=round(idle, 1),
                            n_assets=len(asset_ids),
                        )
                        # Break out of the recv loop; the outer try/except
                        # will catch and reconnect after backoff.
                        raise
                    last_msg_ts = time.time()
                    if not raw:
                        continue
                    if raw == "PONG":
                        # Server's RTDS PONG response — keepalive worked.
                        continue
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    # Server may send a list of events or a single event.
                    if isinstance(parsed, list):
                        for evt in parsed:
                            await _enqueue_or_drop(queue, evt)
                    elif isinstance(parsed, dict):
                        await _enqueue_or_drop(queue, parsed)
        except (ConnectionClosed, WebSocketException, OSError, asyncio.TimeoutError) as e:
            log.warning("ws_disconnect", err=str(e), backoff=backoff)
            if on_disconnect is not None:
                try:
                    on_disconnect(asset_ids)
                except Exception as cb_err:
                    log.warning("on_disconnect_cb_error", err=str(cb_err))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        finally:
            # Stop the ping task before next iteration / exit
            ping_stop.set()
            if ping_task is not None:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass


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
