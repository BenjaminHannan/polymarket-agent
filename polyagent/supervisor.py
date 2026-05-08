"""Task supervisor — keeps long-running coroutines alive across crashes.

Every task wrapped via `supervised(name, factory)` runs inside an outer loop
that catches every exception, logs it loudly, and restarts the coroutine
with exponential backoff (1s -> 60s cap). On clean voluntary exit (e.g. an
ingest task that decides there's nothing to do), the supervisor just lets
it return — no restart.

Without this wrapper, a task that raises produces a `Task exception was
never retrieved` warning at GC time and silently disappears, which is what
killed status_loop / combined_signal during the lock storm.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

import structlog

log = structlog.get_logger()

CoroFactory = Callable[[], Awaitable[None]]


async def supervised(name: str, factory: CoroFactory, max_backoff: float = 60.0) -> None:
    """Run `factory()` in a loop. On exception: log + back off + retry.

    `factory` should be a zero-arg function that returns a fresh coroutine each
    call. (Coroutines can't be re-awaited, so we need a factory.) Common idiom:
        supervised("status_loop", lambda: status_loop(broker, book_store))
    """
    backoff = 1.0
    while True:
        started = time.time()
        try:
            await factory()
            log.info("supervised_task_returned", task=name)
            return  # clean exit, don't restart
        except asyncio.CancelledError:
            log.info("supervised_task_cancelled", task=name)
            raise
        except Exception as e:
            ran_for = time.time() - started
            # If the task ran for a while before crashing, reset backoff —
            # the failure was probably transient, not a startup loop.
            if ran_for > 60:
                backoff = 1.0
            log.error(
                "supervised_task_crashed",
                task=name,
                err=str(e),
                err_type=type(e).__name__,
                ran_for_sec=round(ran_for, 1),
                next_retry_sec=backoff,
                exc_info=True,
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, max_backoff)
