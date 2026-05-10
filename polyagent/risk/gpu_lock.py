"""Global GPU serialization lock (pmwhy.md §D).

The doc warned: "On a single RTX 5070 Ti with 17GB, running gpt-oss-20b
and Phi-4-mini and DeBERTa-NLI and bge-large concurrently will cause
memory pressure that triggers GPU thrashing under load. Practical fixes:
serialize LLM calls behind a queue, never run two transformer forwards
concurrently."

This module exposes a single process-wide reentrant lock that all
GPU-bound forward passes acquire before running. It's a `threading.RLock`
so it works correctly across both async and sync call paths (sync calls
made from `asyncio.to_thread` workers, async calls from event-loop
tasks). RLock is reentrant so a function holding the lock can re-enter
itself or call other lock-holding functions without deadlocking.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator

import structlog

log = structlog.get_logger()


# Process-wide reentrant lock. Held while a transformer forward pass
# is in flight. Lock acquisition is sub-millisecond when uncontended;
# under contention the second caller blocks until the first releases.
_gpu_lock = threading.RLock()


@contextlib.contextmanager
def gpu_section(label: str = "fwd") -> Iterator[None]:
    """Context manager: acquire the global GPU lock, log slow waits.

    Usage:
        with gpu_section("nli_verify"):
            outputs = model(**inputs)
    """
    t0 = time.time()
    acquired = _gpu_lock.acquire(timeout=120.0)
    wait_sec = time.time() - t0
    if not acquired:
        log.warning("gpu_lock_acquire_timeout", label=label, wait_sec=round(wait_sec, 2))
        # Yield without the lock as a degraded fallback rather than block
        # the event loop indefinitely.
        try:
            yield
        finally:
            return
    if wait_sec > 1.0:
        log.info("gpu_lock_waited", label=label, wait_sec=round(wait_sec, 2))
    try:
        yield
    finally:
        _gpu_lock.release()


def is_locked() -> bool:
    """Diagnostic: True if any thread currently holds the lock."""
    acquired = _gpu_lock.acquire(blocking=False)
    if acquired:
        _gpu_lock.release()
        return False
    return True
