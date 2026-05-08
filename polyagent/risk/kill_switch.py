"""File-based emergency kill switch.

Touch `data/.STOP` (no contents needed) to halt all new orders. The bot
keeps running so resolutions still settle and the dashboard stays live, but
every strategy refuses to open new positions until the file is removed.

This is intentionally low-tech (no API, no SIGUSR1, no env var) so anyone —
including you under stress — can flip it from a terminal in 1 second:
    touch data/.STOP        # stop new orders
    rm data/.STOP           # resume
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from polyagent.config import settings

_KILL_PATH = Path(settings.db_path).parent / ".STOP"
_LAST_CHECK = 0.0
_LAST_VALUE = False


def is_killed() -> bool:
    """Returns True if the kill switch file exists. Cached for 1s to avoid
    hammering the filesystem on hot paths."""
    global _LAST_CHECK, _LAST_VALUE
    now = time.time()
    if now - _LAST_CHECK < 1.0:
        return _LAST_VALUE
    _LAST_CHECK = now
    _LAST_VALUE = _KILL_PATH.exists()
    return _LAST_VALUE
