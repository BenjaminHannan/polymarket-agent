"""NAV high-water mark + drawdown tracker.

Persisted to data/drawdown.json so the HWM survives restarts. Updated
every NAV snapshot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from polyagent.config import settings

log = structlog.get_logger()

_PATH = Path(settings.db_path).parent / "drawdown.json"


@dataclass
class DrawdownTracker:
    hwm: float = 0.0
    hwm_ts: float = 0.0

    @classmethod
    def load(cls) -> "DrawdownTracker":
        if not _PATH.exists():
            return cls()
        try:
            d = json.loads(_PATH.read_text())
            return cls(hwm=float(d.get("hwm", 0.0)), hwm_ts=float(d.get("hwm_ts", 0.0)))
        except Exception as e:
            log.warning("drawdown_load_error", err=str(e))
            return cls()

    def save(self) -> None:
        try:
            _PATH.write_text(json.dumps({"hwm": self.hwm, "hwm_ts": self.hwm_ts}))
        except Exception as e:
            log.warning("drawdown_save_error", err=str(e))

    def update(self, nav: float) -> None:
        if nav > self.hwm:
            self.hwm = nav
            self.hwm_ts = time.time()
            self.save()

    def drawdown(self, nav: float) -> float:
        """Returns drawdown as a positive fraction in [0, 1]. 0 means at peak."""
        if self.hwm <= 0:
            return 0.0
        return max(0.0, (self.hwm - nav) / self.hwm)
