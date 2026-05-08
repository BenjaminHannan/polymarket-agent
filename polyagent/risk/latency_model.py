"""Empirical latency tracker.

Measures how stale our books are at decision time. Per the 2025 microstructure
paper, sub-50 ms median ingest delay is achievable but a multi-second tail
exists. We track the distribution of (now − last_book_update_ts) sampled at
every signal-decision time, and expose a `p99_age_sec()` estimator that
strategies use to refuse trades against books older than the p99 age.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class LatencyTracker:
    """Multi-source latency histogram. The doc emphasized per-source
    distributions matter (RSS poll cadence vs Bluesky firehose vs WSS
    book updates have very different latency profiles)."""
    window: int = 5000
    samples: deque = field(default_factory=lambda: deque(maxlen=5000))
    by_source: dict = field(default_factory=dict)  # source -> deque

    def __post_init__(self):
        if self.samples.maxlen != self.window:
            self.samples = deque(maxlen=self.window)

    def record(self, age_sec: float, source: str = "default") -> None:
        if age_sec < 0:
            return
        self.samples.append(age_sec)
        d = self.by_source.get(source)
        if d is None:
            d = deque(maxlen=self.window)
            self.by_source[source] = d
        d.append(age_sec)

    def _pct(self, samples, p: float) -> float | None:
        n = len(samples)
        if n < 50:
            return None
        s = sorted(samples)
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx]

    def percentile(self, p: float, source: str | None = None) -> float | None:
        if source is None:
            return self._pct(self.samples, p)
        d = self.by_source.get(source)
        if d is None:
            return None
        return self._pct(d, p)

    def p50(self, source: str | None = None) -> float | None:
        return self.percentile(0.50, source)

    def p99(self, source: str | None = None) -> float | None:
        return self.percentile(0.99, source)

    def summary(self) -> dict:
        out = {
            "n": len(self.samples),
            "p50": self.p50(),
            "p99": self.p99(),
            "by_source": {},
        }
        for src, d in self.by_source.items():
            out["by_source"][src] = {
                "n": len(d),
                "p50": self._pct(d, 0.50),
                "p99": self._pct(d, 0.99),
            }
        return out
