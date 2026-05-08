"""Local-edge pre-filter classifier.

Idea adapted from `takakhoo/Polymarket_Agent` `local_market` prompt.
Before spending heavyweight LLM forecasting compute on a market, ask
the local LLM a single yes/no question: does this market plausibly
have a local / insider / non-English news edge that an English-only
question-text model cannot reach?

If yes, the runtime
  - upweights the news_match and llm_forecaster experts in the
    log-pool for this market
  - allows the Telegram-ingest pipeline (when enabled) to route
    matched messages to this market with a higher confidence floor

If no, default behaviour (the regular log-pool weights apply).

Cached forever per question hash because the answer doesn't change
once the question text is fixed. Falls back to "unknown" (no-op) when
the LLM is offline.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field

import structlog

from polyagent.models.llm_forecaster import LLMForecaster

log = structlog.get_logger()


_PROMPT = """You are categorising a prediction-market question for a
trading agent. The agent's main forecaster reads only English news. We
want to know whether a question's resolution is likely to be moved by
LOCAL or NON-ENGLISH information that the main forecaster cannot reach.

Examples of LOCAL-EDGE markets (answer: yes):
  - "Will Israel and Hamas sign a ceasefire by July?" — IDF / Pikud HaOref
    (Hebrew/Arabic) post 60-120s before Reuters.
  - "Will Putin visit Beijing before May 31?" — Russian/Chinese state
    media + Telegram government channels move first.
  - "Will the Mexican Senate pass the energy reform bill?" — Spanish
    legislative channels.

Examples of NON-LOCAL markets (answer: no):
  - "Will BTC close above $100k on Dec 31?" — global financial markets,
    English news is the source of truth.
  - "Will the Patriots win the AFC East?" — English sports media.
  - "Will the Fed cut rates 25bps in March?" — English Fed minutes.

Output STRICT JSON in this exact schema and nothing else:
{
  "has_local_edge": true|false,
  "confidence": 0.0..1.0,
  "reason": "<single short sentence>"
}

Question: {question}

JSON:"""


@dataclass
class LocalEdgeResult:
    has_local_edge: bool
    confidence: float
    reason: str


@dataclass
class LocalEdgeClassifier:
    llm: LLMForecaster | None = None
    _cache: dict[str, LocalEdgeResult] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMForecaster()

    def _hash(self, question: str) -> str:
        return hashlib.sha256(question.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def classify(self, question: str) -> LocalEdgeResult:
        if not question:
            return LocalEdgeResult(False, 0.0, "")
        key = self._hash(question)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not self.llm or not self.llm.is_enabled():
            r = LocalEdgeResult(False, 0.0, "llm_disabled")
            with self._lock:
                self._cache[key] = r
            return r
        prompt = _PROMPT.format(question=question[:400])
        try:
            text = self.llm._generate(prompt, temperature=0.1)
        except Exception as e:
            log.warning("local_edge_llm_error", err=str(e))
            text = ""
        r = self._parse(text)
        with self._lock:
            self._cache[key] = r
        if r.has_local_edge:
            log.info(
                "local_edge_yes",
                question=question[:80],
                confidence=round(r.confidence, 3),
                reason=r.reason[:140],
            )
        return r

    def _parse(self, text: str) -> LocalEdgeResult:
        if not text:
            return LocalEdgeResult(False, 0.0, "")
        m = re.search(r"\{[\s\S]*?\}", text)
        if not m:
            return LocalEdgeResult(False, 0.0, "")
        try:
            import json
            d = json.loads(m.group(0))
        except Exception:
            return LocalEdgeResult(False, 0.0, "")
        try:
            has = bool(d.get("has_local_edge", False))
            conf = float(d.get("confidence", 0.0))
            reason = str(d.get("reason", ""))[:300]
        except (TypeError, ValueError):
            return LocalEdgeResult(False, 0.0, "")
        conf = max(0.0, min(1.0, conf))
        return LocalEdgeResult(has, conf, reason)
