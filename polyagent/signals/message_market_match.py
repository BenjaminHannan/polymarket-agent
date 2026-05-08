"""Per-message → candidate-markets LLM matcher.

Adapted from `takakhoo/Polymarket_Agent` `telegram_live_message_matcher`.
For each incoming news event (Telegram, RSS, Bluesky, etc.) we:

  1. Pull the top-N semantically-similar markets from the existing
     `SemanticMarketIndex` (BGE-large embeddings).
  2. Ask a local LLM to produce a structured
     `(market_id, confidence ∈ [0,1], reason_short, direction)` per
     candidate market the message materially supports/contradicts.
  3. Persist each high-confidence match as a `signals` row with
     ``strategy="message_market_match"``. Downstream, the news_match
     aggregator and the live ECE / PSI dashboards read these rows.

Why this is shaped this way and not just a similarity scalar (which
we already have): the structured output gives us a per-trade audit
trail (the ``reason_short`` shows up in the dashboard so we can see
*why* we entered) and the ``direction`` flag distinguishes a message
that pushes towards YES from one that pushes towards NO — semantic
similarity alone can't tell you which side to take.

This module is an LLM-only signal source; it never sends orders.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

import structlog

from polyagent.models.llm_forecaster import LLMForecaster
from polyagent.models.news_embed_matcher import SemanticMarketIndex
from polyagent.news_store import NewsEvent, NewsStore

log = structlog.get_logger()


_MATCHER_PROMPT = """You are mapping a news event to candidate prediction
markets. For each candidate, decide whether the event is meaningfully
relevant. If yes, output the market id, a confidence in [0,1], a single
short reason, and whether the event pushes the market towards YES or NO.

Output STRICT JSON in this exact schema and nothing else:
{
  "matches": [
    {"market_id": "<string>",
     "confidence": 0.0..1.0,
     "direction": "yes"|"no"|"unclear",
     "reason_short": "<single sentence>"}
  ]
}

Rules:
- Only include markets with confidence >= 0.5. Drop the rest.
- "reason_short" must be one sentence, factual, no hype.
- If no candidates match, return {"matches": []}.

News event:
  source: {source}
  text: {text}

Candidate markets:
{candidates}

JSON:"""


@dataclass
class MarketMatch:
    market_id: str
    confidence: float
    direction: str
    reason_short: str


@dataclass
class MessageMarketMatcher:
    """Stateless signal producer: in = (NewsEvent, candidates), out =
    persistent ``signals`` rows + log lines."""
    news_store: NewsStore
    semantic_index: SemanticMarketIndex
    llm: LLMForecaster | None = None
    top_k: int = 5
    min_sim: float = 0.42
    min_confidence: float = 0.55

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMForecaster()

    async def on_event(self, evt: NewsEvent) -> int:
        """Returns count of stored matches (0 if no LLM / no candidates)."""
        if not evt.title and not evt.body:
            return 0
        if self.llm is None or not self.llm.is_enabled():
            return 0
        # Top-K semantic candidate markets. SemanticMarketIndex.search
        # returns (condition_id, score, question) triples.
        text = (evt.title or "") + "\n" + (evt.body or "")
        try:
            cand = self.semantic_index.search(text, top_k=self.top_k)
        except Exception as e:
            log.warning("mmm_search_error", err=str(e))
            return 0
        cand = [(cid, sim, q) for (cid, sim, q) in cand if sim >= self.min_sim]
        if not cand:
            return 0
        cand_block = "\n".join(
            f"  - id={cid} q=\"{q[:140]}\""
            for (cid, _sim, q) in cand
        )
        prompt = _MATCHER_PROMPT.format(
            source=evt.source,
            text=text[:600],
            candidates=cand_block,
        )
        try:
            text_out = await asyncio.to_thread(
                self.llm._generate, prompt, 0.2
            )
        except Exception as e:
            log.warning("mmm_llm_error", err=str(e))
            return 0
        matches = self._parse(text_out)
        if not matches:
            return 0
        n_stored = 0
        for mm in matches:
            if mm.confidence < self.min_confidence:
                continue
            try:
                await self.news_store.insert_signal(
                    strategy="message_market_match",
                    condition_id=mm.market_id,
                    direction=mm.direction or "unclear",
                    score=float(mm.confidence),
                    news_hash=evt.hash(),
                    detail={
                        "source": evt.source,
                        "title": (evt.title or "")[:200],
                        "url": evt.url[:300] if evt.url else "",
                        "confidence": round(mm.confidence, 3),
                        "direction": mm.direction,
                        "reason_short": mm.reason_short[:280],
                        "ts": time.time(),
                    },
                )
                n_stored += 1
                log.info(
                    "message_market_match",
                    source=evt.source,
                    market_id=mm.market_id,
                    confidence=round(mm.confidence, 3),
                    direction=mm.direction,
                    reason=mm.reason_short[:140],
                )
            except Exception as e:
                log.warning("mmm_persist_error", err=str(e))
        return n_stored

    def _parse(self, text: str) -> list[MarketMatch]:
        if not text:
            return []
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        matches = d.get("matches") or []
        if not isinstance(matches, list):
            return []
        out: list[MarketMatch] = []
        for entry in matches:
            try:
                mid = str(entry.get("market_id", ""))
                conf = float(entry.get("confidence", 0.0))
                direction = str(entry.get("direction", "unclear")).lower().strip()
                if direction not in ("yes", "no", "unclear"):
                    direction = "unclear"
                reason = str(entry.get("reason_short", ""))[:300]
            except (TypeError, ValueError):
                continue
            if not mid or conf <= 0:
                continue
            out.append(MarketMatch(mid, max(0.0, min(1.0, conf)), direction, reason))
        return out
