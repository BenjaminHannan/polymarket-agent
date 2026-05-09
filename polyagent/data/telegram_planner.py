"""Multilingual Telegram handle planner.

Inspired by `takakhoo/Polymarket_Agent` `telegram_query_planner` prompt.
Given a Polymarket question, ask a local LLM to enumerate the structured
list of likely-relevant Telegram channels organized by tier:

  1. Actors and countries mentioned in the question.
  2. Bridge languages for each (e.g. Israel: Hebrew + Arabic + English;
     Iran: Farsi + Arabic + English; Russia/Ukraine: Russian + Ukrainian
     + English).
  3. Tier-1 official military / government / spokesperson handles
     (highest signal-to-noise: e.g. @idfofficial, @PikudHaOref_all).
  4. Tier-2 named-journalist handles (slower than tier-1 but more
     analysis).
  5. Tier-3 community / alert channels (faster but noisier).
  6. Bare keyword combinations (no full sentences) for further search.

This is the data layer that closes our biggest gap — every existing
ingest source is English; geopolitical edge largely lives in
non-English official handles posting 60-120s before Reuters picks it
up.

The planner is cached by question hash since plans don't change with
news velocity. Falls back to an empty (no-op) plan when the local LLM
isn't available, so callers can always treat the result as advisory.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import structlog

from polyagent.models.llm_forecaster import LLMForecaster

log = structlog.get_logger()


_PLANNER_PROMPT = """You are a research analyst building a multilingual
Telegram-channel watch-list for a prediction market. Given the question
below, enumerate likely-relevant channels organized by tier, prioritising
official sources first, then named journalists, then community/alert
channels. Always include the bridge languages of the entities involved.

Output STRICT JSON in this exact schema and nothing else:
{
  "actors": [string],
  "countries": [string],
  "languages": [string],
  "official_handles": [string],
  "journalist_handles": [string],
  "community_handles": [string],
  "keyword_combos": [string]
}

Rules:
- Handles must be lowercase Telegram usernames WITHOUT the leading @.
- "languages" should include English plus all bridge languages of the
  countries (Hebrew, Arabic, Farsi, Russian, Ukrainian, Mandarin, etc).
- "keyword_combos" are 1-3 word combinations a search engine could use
  to find more channels — NEVER full sentences.
- If the question is fundamentally non-geopolitical (sports, weather,
  crypto price, US-domestic-only) return empty arrays.

Question: {question}

JSON:"""


@dataclass
class TelegramTargets:
    actors: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    official_handles: list[str] = field(default_factory=list)
    journalist_handles: list[str] = field(default_factory=list)
    community_handles: list[str] = field(default_factory=list)
    keyword_combos: list[str] = field(default_factory=list)

    def all_handles(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for h in self.official_handles + self.journalist_handles + self.community_handles:
            h = h.strip().lstrip("@").lower()
            if h and h not in seen:
                seen.add(h)
                out.append(h)
        return out

    def is_empty(self) -> bool:
        return not (self.official_handles or self.journalist_handles or self.community_handles)


@dataclass
class TelegramHandlePlanner:
    """Caches plans by question hash to avoid re-running LLM on every poll."""
    llm: LLMForecaster | None = None
    _cache: dict[str, TelegramTargets] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.llm is None:
            self.llm = LLMForecaster()

    def _hash(self, question: str) -> str:
        return hashlib.sha256(question.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def plan(self, question: str) -> TelegramTargets:
        if not question:
            return TelegramTargets()
        key = self._hash(question)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not self.llm or not self.llm.is_enabled():
            t = TelegramTargets()
            with self._lock:
                self._cache[key] = t
            return t
        prompt = _PLANNER_PROMPT.format(question=question[:400])
        try:
            text = self.llm._generate(prompt, temperature=0.2)
        except Exception as e:
            log.warning("telegram_planner_llm_error", err=str(e))
            text = ""
        targets = self._parse(text)
        with self._lock:
            self._cache[key] = targets
        log.info(
            "telegram_plan_built",
            question=question[:80],
            n_official=len(targets.official_handles),
            n_journalists=len(targets.journalist_handles),
            n_community=len(targets.community_handles),
            n_keywords=len(targets.keyword_combos),
            languages=targets.languages,
        )
        return targets

    def _parse(self, text: str) -> TelegramTargets:
        if not text:
            return TelegramTargets()
        # Extract first JSON object-like substring (LLM may add prose).
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return TelegramTargets()
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return TelegramTargets()
        def _strs(key: str) -> list[str]:
            v = d.get(key) or []
            if not isinstance(v, list):
                return []
            return [str(x).strip() for x in v if str(x).strip()]
        return TelegramTargets(
            actors=_strs("actors"),
            countries=_strs("countries"),
            languages=_strs("languages"),
            official_handles=[h.strip().lstrip("@").lower() for h in _strs("official_handles")],
            journalist_handles=[h.strip().lstrip("@").lower() for h in _strs("journalist_handles")],
            community_handles=[h.strip().lstrip("@").lower() for h in _strs("community_handles")],
            keyword_combos=_strs("keyword_combos"),
        )


# Curated seeds — when the LLM is offline we still want SOMETHING for
# common geopolitical regions. Conservative: only well-known official
# handles. Add to as needed. Keys are lowercase substrings to match in
# the question.
SEED_HANDLES: dict[str, list[str]] = {
    "israel": ["idfofficial", "pikudhaoref_all", "amitsegal"],
    "gaza": ["idfofficial", "pikudhaoref_all"],
    "iran": ["iranintl_en", "iranwirep"],
    "ukraine": ["zelenskiyofficial", "general_staff_ua", "kyivindependent_official"],
    "russia": ["kremlinrussia_e", "rian_ru"],
    "north korea": ["kcna_e"],
    "china": ["xinhuanet"],
    "taiwan": ["focustaiwan_news"],
    "venezuela": ["venezuelaalday"],
}


def seeds_for(question: str) -> list[str]:
    q = (question or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for kw, handles in SEED_HANDLES.items():
        if kw in q:
            for h in handles:
                if h not in seen:
                    seen.add(h)
                    out.append(h)
    return out
