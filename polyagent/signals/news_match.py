"""Lightweight news -> market matcher with directional classifier.

Pipeline per news event:
  1. Tokenize news text.
  2. Find markets with overlap >= min_overlap (Jaccard-like ranking).
  3. For the top candidates, classify direction via VADER + question polarity.
  4. Persist all candidates to the signals table.
  5. Hand the strongest directional candidate (if any) to a trader callback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import structlog

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.models.news_embed_matcher import SemanticMarketIndex
from polyagent.news_store import NewsEvent, NewsStore
from polyagent.signals.direction import DirectionResult, classify
from polyagent.signals import news_verifier as _nli

log = structlog.get_logger()

_STOPWORDS = frozenset(
    """
    the a an of to in on for and or by with from at as is are was were be been
    being have has had do does did will would should could may might must
    this that these those it its their there they them his her him she he we us
    our your you i me my mine yours theirs
    yes no will not but if then than which who what when where why how
    market markets bet bets question prediction polymarket
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    out = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


@dataclass
class MarketIndex:
    by_condition: dict[str, set[str]] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)
    categories: dict[str, str | None] = field(default_factory=dict)
    market_objs: dict[str, Market] = field(default_factory=dict)

    @classmethod
    def build(cls, markets: list[Market]) -> "MarketIndex":
        idx = cls()
        for m in markets:
            toks = tokenize(m.question)
            if not toks:
                continue
            idx.by_condition[m.condition_id] = toks
            idx.questions[m.condition_id] = m.question
            idx.categories[m.condition_id] = m.category
            idx.market_objs[m.condition_id] = m
        return idx

    def candidates(self, news_tokens: set[str], min_overlap: int) -> list[tuple[str, int, set[str]]]:
        out: list[tuple[str, int, set[str]]] = []
        for cid, mtoks in self.by_condition.items():
            shared = news_tokens & mtoks
            if len(shared) >= min_overlap:
                out.append((cid, len(shared), shared))
        out.sort(key=lambda t: t[1], reverse=True)
        return out


# Trader callback signature: (market, direction_result, news_event, score) -> awaitable
TraderCallback = Callable[[Market, DirectionResult, NewsEvent, float], Awaitable[None]]


@dataclass
class NewsMatcher:
    index: MarketIndex
    store: NewsStore
    min_overlap: int = settings.news_match_min_overlap
    trade_min_overlap: int = 3
    trade_min_confidence: float = 0.4
    trade_min_score: float = 0.20
    trader: Optional[TraderCallback] = None
    semantic_index: Optional[SemanticMarketIndex] = None
    semantic_min_sim: float = 0.40
    semantic_top_k: int = 5

    async def on_event(self, evt: NewsEvent) -> int:
        text = f"{evt.title} {evt.body}"
        h = evt.hash()

        # Prefer semantic matching when an embedding index is available.
        # Fall back to keyword overlap if not.
        scored: list[tuple[str, int, set[str], float]] = []
        if self.semantic_index is not None:
            sem_hits = self.semantic_index.search(
                text, top_k=self.semantic_top_k, min_sim=self.semantic_min_sim
            )
            for cid, sim, _q in sem_hits:
                # We still compute keyword overlap purely for explainability
                # in the signal detail (which words the news shares with the question).
                ntoks = tokenize(text)
                mtoks = self.index.by_condition.get(cid, set())
                shared = ntoks & mtoks
                # Use cosine similarity directly as the "score" — much more
                # meaningful than keyword Jaccard.
                scored.append((cid, len(shared), shared, sim))
            if not scored:
                return 0
        else:
            ntoks = tokenize(text)
            if len(ntoks) < self.min_overlap:
                return 0
            cands = self.index.candidates(ntoks, self.min_overlap)
            if not cands:
                return 0
            for cid, overlap, shared in cands[:5]:
                mtoks = self.index.by_condition[cid]
                denom = (max(len(ntoks), 1) * max(len(mtoks), 1)) ** 0.5
                scored.append((cid, overlap, shared, overlap / denom))

        # Persist all top-5 with a direction classification.
        best_for_trade: tuple[str, DirectionResult, float, int] | None = None
        nli_enabled = _nli.is_enabled()
        for cid, overlap, shared, score in scored:
            question = self.index.questions[cid]
            direction = classify(text, question)

            # NLI parallel verifier (A/B against the lexicon classifier).
            # Logs a separate `news_nli_match` signal so we can audit hit-rate
            # vs. the lexicon baseline once enough markets have resolved.
            # Default OFF; activate via ENABLE_NLI_VERIFIER=1.
            nli_detail: dict | None = None
            if nli_enabled:
                try:
                    r = _nli.verify(evt.title, question, body=evt.body)
                except Exception as e:
                    log.warning("nli_verify_error", err=str(e))
                    r = None
                if r is not None:
                    nli_detail = {
                        "direction": r.direction,
                        "confidence": round(r.confidence, 4),
                        "p_entail_yes": round(r.p_entail_yes, 4),
                        "p_entail_no": round(r.p_entail_no, 4),
                        "margin": round(r.margin, 4),
                        "elapsed_ms": round(r.elapsed_ms, 1),
                        "yes_hyp": r.yes_hypothesis[:120],
                        "no_hyp": r.no_hypothesis[:120],
                        "source": evt.source,
                        "title": evt.title[:140],
                        "lexicon_direction": direction.direction,
                    }
                    await self.store.insert_signal(
                        strategy="news_nli_match",
                        condition_id=cid,
                        direction=r.direction,
                        score=r.confidence,
                        news_hash=h,
                        detail=nli_detail,
                    )

            await self.store.insert_signal(
                strategy="news_keyword_match",
                condition_id=cid,
                direction=direction.direction,
                score=score,
                news_hash=h,
                detail={
                    "overlap": overlap,
                    "shared": sorted(shared)[:20],
                    "n_news_toks": len(ntoks),
                    "n_mkt_toks": len(self.index.by_condition[cid]),
                    "source": evt.source,
                    "title": evt.title[:140],
                    "sentiment": direction.sentiment,
                    "confidence": direction.confidence,
                    "polarity": direction.polarity,
                    **({"nli": nli_detail} if nli_detail else {}),
                },
            )
            log.info(
                "news_signal",
                source=evt.source,
                title=evt.title[:90],
                question=question[:90],
                overlap=overlap,
                score=round(score, 3),
                direction=direction.direction,
                conf=round(direction.confidence, 2),
                sentiment=round(direction.sentiment, 2),
            )

            # Eligible-to-trade: strong overlap, high confidence, decent cosine.
            if (
                self.trader is not None
                and direction.direction in ("yes", "no")
                and overlap >= self.trade_min_overlap
                and direction.confidence >= self.trade_min_confidence
                and score >= self.trade_min_score
            ):
                if best_for_trade is None or score > best_for_trade[2]:
                    best_for_trade = (cid, direction, score, overlap)

        if best_for_trade is not None and self.trader is not None:
            cid, direction, score, _ = best_for_trade
            market = self.index.market_objs.get(cid)
            if market is not None:
                await self.trader(market, direction, evt, score)

        return len(scored)
