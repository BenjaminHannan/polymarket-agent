"""Semantic news → market matcher using sentence embeddings.

Replaces the keyword-Jaccard matcher in signals/news_match.py.

At startup, every market question is embedded once into a 384-dim vector
and cached in a numpy matrix. Each news event embeds its title+body once
(GPU, ~3ms), then a single matmul against the cached matrix gives cosine
similarity to all markets in <1ms. Top-K above a threshold are emitted.

This is much more accurate than keyword overlap:
- "Iran-Israel ceasefire reached" matches "Will US x Iran peace deal..."
  even though they share zero rare keywords.
- "Asia stocks fall on rate hike fears" doesn't match "Lakers vs Rockets"
  even though they share "fall" and "rate" tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from polyagent.gamma import Market
from polyagent.models.cross_encoder import score_pairs as ce_score
from polyagent.models.embedder import embed, embed_batch, EMBED_DIMS

log = structlog.get_logger()


@dataclass
class SemanticMarketIndex:
    market_ids: list[str]
    questions: list[str]
    categories: list[str | None]
    embeddings: np.ndarray  # shape (N, EMBED_DIMS), L2-normalized
    market_objs: dict[str, Market]

    @classmethod
    def build(cls, markets: list[Market]) -> "SemanticMarketIndex":
        if not markets:
            return cls(
                market_ids=[],
                questions=[],
                categories=[],
                embeddings=np.zeros((0, EMBED_DIMS), dtype=np.float32),
                market_objs={},
            )
        ids = [m.condition_id for m in markets]
        qs = [m.question for m in markets]
        cats = [m.category for m in markets]
        log.info("semantic_index_building", n=len(markets))
        vecs = embed_batch(qs)
        emb = np.asarray(vecs, dtype=np.float32)  # already L2-normalized in embedder
        idx = cls(
            market_ids=ids,
            questions=qs,
            categories=cats,
            embeddings=emb,
            market_objs={m.condition_id: m for m in markets},
        )
        log.info("semantic_index_built", n=len(markets), shape=list(emb.shape))
        return idx

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        min_sim: float = 0.30,
        rerank_pool: int = 20,
        use_cross_encoder: bool = True,
    ) -> list[tuple[str, float, str]]:
        """Return list of (condition_id, score, question) for top_k matches above min_sim.

        Two-stage:
          1) Bi-encoder cosine sim picks top `rerank_pool` candidates (cheap).
          2) Cross-encoder reranks them precisely (more expensive, but only on K).
        Final score = cross-encoder logit if use_cross_encoder else cosine sim.
        """
        if not query_text or self.embeddings.shape[0] == 0:
            return []
        q = embed(query_text)
        qv = np.asarray(q, dtype=np.float32)
        sims = self.embeddings @ qv  # (N,)
        if sims.size == 0:
            return []
        n = sims.size
        pool = min(max(rerank_pool, top_k), n)
        top_idx = np.argpartition(-sims, kth=pool - 1)[:pool]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        # Pre-filter: drop anything below min_sim before reranking
        top_idx = [int(i) for i in top_idx if float(sims[i]) >= min_sim]
        if not top_idx:
            return []
        if use_cross_encoder and len(top_idx) > 1:
            try:
                import math
                pairs = [(query_text, self.questions[i]) for i in top_idx]
                logits = ce_score(pairs)
                # Sigmoid to [0, 1] so the score stays comparable to cosine.
                probs = [1.0 / (1.0 + math.exp(-x)) for x in logits]
                scored = sorted(
                    zip(top_idx, probs), key=lambda t: -t[1]
                )[:top_k]
                return [
                    (self.market_ids[i], float(s), self.questions[i])
                    for i, s in scored
                ]
            except Exception as e:
                log.warning("cross_encoder_error_falling_back", err=str(e))
        # Cosine-only fallback
        return [
            (self.market_ids[i], float(sims[i]), self.questions[i])
            for i in top_idx[:top_k]
        ]
