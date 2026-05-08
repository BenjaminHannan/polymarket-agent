"""Hybrid BM25 + dense article retrieval for the LLM forecaster.

Today the LLM forecaster receives the most-recent K news titles within
an asof-clean window — pure recency, no question-relevance ranking.
That's wasteful: a question like "Will Bitcoin hit $150k by June 30"
needs articles about *Bitcoin*, not "the latest article in the
window". The 2025 RAG benchmarks (BM25-to-Corrective-RAG, ColBERTv2)
report +17pp MRR@3 from hybrid BM25+dense+rerank vs dense-alone, and
on financial documents BM25 alone often beats dense — Polymarket
questions are entity- and date-heavy, the textbook BM25 regime.

Pipeline:
  1. Pull asof-clean candidate news titles from the store (recency-
     bounded so we don't BM25 over years of data).
  2. Score each title against the question via BM25 (rank_bm25).
  3. Score each title against the question via the BGE embedder
     (cosine — we already have the embedder loaded).
  4. RRF fuse the two rankings (k=60, standard) and return top-K.

Caching is per-question for the duration of one forecast call; we
don't memoize across calls because news arrives continuously.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger()


_TOKEN_RE = re.compile(r"[A-Za-z0-9$%]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]


def _rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Each input is a list of doc-indices in
    rank order. Returns a fused list of (doc_idx, score) sorted desc."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


@dataclass
class HybridArticleRetriever:
    """One-shot retriever: pass a question and a candidate corpus,
    get back the top-K articles by RRF(BM25, dense)."""
    top_k: int = 6
    rrf_k: int = 60

    def retrieve(
        self,
        question: str,
        candidates: list[str],
        candidate_embeddings: np.ndarray | None = None,
        question_embedding: np.ndarray | None = None,
    ) -> list[str]:
        """Returns top-K candidate titles ranked by hybrid score."""
        if not candidates:
            return []
        if not question:
            return list(candidates[: self.top_k])
        # 1) BM25 ranking
        tokenized = [_tokenize(c) for c in candidates]
        try:
            bm25 = BM25Okapi(tokenized)
            q_tokens = _tokenize(question)
            bm25_scores = bm25.get_scores(q_tokens)
            bm25_order = np.argsort(-bm25_scores).tolist()
        except Exception as e:
            log.warning("bm25_failed", err=str(e))
            bm25_order = list(range(len(candidates)))

        # 2) Dense ranking (cosine) — only if embeddings provided.
        if (
            candidate_embeddings is not None
            and question_embedding is not None
            and len(candidate_embeddings) == len(candidates)
        ):
            qv = question_embedding / (np.linalg.norm(question_embedding) + 1e-9)
            cv = candidate_embeddings / (
                np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
            )
            sims = (cv @ qv).flatten()
            dense_order = np.argsort(-sims).tolist()
            rankings = [bm25_order, dense_order]
        else:
            rankings = [bm25_order]

        # 3) RRF fuse
        fused = _rrf_fuse(rankings, k=self.rrf_k)
        top_idx = [idx for idx, _ in fused[: self.top_k]]
        return [candidates[i] for i in top_idx]
