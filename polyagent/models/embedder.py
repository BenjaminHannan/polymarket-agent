"""Sentence-transformer embedder for question semantics.

Loads `all-MiniLM-L6-v2` (22M params, 384-dim output, ~80 MB on disk).
Single shared instance per process; GPU-accelerated when available.

Used for:
1. LGBM features: each question becomes a 384-dim semantic vector,
   added alongside hand-coded keyword features. Captures patterns
   that "Will Bitcoin..." and "Will BTC..." share that hashing
   collisions or keyword lists miss.
2. (Future) news → market matching: cosine similarity between news
   text and market question, replacing keyword Jaccard.
"""

from __future__ import annotations

import threading
from typing import Optional

import structlog

log = structlog.get_logger()

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIMS = 1024

_lock = threading.Lock()
_model: Optional[object] = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("embedder_loading", model=EMBED_MODEL, device=device)
        _model = SentenceTransformer(EMBED_MODEL, device=device)
        return _model


def embed(text: str) -> list[float]:
    if not text:
        return [0.0] * EMBED_DIMS
    m = _get_model()
    v = m.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return v[0].tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Vectorize a batch of texts. Empty/None inputs get a zero vector."""
    if not texts:
        return []
    m = _get_model()
    safe = [t if t else " " for t in texts]
    v = m.encode(
        safe,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    out: list[list[float]] = []
    for i, t in enumerate(texts):
        if not t:
            out.append([0.0] * EMBED_DIMS)
        else:
            out.append(v[i].tolist())
    return out


def embed_features(text: str) -> dict[str, float]:
    """Return embedding as a dict {emb_0: ..., emb_1: ..., ...} for LGBM."""
    vec = embed(text)
    return {f"emb_{i}": float(vec[i]) for i in range(EMBED_DIMS)}


def embed_features_batch(texts: list[str]) -> list[dict[str, float]]:
    vecs = embed_batch(texts)
    return [{f"emb_{i}": float(v[i]) for i in range(EMBED_DIMS)} for v in vecs]
