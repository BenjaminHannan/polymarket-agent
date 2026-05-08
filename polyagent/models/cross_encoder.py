"""Cross-encoder reranker for news → market relevance.

Bi-encoder (bge-large) is fast and good for retrieval. Cross-encoder is much
slower per-pair but produces a single relevance score that's substantially
more accurate. Use it to rerank the top-K candidates from the bi-encoder.

`cross-encoder/ms-marco-MiniLM-L-12-v2` — 33M params, ~120 MB. Very fast on
GPU; ~5ms per pair at batch sizes we care about.
"""

from __future__ import annotations

import threading
from typing import Optional

import structlog

log = structlog.get_logger()

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-12-v2"

_lock = threading.Lock()
_model: Optional[object] = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("cross_encoder_loading", model=MODEL_ID, device=device)
        _model = CrossEncoder(MODEL_ID, device=device, max_length=256)
        return _model


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Score (query, doc) pairs. Higher = more relevant.

    Output is the model's logits (typically in [-15, +15]). Pass through a
    sigmoid if you want probabilities; for ranking purposes raw logits are fine.
    """
    if not pairs:
        return []
    m = _get_model()
    scores = m.predict(pairs, show_progress_bar=False)
    return [float(s) for s in scores]
