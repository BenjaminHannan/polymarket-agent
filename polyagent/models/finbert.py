"""FinBERT direction classifier.

ProsusAI/finbert outputs three classes: positive / negative / neutral on
financial text. We map that to a continuous sentiment score in [-1, +1].
Combined with the question-polarity heuristic from signals/direction.py
to produce a market-side direction (yes/no).

Replaces VADER which is finance-blind ("Bitcoin surges past 130k" is
neutral to VADER but obviously bullish here).

GPU-accelerated. First load downloads ~440 MB from HuggingFace; cached
under ~/.cache/huggingface/hub afterwards.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger()

MODEL_ID = "ProsusAI/finbert"

_lock = threading.Lock()
_pipe: Optional[object] = None


def _get_pipe():
    """Singleton zero-shot text-classification pipeline on GPU."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        log.info("finbert_loading", model=MODEL_ID, device="cuda" if device == 0 else "cpu")
        _pipe = pipeline(
            "text-classification",
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            top_k=None,  # return all class probs
            device=device,
            truncation=True,
            max_length=512,
        )
        return _pipe


@dataclass
class FinSentiment:
    """Continuous sentiment in [-1, +1] derived from FinBERT class probs."""

    score: float       # P(positive) - P(negative); 0 = neutral
    confidence: float  # max class prob; how decisive the classification is
    raw: dict          # {"positive": p, "neutral": p, "negative": p}


def score_text(text: str) -> FinSentiment | None:
    if not text or not text.strip():
        return None
    try:
        pipe = _get_pipe()
        out = pipe(text)[0]  # list of dicts
        probs = {item["label"].lower(): float(item["score"]) for item in out}
    except Exception as e:
        log.warning("finbert_error", err=str(e))
        return None
    pos = probs.get("positive", 0.0)
    neg = probs.get("negative", 0.0)
    score = pos - neg
    confidence = max(probs.values()) if probs else 0.0
    return FinSentiment(score=score, confidence=confidence, raw=probs)


def score_batch(texts: list[str]) -> list[FinSentiment | None]:
    if not texts:
        return []
    try:
        pipe = _get_pipe()
        results = pipe(texts, batch_size=32)
    except Exception as e:
        log.warning("finbert_batch_error", err=str(e))
        return [None] * len(texts)
    out: list[FinSentiment | None] = []
    for r in results:
        if not r:
            out.append(None)
            continue
        probs = {item["label"].lower(): float(item["score"]) for item in r}
        pos = probs.get("positive", 0.0)
        neg = probs.get("negative", 0.0)
        out.append(FinSentiment(score=pos - neg, confidence=max(probs.values()), raw=probs))
    return out
