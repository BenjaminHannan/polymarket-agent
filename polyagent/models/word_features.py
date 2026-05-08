"""Word-level features for the LGBM model.

Hand-coded keyword features (q_pos_event_hits, cat_*) capture the obvious
patterns. This adds learned word-level signal via a hashing vectorizer:
stateless (no fit step), fixed-dimension (deterministic across train/infer),
captures both single words and bigrams.

Why hashing instead of TF-IDF?
- No vocabulary state to persist alongside the model.
- Robust to out-of-vocabulary words at inference time.
- Cheap memory/CPU.
The downside is collisions on the hash; with 256 dims and short questions
collisions are rare enough not to matter at our scale.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer

WORD_DIMS = 256

_VECTORIZER = HashingVectorizer(
    n_features=WORD_DIMS,
    ngram_range=(1, 2),
    analyzer="word",
    alternate_sign=False,   # all-positive features (LGBM doesn't care, but clearer)
    lowercase=True,
    norm=None,
    stop_words="english",
)


def vectorize(question: str) -> dict[str, float]:
    """Returns a dict {wf_0: count, wf_1: count, ...} of WORD_DIMS features."""
    if not question:
        return {f"wf_{i}": 0.0 for i in range(WORD_DIMS)}
    sparse = _VECTORIZER.transform([question])
    arr = sparse.toarray()[0]
    return {f"wf_{i}": float(arr[i]) for i in range(WORD_DIMS)}


def vectorize_batch(questions: list[str]) -> list[dict[str, float]]:
    if not questions:
        return []
    sparse = _VECTORIZER.transform(questions)
    arr = sparse.toarray()
    out = []
    for r in arr:
        out.append({f"wf_{i}": float(r[i]) for i in range(WORD_DIMS)})
    return out
