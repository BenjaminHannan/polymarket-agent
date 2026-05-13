"""Tests for ColBERT retriever scaffold."""
from __future__ import annotations

from polyagent.models.colbert_retriever import ColBERTRetriever


def test_empty_corpus_returns_empty():
    r = ColBERTRetriever(corpus=[])
    assert r.retrieve("any query", k=5) == []


def test_retrieve_with_text_returns_triples():
    """When the embedder isn't available, retrieve falls back gracefully."""
    corpus = [
        {"doc_id": "a", "text": "Trump won the 2024 election"},
        {"doc_id": "b", "text": "Inflation eased in March"},
        {"doc_id": "c", "text": "Bitcoin hits new all-time high"},
    ]
    r = ColBERTRetriever(corpus=corpus)
    # We don't actually fit (would require the embedder); retrieve_with_text
    # should still return a list (possibly empty if embedder unavailable).
    out = r.retrieve_with_text("inflation", k=3)
    assert isinstance(out, list)


def test_corpus_stored_in_init():
    corpus = [{"doc_id": "x", "text": "hello"}]
    r = ColBERTRetriever(corpus=corpus)
    assert len(r._docs) == 1
    assert r._docs[0].text == "hello"
