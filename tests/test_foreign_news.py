"""Tests for foreign-language news pipeline scaffold."""
from __future__ import annotations

import asyncio

from polyagent.data.foreign_news import (
    SOURCES, _hash_for, _translate, ForeignArticle,
)


def test_sources_have_required_fields():
    for src in SOURCES:
        assert "name" in src
        assert "language" in src
        assert "url" in src
        assert src["url"].startswith("http")


def test_hash_for_deterministic():
    a = _hash_for("https://example.com/article-1")
    b = _hash_for("https://example.com/article-1")
    c = _hash_for("https://example.com/article-2")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_translate_returns_none_without_llm():
    """No LLM forecaster ⇒ translation returns None."""
    out = asyncio.run(_translate("Bonjour", "fr", None))
    assert out is None


def test_foreign_article_construct():
    a = ForeignArticle(
        source="le_monde", language="fr",
        title_original="Titre", body_original="Corps",
        title_english=None, body_english=None,
        published_ts=1.0, url="https://example.com", hash_id="abc",
    )
    assert a.source == "le_monde"
    assert a.title_english is None
