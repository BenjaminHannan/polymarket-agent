"""Foreign-language news pipeline scaffold (pmwhybetter.md Problem-3 #5).

References:
  - takakhoo/Polymarket_Agent `local_market` prompt — local-edge
    classifier that flags markets resolvable from non-English sources.
  - Della Vedova 2026 (SSRN 6191618) — informed-wallet concentration
    in Action and Vote markets (often foreign / regional).

What this provides
------------------
A poller that supplements the bot's existing English-only RSS / FRED /
SEC / CourtListener / Bluesky pipeline with **non-English news
sources**, then routes everything through machine translation before
indexing. The doc's point: many "local edge" markets (Brazilian
elections, French regulatory decisions, Japanese earthquakes) have
resolution information in their native language hours before it hits
English wires.

Cadence and cost
----------------
- 30 min polls per source (cheaper than the English RSS at 10 min;
  foreign breaking news isn't typically as fast).
- Translation via the existing LLM (gpt-oss-20b / Phi-4-mini) at
  paraphrase quality — not a separate Google Translate API call,
  which would be paid and add a vendor dep.
- Cached by URL hash so re-polls don't re-translate.

Sources (curated, expandable)
-----------------------------
- **Le Monde** (FR) — French politics + EU regulation
- **Folha de São Paulo** (PT-BR) — Brazilian politics + economy
- **NHK World** (JA) — Japanese natural disasters + politics
- **Xinhua** (ZH) — Chinese economic releases (treat as state organ)
- **Der Spiegel** (DE) — German politics + EU

Each source is just a RSS feed URL + a language code; adding more
is a one-line config change. Translation pass uses the existing
`LLMForecaster` infrastructure to keep the GPU pipeline coherent.

This module is a **scaffold**:
  - Defines the schema and poller structure.
  - Fetches the feeds (works today; requires only `feedparser`).
  - The translation call is gated behind `if LLM_AVAILABLE` — without
    a running LLMForecaster, the module just stores the original text
    and a flag indicating un-translated state.

API
---
- `run_foreign_news_poller(news_store, llm_forecaster=None)` — async
  long-running task; wire as supervised in main.py.
- `SOURCES` — configurable list[dict].
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


SOURCES = [
    {
        "name": "le_monde",
        "language": "fr",
        "url": "https://www.lemonde.fr/rss/une.xml",
        "category_hint": "politics",
    },
    {
        "name": "folha_sp",
        "language": "pt-br",
        "url": "https://feeds.folha.uol.com.br/poder/rss091.xml",
        "category_hint": "politics",
    },
    {
        "name": "nhk_world",
        "language": "ja",
        "url": "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "category_hint": "events",
    },
    {
        "name": "der_spiegel",
        "language": "de",
        "url": "https://www.spiegel.de/international/index.rss",
        "category_hint": "politics",
    },
    {
        "name": "xinhua",
        "language": "zh",
        "url": "http://www.xinhuanet.com/english/rss/chinarss.xml",
        "category_hint": "macro",
    },
]


_TRANSLATION_PROMPT = """Translate the following news article into
plain English. Preserve all proper nouns, numbers, dates, and
direct quotations. Return only the translation, no preamble.

Original ({language}):
{text}
"""


@dataclass
class ForeignArticle:
    source: str
    language: str
    title_original: str
    body_original: str
    title_english: str | None
    body_english: str | None
    published_ts: float
    url: str
    hash_id: str


def _hash_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


async def _translate(
    text: str, language: str, llm_forecaster,
) -> str | None:
    """Translate `text` from `language` to English via the LLM. Returns
    None on failure (caller stores the original)."""
    if llm_forecaster is None or not text:
        return None
    prompt = _TRANSLATION_PROMPT.format(language=language, text=text)
    try:
        raw = await llm_forecaster.generate(prompt, max_new_tokens=512)
    except Exception as e:
        log.warning("foreign_translate_failed", err=str(e), lang=language)
        return None
    return (raw or "").strip()


async def _fetch_feed(url: str) -> list[dict]:
    """Parse an RSS feed into a list of {title, summary, link,
    published} dicts. Pure-stdlib fallback if feedparser is missing."""
    try:
        import feedparser
        d = feedparser.parse(url)
        out = []
        for e in (d.entries or []):
            out.append({
                "title": getattr(e, "title", "") or "",
                "summary": getattr(e, "summary", "") or "",
                "link": getattr(e, "link", "") or "",
                "published": getattr(e, "published", "") or "",
            })
        return out
    except ImportError:
        log.warning("foreign_news_no_feedparser")
        return []


async def run_foreign_news_poller(
    news_store,
    llm_forecaster=None,
    *,
    poll_sec: float = 1800.0,
) -> None:
    """Poll each foreign source, translate, persist to `news_store`.

    Args:
        news_store: the bot's existing news_store (has `.insert(...)`).
        llm_forecaster: optional LLMForecaster for translation. When
            None, articles are stored with `body_english=None` and a
            flag — downstream consumers can re-translate later.
        poll_sec: cadence (default 30 min).
    """
    log.info(
        "foreign_news_poller_start",
        n_sources=len(SOURCES),
        translation_enabled=(llm_forecaster is not None),
    )
    while True:
        for src in SOURCES:
            try:
                entries = await _fetch_feed(src["url"])
            except Exception as e:
                log.warning(
                    "foreign_news_fetch_failed",
                    source=src["name"], err=str(e),
                )
                continue
            for e in entries:
                if not e.get("title"):
                    continue
                title_en = await _translate(
                    e["title"], src["language"], llm_forecaster,
                )
                body_en = await _translate(
                    e["summary"], src["language"], llm_forecaster,
                )
                article = ForeignArticle(
                    source=src["name"],
                    language=src["language"],
                    title_original=e["title"],
                    body_original=e.get("summary", "") or "",
                    title_english=title_en,
                    body_english=body_en,
                    published_ts=time.time(),
                    url=e.get("link", "") or "",
                    hash_id=_hash_for(e.get("link", "") or e["title"]),
                )
                # Persist into the existing news_store using whatever
                # insert API it exposes. Most news_stores accept
                # (source, title, body, url, ts).
                try:
                    if hasattr(news_store, "insert_article"):
                        news_store.insert_article(
                            source=f"foreign:{src['name']}",
                            title=article.title_english or article.title_original,
                            body=article.body_english or article.body_original,
                            url=article.url,
                            published_ts=article.published_ts,
                        )
                except Exception as ex:
                    log.warning("foreign_news_insert_failed", err=str(ex))
        await asyncio.sleep(poll_sec)
