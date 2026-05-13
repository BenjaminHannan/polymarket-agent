"""NegRisk semantic clustering scaffold (Saguillo et al. AFT 2025).

Direct implementation of the doc's Problem-6 fix #1: Saguillo's IMDEA
paper (arXiv 2508.03474) reduces the O(2^(n+m)) NegRisk search to
tractable using Linq-Embed-Mistral embeddings + LLM relationship
extraction. This scaffold encodes the algorithm but defers the heavy
calls (embeddings, LLM) to the existing Polyagent infrastructure
(`polyagent/models/embedder.py`, `polyagent/models/llm_forecaster.py`).

Algorithm
---------
Saguillo's reduction (per their Section 3):

  1. Cluster questions by embedding similarity. NegRisk-eligible
     questions tend to share lexical structure ("Will X win the Y?"
     for varying X, fixed Y).
  2. Within each cluster, ask an LLM whether the questions form a
     **mutually-exclusive exhaustive** (MEE) set. NegRisk requires
     this; non-MEE clusters get the monotonicity-arb path
     (`monotonicity_arb.py`).
  3. For confirmed MEE clusters, check the sum-of-YES-prices
     constraint: |sum(p_yes) − 1| > epsilon ⇒ arb opportunity.
  4. Filter by `min_leg_size` (smallest leg's available size at the
     stale-price); reject arbs that can only be executed for trivial
     notional.

Empirical context (Saguillo paper):
  - $40M extracted Apr 2024 – Apr 2025
  - $29M from NegRisk rebalancing
  - 7,051 single-condition arb-opportunity conditions
  - 662 NegRisk-rebalancing markets
  - top wallet $2.01M across 4,049 trades

Why this is a scaffold rather than a turnkey implementation
-----------------------------------------------------------
The full Saguillo pipeline requires two ML deployments we don't currently
expose to Polyagent at runtime:

  1. **Linq-Embed-Mistral** — a specific Mistral-7B fine-tune for
     short-text similarity. We have `bge-large` in `embedder.py`
     which is a reasonable substitute but not validated against the
     paper's results.
  2. **LLM relationship extraction** — Saguillo uses GPT-4-class
     output to classify cluster relationships. Our gpt-oss-20B is
     ~order-of-magnitude smaller. Phi-4-mini fallback will be
     unreliable.

The scaffold *runs* on Polyagent's existing stack — but the
relationship-classifier prompt is a placeholder, and the user should
validate cluster quality before enabling auto-execution.
"""
from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# Prompt template for MEE classification. Saguillo paper used a
# longer few-shot prompt; this is a starting point.
_MEE_PROMPT = """You are classifying a cluster of prediction-market
questions. Answer YES iff the following questions form a
MUTUALLY-EXCLUSIVE EXHAUSTIVE set — meaning exactly one will resolve
YES (and the others will all resolve NO).

Examples of mutually-exclusive exhaustive sets:
- "Will Trump win 2024?" / "Will Harris win 2024?" / "Will another candidate win 2024?"
- "Will the FOMC cut by 50bps in March?" / "Will the FOMC cut by 25bps in March?"
  / "Will the FOMC hold rates in March?" / "Will the FOMC hike rates in March?"

NON-examples (these are not mutually-exclusive or not exhaustive):
- "Will the S&P hit 6000 by Dec?" / "Will the Nasdaq hit 20k by Dec?"
  (not mutually-exclusive — both can resolve YES)
- "Will Trump win 2024?" / "Will Trump win the Republican primary?"
  (subset relationship, not mutually-exclusive)

Questions:
{questions}

Answer: YES or NO. Then on a new line, briefly explain.
"""


@dataclass
class NegRiskCluster:
    cluster_id: str
    token_ids: list[str]
    questions: list[str]
    yes_prices: list[float]
    sum_yes: float
    arb_gap: float           # |sum_yes − 1|
    mee_confirmed: bool      # True if LLM said MEE
    confidence: float        # cluster-cohesion score in [0, 1]


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS negrisk_clusters (
            cluster_id TEXT PRIMARY KEY,
            n_legs INTEGER NOT NULL,
            sum_yes REAL NOT NULL,
            arb_gap REAL NOT NULL,
            mee_confirmed INTEGER NOT NULL,
            confidence REAL NOT NULL,
            detected_ts REAL NOT NULL,
            token_ids TEXT NOT NULL,
            questions TEXT NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS negrisk_clusters_arb ON negrisk_clusters(mee_confirmed, arb_gap)"
    )
    conn.commit()


def _embed_questions(questions: list[str]):
    """Wrapper around `polyagent.models.embedder.embed_batch`. Returns
    np.ndarray of shape (N, D). Caller handles None when the embedder
    is unavailable."""
    try:
        from polyagent.models.embedder import embed_batch
        return embed_batch(questions)
    except Exception as e:
        log.warning("negrisk_embedder_unavailable", err=str(e))
        return None


def _cluster_by_similarity(
    embeddings,
    similarity_threshold: float = 0.85,
):
    """Greedy single-link clustering on cosine similarity. Returns
    list[list[int]] — each inner list is the indices of one cluster.

    Pure-Python so we don't take a hard sklearn dependency. For
    N ~ 500 markets this is O(N²) which is fine; for larger N use
    sklearn AgglomerativeClustering.
    """
    if embeddings is None or len(embeddings) == 0:
        return []
    import numpy as np
    emb = np.asarray(embeddings, dtype=float)
    # Normalize once for cosine sim.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    unit = emb / norms
    n = unit.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(unit[i] @ unit[j])
            if sim >= similarity_threshold:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


async def _llm_classify_mee(
    questions: list[str],
    llm_forecaster=None,
) -> tuple[bool, float]:
    """Ask the LLM whether `questions` form an MEE set. Returns
    (is_mee, confidence). If no LLM is available, returns
    (False, 0.0) — i.e. defaults to "not MEE" so we never auto-act on
    an unverified cluster."""
    if llm_forecaster is None:
        return False, 0.0
    prompt = _MEE_PROMPT.format(
        questions="\n".join(f"- {q}" for q in questions),
    )
    try:
        raw = await llm_forecaster.generate(prompt, max_new_tokens=64)
    except Exception as e:
        log.warning("negrisk_llm_classify_failed", err=str(e))
        return False, 0.0
    txt = (raw or "").strip().upper()
    is_mee = txt.startswith("YES")
    # Confidence proxy: 1.0 if response is short+decisive; 0.5 if hedged.
    conf = 0.9 if (is_mee and "BECAUSE" in txt) else (0.7 if is_mee else 0.0)
    return is_mee, conf


def detect_arb_candidates(
    clusters: list[NegRiskCluster],
    *,
    min_arb_gap: float = 0.02,
    min_leg_size: float | None = None,
    leg_size_lookup=None,
) -> list[NegRiskCluster]:
    """Filter MEE-confirmed clusters down to executable arbs.

    `min_arb_gap`: minimum |sum_yes − 1| to count as a tradeable
    opportunity (0.02 = 2 cents).

    `min_leg_size`: optional notional floor; reject clusters whose
    smallest leg has less than this available size. Requires
    `leg_size_lookup(token_id) -> float`.
    """
    out: list[NegRiskCluster] = []
    for c in clusters:
        if not c.mee_confirmed or c.arb_gap < min_arb_gap:
            continue
        if min_leg_size is not None and leg_size_lookup is not None:
            sizes = [float(leg_size_lookup(t)) for t in c.token_ids]
            if min(sizes) < min_leg_size:
                continue
        out.append(c)
    return out


async def scan_negrisk_clusters(
    markets: list,
    *,
    similarity_threshold: float = 0.85,
    llm_forecaster=None,
    conn: sqlite3.Connection | None = None,
) -> list[NegRiskCluster]:
    """End-to-end scan: embed → cluster → MEE-classify → arb-detect.

    `markets` is a list of objects with attributes
    (token_id, question, yes_price).

    Returns the full list of NegRiskCluster (MEE and non-MEE) so the
    caller can persist all of them and filter by mee_confirmed later.
    """
    if not markets:
        return []
    questions = [getattr(m, "question", "") for m in markets]
    tokens = [getattr(m, "token_id", "") for m in markets]
    prices = [getattr(m, "yes_price", None) for m in markets]
    embs = _embed_questions(questions)
    if embs is None:
        return []
    groups = _cluster_by_similarity(embs, similarity_threshold)
    results: list[NegRiskCluster] = []
    for gi, idxs in enumerate(groups):
        sub_q = [questions[i] for i in idxs]
        sub_t = [tokens[i] for i in idxs]
        sub_p = [prices[i] for i in idxs if prices[i] is not None]
        if len(sub_p) != len(idxs):
            continue
        sum_yes = float(sum(sub_p))
        arb_gap = abs(sum_yes - 1.0)
        mee, conf = await _llm_classify_mee(sub_q, llm_forecaster)
        cid = f"neg_{int(time.time())}_{gi}"
        cluster = NegRiskCluster(
            cluster_id=cid,
            token_ids=sub_t,
            questions=sub_q,
            yes_prices=sub_p,
            sum_yes=sum_yes,
            arb_gap=arb_gap,
            mee_confirmed=mee,
            confidence=conf,
        )
        results.append(cluster)
        if conn is not None:
            ensure_table(conn)
            conn.execute(
                """INSERT INTO negrisk_clusters
                   (cluster_id, n_legs, sum_yes, arb_gap, mee_confirmed,
                    confidence, detected_ts, token_ids, questions)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cluster_id) DO UPDATE SET
                      sum_yes=excluded.sum_yes,
                      arb_gap=excluded.arb_gap,
                      mee_confirmed=excluded.mee_confirmed,
                      confidence=excluded.confidence""",
                (cid, len(sub_t), sum_yes, arb_gap, int(mee), conf,
                 time.time(), ",".join(sub_t), "\n".join(sub_q)),
            )
    if conn is not None:
        conn.commit()
    log.info("negrisk_clusters_scanned", n_clusters=len(results))
    return results
