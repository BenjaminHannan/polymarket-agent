"""Monotonicity-arb detector (Saguillo et al. AFT 2025 Section 5).

The doc's Problem-6 fix #5 enumerates a constraint family that lives
*outside* NegRisk: questions where one is a strict superset of another.

  - "Trump wins" ⊆ "Republican wins"
  - "Bitcoin > $80k on Dec 31" ⊆ "Bitcoin > $50k on Dec 31"
  - "Hurricane lands in Florida" ⊆ "Atlantic hurricane forms"

Saguillo's 2024 enumeration found 13 dependent U.S. election pairs
with 5 yielding profitable monotonicity arbs. They are *separate* from
NegRisk (which constrains mutually-exclusive sets to sum to 1) — the
constraint here is **p(A) ≤ p(B)** when A ⊆ B as resolution events.

This module is *not* a full LLM-relationship extractor (that's the
`negrisk_clustering.py` Saguillo-methodology piece). It catches the
high-precision low-recall families:

  1. **Strict-numerical-bound monotonicity** — "X > Y by date D" where
     Y is a numerical threshold and the questions share calendar and
     subject. p("X > 80") ≤ p("X > 50") under the same D.
  2. **Hierarchical-resolution monotonicity** — "A wins the
     [sub-category]" ⊆ "A wins the [super-category]". Right now we
     only catch the explicit numerical case; sub/super-category needs
     LLM-extracted relationships (deferred).

When the rule p(A) ≤ p(B) is violated by the current YES prices, we
emit a `MonotonicityArb` opportunity with the implied leg-by-leg
trade: buy A's NO + buy B's YES (or sell A's YES + sell B's NO),
sized by the *minimum* of the two legs' available liquidity.

Storage
-------
Detected violations get persisted to `monotonicity_arbs`:
  pair_id TEXT PRIMARY KEY  -- hash(token_subset, token_superset)
  token_subset TEXT
  token_superset TEXT
  question_subset TEXT
  question_superset TEXT
  p_subset REAL
  p_superset REAL
  gap REAL                  -- p_subset − p_superset (positive == arb)
  min_size_at_or_better REAL
  detected_ts REAL
  resolved INTEGER          -- 0 if still open
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# Numerical-comparator pattern: "more than", ">", "above", "at least"
# followed by a number. Captures (threshold, unit).
_NUMERIC_BOUND_RE = re.compile(
    r"(?:more\s+than|greater\s+than|above|exceed(?:s|ed)?|at\s+least|>=|>)\s*"
    r"\$?(\d+(?:[.,]\d+)?(?:[kKmMbB])?)\s*"
    r"(\w+)?",
    re.IGNORECASE,
)

# Date suffix pattern: "by Dec 31", "on or before March 1, 2026"
_DATE_RE = re.compile(
    r"(?:by|before|on\s+or\s+before)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)


def _parse_threshold(value: str) -> float | None:
    v = value.replace(",", "").strip()
    multiplier = 1.0
    if v[-1:].lower() in ("k", "m", "b"):
        suffix = v[-1].lower()
        multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}[suffix]
        v = v[:-1]
    try:
        return float(v) * multiplier
    except ValueError:
        return None


def _normalize_subject(question: str) -> str:
    """Strip the numerical bound and date from a question, leaving only
    the subject. Two questions with the same subject + date but
    different thresholds form a monotonicity pair."""
    q = _NUMERIC_BOUND_RE.sub("", question)
    q = re.sub(r"\s+", " ", q).strip().lower()
    return q


@dataclass
class MonotonicityCandidate:
    """A pair (subset, superset) detected to violate the p(A) ≤ p(B)
    constraint. The 'subset' is the strictly more-restrictive
    question; its YES price must be ≤ the 'superset' YES price."""
    token_subset: str
    token_superset: str
    question_subset: str
    question_superset: str
    p_subset: float
    p_superset: float
    threshold_subset: float
    threshold_superset: float

    @property
    def gap(self) -> float:
        """Positive gap == arb opportunity. p(A) > p(B) violates."""
        return float(self.p_subset - self.p_superset)

    @property
    def pair_id(self) -> str:
        h = hashlib.sha1(
            (self.token_subset + "|" + self.token_superset).encode("utf-8")
        ).hexdigest()
        return f"mono_{h[:16]}"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS monotonicity_arbs (
            pair_id TEXT PRIMARY KEY,
            token_subset TEXT NOT NULL,
            token_superset TEXT NOT NULL,
            question_subset TEXT NOT NULL,
            question_superset TEXT NOT NULL,
            p_subset REAL NOT NULL,
            p_superset REAL NOT NULL,
            gap REAL NOT NULL,
            min_size_at_or_better REAL,
            detected_ts REAL NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS mono_open ON monotonicity_arbs(resolved, gap)"
    )
    conn.commit()


def detect_pairs(markets: list) -> list[MonotonicityCandidate]:
    """Group markets by normalized subject, then within each group
    look for numerical-bound pairs where the higher-threshold price
    exceeds the lower-threshold price (a violation of monotonicity).

    `markets` is a list of objects with attributes:
        token_id (str), question (str), yes_price (float)
    Anything missing yes_price is skipped.
    """
    by_subject: dict[str, list] = {}
    for m in markets:
        question = getattr(m, "question", None)
        yes_price = getattr(m, "yes_price", None)
        token = getattr(m, "token_id", None)
        if question is None or yes_price is None or token is None:
            continue
        match = _NUMERIC_BOUND_RE.search(question)
        if not match:
            continue
        threshold = _parse_threshold(match.group(1))
        if threshold is None:
            continue
        subject = _normalize_subject(question)
        by_subject.setdefault(subject, []).append(
            (threshold, token, question, float(yes_price))
        )

    candidates: list[MonotonicityCandidate] = []
    for subject, items in by_subject.items():
        if len(items) < 2:
            continue
        # Sort by threshold ascending: lower-threshold = superset.
        items.sort(key=lambda x: x[0])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                th_super, tok_super, q_super, p_super = items[i]
                th_sub, tok_sub, q_sub, p_sub = items[j]
                if th_sub <= th_super:
                    continue  # not a strict superset relationship
                if p_sub > p_super:
                    candidates.append(MonotonicityCandidate(
                        token_subset=tok_sub,
                        token_superset=tok_super,
                        question_subset=q_sub,
                        question_superset=q_super,
                        p_subset=p_sub,
                        p_superset=p_super,
                        threshold_subset=th_sub,
                        threshold_superset=th_super,
                    ))
    if candidates:
        log.info("monotonicity_arb_candidates", n=len(candidates))
    return candidates


def persist_candidates(
    conn: sqlite3.Connection,
    candidates: list[MonotonicityCandidate],
    *,
    min_size_lookup=None,
) -> int:
    """Insert detected candidates into the `monotonicity_arbs` table.
    Returns count inserted/updated.

    `min_size_lookup` (optional): callable (token_id, side) -> float
    returning available size at price = our quote. Used to fill the
    `min_size_at_or_better` column for the executable-arb filter.
    """
    ensure_table(conn)
    now = time.time()
    n = 0
    for c in candidates:
        if c.gap <= 0:
            continue
        min_size = None
        if min_size_lookup is not None:
            try:
                a = float(min_size_lookup(c.token_subset, "NO"))
                b = float(min_size_lookup(c.token_superset, "YES"))
                min_size = min(a, b)
            except Exception:
                min_size = None
        conn.execute(
            """INSERT INTO monotonicity_arbs
               (pair_id, token_subset, token_superset, question_subset,
                question_superset, p_subset, p_superset, gap,
                min_size_at_or_better, detected_ts, resolved)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
               ON CONFLICT(pair_id) DO UPDATE SET
                  p_subset=excluded.p_subset,
                  p_superset=excluded.p_superset,
                  gap=excluded.gap,
                  min_size_at_or_better=excluded.min_size_at_or_better,
                  detected_ts=excluded.detected_ts""",
            (c.pair_id, c.token_subset, c.token_superset, c.question_subset,
             c.question_superset, c.p_subset, c.p_superset, c.gap,
             min_size, now),
        )
        n += 1
    conn.commit()
    return n


def open_candidates(conn: sqlite3.Connection, min_gap: float = 0.02) -> list[dict]:
    """Query unresolved monotonicity arbs with gap >= min_gap."""
    ensure_table(conn)
    rows = conn.execute(
        """SELECT pair_id, token_subset, token_superset, question_subset,
                  question_superset, p_subset, p_superset, gap,
                  min_size_at_or_better, detected_ts
           FROM monotonicity_arbs
           WHERE resolved = 0 AND gap >= ?
           ORDER BY gap DESC""",
        (float(min_gap),),
    ).fetchall()
    cols = ("pair_id", "token_subset", "token_superset", "question_subset",
            "question_superset", "p_subset", "p_superset", "gap",
            "min_size_at_or_better", "detected_ts")
    return [dict(zip(cols, r)) for r in rows]
