"""News -> market direction classifier (lexicon-only, no LLM).

Combines:
1. VADER compound sentiment over the news headline + body.
2. A question-polarity heuristic: most prediction-market questions are framed
   as "will X happen?", so a YES outcome is the positive event by default.
   Questions about negative events ("out", "lose", "fail", "war") flip that.

direction = sign(sentiment * question_polarity)

Calibration is poor without resolved-market labels; this exists to gate
paper trades and to produce labeled candidates we'll later refine with a
real classifier.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# FinBERT is heavier but finance-aware. Opt in via env so we can A/B.
_USE_FINBERT = os.getenv("USE_FINBERT", "1") == "1"

# Words in a market question that flip "positive sentiment -> YES" to "positive sentiment -> NO".
_NEGATIVE_EVENT_WORDS = frozenset(
    """
    out fire fired resign resigns resignation step-down impeach impeached impeachment
    leave leaves lose loses losing loss default defaults
    fall falls falling fell decline declines declining drop drops dropping dropped
    fail fails failing ban bans banned block blocks blocked
    war conflict invasion invade invades invaded recession crash crashing slump dump
    bankruptcy bankrupt collapse collapses collapsing collapsed default defaults
    strike strikes attack attacks attacked
    indicted indictment convicted conviction guilty
    shutdown shutdowns shutdown
    """.split()
)

# Strong "positive event" words to break ties when both signals appear.
_POSITIVE_EVENT_WORDS = frozenset(
    """
    win wins winning won succeed succeeds approve approves approved approval
    pass passes passed passing rise rises rising rose
    deal deals agreement ceasefire peace settle settles settled settlement
    elected election confirm confirms confirmed confirmation
    above exceed exceeds exceeded hit hits hitting hits
    grow grows growing growth surge surges surged rally rallies rallied
    """.split()
)

_TOK = re.compile(r"[A-Za-z][A-Za-z\-']+")


def question_polarity(question: str) -> int:
    """+1 if a positive event is the YES side, -1 if a negative event is YES, 0 if ambiguous."""
    toks = {t.lower() for t in _TOK.findall(question)}
    neg_hits = len(toks & _NEGATIVE_EVENT_WORDS)
    pos_hits = len(toks & _POSITIVE_EVENT_WORDS)
    if neg_hits > pos_hits and neg_hits > 0:
        return -1
    if pos_hits > 0 and neg_hits == 0:
        return 1
    if pos_hits == 0 and neg_hits == 0:
        return 1  # default: framing usually treats YES as the headline event happening
    return 0  # tie, both sides keyword-loaded → unknown


@dataclass
class DirectionResult:
    direction: str  # "yes" | "no" | "unknown"
    confidence: float  # 0..1, based on |sentiment| (capped)
    sentiment: float  # VADER compound, [-1, 1]
    polarity: int  # -1, 0, +1


def _finbert_score(text: str) -> float | None:
    """Returns FinBERT continuous score in [-1, +1], or None on error."""
    try:
        from polyagent.models.finbert import score_text
        r = score_text(text)
        return r.score if r is not None else None
    except Exception:
        return None


def classify(news_text: str, question: str, sentiment_threshold: float = 0.3) -> DirectionResult:
    """Classify news → market direction.

    Uses FinBERT when USE_FINBERT=1 (default), else falls back to VADER.
    Multiplies the sentiment by the question-polarity heuristic to map
    sentiment to YES/NO direction.
    """
    if _USE_FINBERT:
        compound = _finbert_score(news_text or "")
        if compound is None:
            # Fall back if FinBERT errors
            score = _vader.polarity_scores(news_text or "")
            compound = float(score.get("compound", 0.0))
    else:
        score = _vader.polarity_scores(news_text or "")
        compound = float(score.get("compound", 0.0))
    pol = question_polarity(question)
    if pol == 0:
        return DirectionResult("unknown", 0.0, compound, pol)
    if abs(compound) < sentiment_threshold:
        return DirectionResult("unknown", abs(compound), compound, pol)
    direction = "yes" if (compound * pol) > 0 else "no"
    confidence = min(1.0, abs(compound))
    return DirectionResult(direction, confidence, compound, pol)
