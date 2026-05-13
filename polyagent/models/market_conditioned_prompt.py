"""Market-Conditioned Prompting (pmwhybetter.md Problem-2 #5).

Direct implementation of the doc's recommendation from arXiv 2602.21229
("Forecasting Future Language: Context Design for Mention Markets,"
2025): explicitly treat market-implied probability as a Bayesian
**prior** and condition the LLM update on textual context.

Why this matters
----------------
Polyagent's existing combiner does `log_pool(market_p, llm_p)` with a
fixed market weight. The published Market-Conditioned Prompting paper
finds that *injecting the market price into the prompt itself* + asking
the LLM to compute a posterior is materially better than mixing two
independent estimates at the output. The intuition: the LLM has access
to information the log-pool doesn't (the news context), and explicit
conditioning lets it weight the prior appropriately per-question.

This module provides a single function:
  `build_market_conditioned_prompt(question, market_p, news_context)`
that returns a string ready to feed `LLMForecaster.generate`.

Three sections, all in the doc's recipe:
  1. **Prior**: state the current market probability + brief reason
     (it has aggregate information).
  2. **Evidence**: the retrieved news context for the question.
  3. **Task**: ask the LLM to compute a posterior probability,
     justifying the *magnitude* of any update from the prior.

The published format adds:
  - "Anti-confabulation" instruction: if the evidence is silent or
    contradictory, prefer the prior.
  - Explicit log-odds-update target rather than absolute probability,
    so the LLM has to reason about magnitude not just sign.

API
---
- `build_market_conditioned_prompt(question, market_p, news_context,
    *, deadline=None, base_rate=None)` → str
- `parse_market_conditioned_response(text)` → (p_posterior, log_odds_update)
"""
from __future__ import annotations

import math
import re
from typing import Optional

import structlog

log = structlog.get_logger()


PROMPT_TEMPLATE = """You are a calibrated forecaster on a prediction
market. Your task is to combine a market-implied prior with retrieved
evidence and output a posterior probability for a binary question.

## Question
{question}

## Market-implied prior
The market currently prices YES at {market_p:.3f}.
{market_reason}

## Base rate (if available)
{base_rate_str}

## Retrieved evidence
{news_context}

## Task
1. Identify the 1–3 most decision-relevant facts from the evidence
   above. If the evidence is silent or contradictory, **stay close
   to the prior** — say so explicitly.
2. Compute a log-odds update from the prior: how many "nats" of
   evidence does this body of text contain in favour of YES (positive)
   or NO (negative)? A single mainstream-news headline rarely carries
   more than ±0.5 nats; a critical-decision-confirming primary source
   may carry ±1.5.
3. Apply the update to the prior log-odds and report a posterior
   probability in [0.001, 0.999].

## Required output format
Reasoning: <1–3 sentences>
Log-odds update: <signed decimal>
Posterior P(YES): <decimal in (0,1)>
"""


def build_market_conditioned_prompt(
    question: str,
    market_p: float,
    news_context: str,
    *,
    deadline: str | None = None,
    base_rate: float | None = None,
    market_reason: str | None = None,
) -> str:
    """Build the Bayesian-update prompt for a market-conditioned forecast.

    Args:
        question: the prediction-market question text.
        market_p: current YES price, in (0, 1).
        news_context: concatenated retrieved-document excerpts. Pass
            "(no relevant news retrieved)" when the retriever returned
            nothing — the doc's anti-confabulation clause depends on
            being told this explicitly.
        deadline: resolution deadline (informational; not used in the
            posterior calculation).
        base_rate: empirical historical rate of YES for this question
            family, if known. Helps the LLM avoid base-rate neglect.
        market_reason: optional override of the default "market includes
            all public information" stub. Used by domain-specific
            callers (e.g. weather forecaster).
    """
    market_p_clamped = max(0.001, min(0.999, float(market_p)))
    market_reason_default = (
        "Treat this as a prior that incorporates aggregate trader "
        "information — many participants have already weighed in. "
        "Significant deviations from this prior require strong evidence."
    )
    base_rate_str = (
        f"Historical YES rate for similar questions: {base_rate:.3f}"
        if base_rate is not None else
        "(no historical base rate available; rely on the market prior)"
    )
    return PROMPT_TEMPLATE.format(
        question=question.strip(),
        market_p=market_p_clamped,
        market_reason=market_reason or market_reason_default,
        base_rate_str=base_rate_str,
        news_context=news_context.strip() if news_context else
                     "(no relevant news retrieved)",
    )


_POST_PROB_RE = re.compile(
    r"Posterior\s*P\(?YES\)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_LOG_ODDS_RE = re.compile(
    r"Log-odds\s+update\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def parse_market_conditioned_response(text: str) -> tuple[float | None, float | None]:
    """Extract (posterior_p, log_odds_update) from the LLM response.

    Either may be None if the LLM didn't follow the format. Caller
    should fall back to a log-pool combination of (market_p, base_rate)
    in that case (which is what the existing combiner already does)."""
    if not text:
        return None, None
    p_post = None
    log_odds = None
    m_post = _POST_PROB_RE.search(text)
    if m_post:
        try:
            p_post = float(m_post.group(1))
            p_post = max(0.001, min(0.999, p_post))
        except ValueError:
            p_post = None
    m_lo = _LOG_ODDS_RE.search(text)
    if m_lo:
        try:
            log_odds = float(m_lo.group(1))
        except ValueError:
            log_odds = None
    return p_post, log_odds


def combine_with_explicit_update(
    market_p: float,
    log_odds_update: float | None,
    posterior_p: float | None,
    *,
    update_cap_nats: float = 1.5,
) -> float:
    """Compose the final posterior from the LLM's two outputs.

    If `posterior_p` was parsed cleanly, use it (trust the LLM's
    own arithmetic). If only `log_odds_update` was parsed, apply it
    to the market prior and clamp the update magnitude to
    `update_cap_nats` (the published Halawi cap; prevents a single
    LLM call from over-riding a liquid market).
    """
    market_p = max(1e-4, min(1 - 1e-4, float(market_p)))
    if posterior_p is not None:
        return float(posterior_p)
    if log_odds_update is None:
        return market_p   # no update; defer to the market
    capped = max(-update_cap_nats, min(update_cap_nats, float(log_odds_update)))
    prior_log_odds = math.log(market_p / (1 - market_p))
    posterior_log_odds = prior_log_odds + capped
    return float(1.0 / (1.0 + math.exp(-posterior_log_odds)))
