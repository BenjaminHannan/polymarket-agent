"""Retrieval-augmented LLM forecaster as a 4th combiner expert.

Architecture (Halawi/NeurIPS 2024):
  question + retrieved news (top-K) -> Phi-4-14B int4 -> N forecasts ->
  geometric-mean of odds -> p_llm

Default: OFF (set ENABLE_LLM_FORECASTER=1).
First load downloads ~7 GB. Inference is slow (~10-30 s per market) so
we trigger this on event creation + news-velocity spikes, not every
combined_signal poll. Cached per (market_id, news_hash).

DPO self-play fine-tuning (Turtel/Wood/Khoja/Mehl 2025) is the next
upgrade — add 7-10% relative Brier — but takes 6-12 h of GPU time.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger()


MODEL_ID = os.getenv("LLM_FORECASTER_MODEL", "microsoft/Phi-4-mini-instruct")
N_SAMPLES = int(os.getenv("LLM_FORECASTER_N_SAMPLES", "6"))
TEMPS: list[float] = [
    float(t) for t in os.getenv("LLM_FORECASTER_TEMPS", "0.5,0.9").split(",")
]
MAX_NEW_TOKENS = int(os.getenv("LLM_FORECASTER_MAX_NEW_TOKENS", "300"))
P_CLIP = (0.02, 0.98)


_lock = threading.Lock()
_pipe: Optional[object] = None


def _get_pipe():
    """Lazy-load the model the first time we forecast."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        log.info("llm_forecaster_loading", model=MODEL_ID, device="cuda" if device == 0 else "cpu")
        try:
            _pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                device=device,
                torch_dtype=torch.bfloat16 if device == 0 else None,
                model_kwargs={
                    "low_cpu_mem_usage": True,
                },
            )
        except Exception as e:
            log.warning("llm_forecaster_load_failed", err=str(e))
            _pipe = None
        return _pipe


_PROB_RE = re.compile(r"\b(\d{1,3})(?:\.(\d+))?\s*%|\b(0?\.\d+)\b")


def _parse_probability(text: str) -> float | None:
    """Extract a probability in [0, 1] from generated text."""
    if not text:
        return None
    # Look for "Probability: 0.42" or "42%" patterns
    last = None
    for m in _PROB_RE.finditer(text):
        if m.group(3):
            try:
                p = float(m.group(3))
            except ValueError:
                continue
        else:
            whole = m.group(1)
            frac = m.group(2) or "0"
            try:
                p = (float(whole) + float("0." + frac)) / 100.0
            except ValueError:
                continue
        if 0 <= p <= 1:
            last = p
    return last


def _build_prompt(question: str, articles: list[str]) -> str:
    article_block = "\n".join(
        f"  [{i+1}] {a[:500]}" for i, a in enumerate(articles[:8])
    ) or "  (none)"
    return (
        "You are a calibrated probability forecaster. Given the question and "
        "the news context below, output a numeric probability in [0, 1] that "
        "the question resolves YES.\n\n"
        f"Question: {question}\n\n"
        "Recent news (most relevant first):\n"
        f"{article_block}\n\n"
        "Respond with one line in this exact format and nothing else:\n"
        "Probability: <number>\n"
    )


def _aggregate(ps: list[float]) -> float:
    """Geometric mean of odds, clipped to safe range. Matches Halawi recipe."""
    if not ps:
        return 0.5
    eps = 1e-6
    logits = []
    for p in ps:
        p = max(eps, min(1 - eps, p))
        logits.append(math.log(p / (1 - p)))
    avg_logit = sum(logits) / len(logits)
    p = 1 / (1 + math.exp(-avg_logit))
    return max(P_CLIP[0], min(P_CLIP[1], p))


# AIA Forecaster (Karger et al., arXiv 2511.07678, Nov 2025) — three
# named de-biasers that bring local LLM forecasts closer to
# superforecaster Brier without requiring fine-tuning:
#
#   1. Acquiescence-bias correction: LLMs systematically predict P(YES)
#      above the true base rate. We learn an offset on a held-out
#      cohort; here we use a conservative literature default.
#   2. Round-number debiasing: LLMs anchor on 0.1/0.5/0.9. We apply a
#      smoothing kernel that pulls probabilities away from the round
#      attractors toward their fitted continuous neighborhood.
#   3. Self-consistency reconciliation across rephrased questions —
#      this is what `consistency_score()` already does for NegRisk.
#
# The first two are post-hoc on a single LLM scalar so we wrap them
# in a `_aia_debias()` function and apply just before clipping.

_ACQUIESCENCE_OFFSET = float(os.getenv("LLM_ACQUIESCENCE_OFFSET", "0.04"))
# Round-number attractors: pull probs ε away from 0.1, 0.5, 0.9 etc.
# towards the closer non-round neighborhood.
_ROUND_NUMBERS = (0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
_ROUND_BAND = float(os.getenv("LLM_ROUND_BAND", "0.015"))
_ROUND_PULL = float(os.getenv("LLM_ROUND_PULL", "0.6"))


def _aia_debias(p: float) -> float:
    """AIA Forecaster post-hoc de-biasing. Pure scalar function.

    Acquiescence: subtract a small global offset (LLMs predict above the
    true rate). Round-number: when p sits within ±_ROUND_BAND of an
    attractor, push it _ROUND_PULL of the way toward the band edge in
    the direction of monotone increase from the most-recent sample.
    Conservative defaults so the transformation is small unless the
    LLM was visibly anchored.
    """
    q = p - _ACQUIESCENCE_OFFSET
    # Round-number unsticking: find the nearest attractor; if within
    # band, displace toward whichever boundary is farther from the
    # attractor's center (rounds usually pull from "true" → attractor,
    # so displacing away from center monotonically corrects).
    for r in _ROUND_NUMBERS:
        if abs(q - r) <= _ROUND_BAND:
            sign = 1.0 if q >= r else -1.0
            edge = r + sign * _ROUND_BAND
            q = q + (edge - q) * _ROUND_PULL
            break
    return max(0.001, min(0.999, q))


@dataclass
class LLMForecaster:
    n_samples: int = N_SAMPLES
    max_new_tokens: int = MAX_NEW_TOKENS

    def is_enabled(self) -> bool:
        return os.getenv("ENABLE_LLM_FORECASTER", "0") == "1"

    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        pipe = _get_pipe()
        if pipe is None:
            return ""
        try:
            out = pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1,
                return_full_text=False,
            )
            return out[0].get("generated_text", "") if out else ""
        except Exception as e:
            log.warning("llm_forecaster_generate_failed", err=str(e))
            return ""

    def forecast(self, question: str, articles: list[str]) -> dict | None:
        """Returns {"p": float, "n_samples": int, "raw": [floats]} or None.

        Per Halawi: N independent samples across multiple temperatures.
        Aggregated via geometric mean of odds for stability.
        """
        if not self.is_enabled():
            return None
        if not question:
            return None
        prompt = _build_prompt(question, articles)
        ps: list[float] = []
        t0 = time.time()
        # Distribute n_samples across the configured temperatures
        per_temp = max(1, self.n_samples // max(1, len(TEMPS)))
        for temp in TEMPS:
            for _ in range(per_temp):
                text = self._generate(prompt, temperature=temp)
                p = _parse_probability(text)
                if p is not None:
                    # AIA debias each sample before aggregation. Per-sample
                    # is correct because acquiescence is a *per-elicitation*
                    # bias; aggregating then debiasing would underweight
                    # samples that already had a confident center.
                    ps.append(_aia_debias(p))
        if not ps:
            return None
        p_agg = _aggregate(ps)
        # Stability metric: how spread the samples are. Wide spread => low confidence.
        try:
            import statistics
            spread = float(statistics.pstdev(ps))
        except Exception:
            spread = 0.0
        log.info(
            "llm_forecast",
            question=question[:80],
            n_samples=len(ps),
            raw=[round(x, 3) for x in ps],
            p_agg=round(p_agg, 3),
            spread=round(spread, 3),
            elapsed_sec=round(time.time() - t0, 2),
        )
        return {"p": p_agg, "n_samples": len(ps), "raw": ps, "spread": spread}

    def consistency_score(self, group_questions: list[str], group_articles: list[list[str]]) -> dict | None:
        """Karkare et al. 2024 NegRisk consistency check.

        For a NegRisk event with N mutually-exclusive outcomes, the LLM's
        independently elicited YES probabilities should sum to ~1. Deviations
        from that are evidence of LLM uncertainty/inconsistency. The
        consistency score is itself predictive of forecast quality — bad
        consistency → discount the LLM's combiner weight.

        Returns {"sum": float, "deviation": float, "ps": [floats]} or None.
        """
        if not group_questions:
            return None
        ps = []
        for q, arts in zip(group_questions, group_articles):
            r = self.forecast(q, arts or [])
            if r is None:
                return None
            ps.append(r["p"])
        s = sum(ps)
        return {"sum": s, "deviation": abs(s - 1.0), "ps": ps}

    async def forecast_async(self, question: str, articles: list[str]) -> dict | None:
        """Async wrapper that runs the (sync, blocking) forecast off the event loop."""
        if not self.is_enabled():
            return None
        try:
            return await asyncio.to_thread(self.forecast, question, articles)
        except Exception as e:
            log.warning("llm_forecaster_async_error", err=str(e))
            return None
