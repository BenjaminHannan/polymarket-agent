"""NLI-based news → market direction verifier.

Replaces the keyword + FinBERT/VADER pipeline with a zero-shot natural-
language-inference model that reads the headline and the market question
and decides whether the news entails YES, NO, or neither.

Architecture:

  premise = news headline (+ optional body snippet)
  hypothesis_yes = question phrased as a positive event
                   ("Manchester City won on 2026-05-09")
  hypothesis_no  = question phrased as a negative event
                   ("Manchester City did not win on 2026-05-09")
  direction = argmax over P_entail(yes), P_entail(no), abstain

Model: cross-encoder/nli-deberta-v3-small (~70M params, ~50ms on GPU per
pair). Lazy-loaded the first time `verify()` is called.

Default off: set ENABLE_NLI_VERIFIER=1 to activate. Runs alongside the
existing direction classifier in news_match.py — the existing one stays
the primary, the NLI is logged as a parallel signal so we can audit
hit-rate vs. the lexicon baseline before flipping the default.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger()


MODEL_ID = os.getenv(
    "NLI_VERIFIER_MODEL",
    "cross-encoder/nli-deberta-v3-small",
)
# Confidence threshold for committing a direction (P_entail must clear this)
NLI_MIN_CONF = float(os.getenv("NLI_MIN_CONF", "0.55"))
# Minimum gap between top hypothesis and second-best (avoids low-confidence
# 0.45/0.40 splits committing on noise)
NLI_MIN_MARGIN = float(os.getenv("NLI_MIN_MARGIN", "0.10"))


_lock = threading.Lock()
_model = None
_tokenizer = None


def is_enabled() -> bool:
    return os.getenv("ENABLE_NLI_VERIFIER", "0") == "1"


def _get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("nli_verifier_loading", model=MODEL_ID, device=device)
            tok = AutoTokenizer.from_pretrained(MODEL_ID)
            mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            mdl.eval()
            mdl.to(device)
            _model, _tokenizer = mdl, tok
            log.info(
                "nli_verifier_ready",
                model=MODEL_ID,
                device=device,
                labels=getattr(mdl.config, "id2label", None),
            )
        except Exception as e:
            log.warning("nli_verifier_load_failed", err=str(e))
            _model, _tokenizer = None, None
        return _model, _tokenizer


# Question → claim form. The question is reframed as a declarative claim;
# we then read BOTH entailment and contradiction probabilities from the NLI
# model (single forward pass per market). p_yes = P_entail; p_no = P_contradict.
# This is more robust than constructing two hypotheses and comparing entail
# probs — cross-encoder NLI models return well-calibrated contradiction
# scores out of the box.
_WILL_PREFIX = re.compile(r"^\s*will\s+", re.I)
_QMARK = re.compile(r"\?+\s*$")
# Empirically, the NLI model is highly sensitive to verb conjugation:
# "Powell resign as Fed Chair" → entail=0.001 (model can't parse).
# "Powell resigns as Fed Chair" → entail=0.001 (still bad — needs aux).
# "Powell has resigned"        → entail=0.96 (works!)
# Best transform: "Will X verb Y" → "X has verbed Y" or "X is verb-ed Y".
# We try a simple past-perfect conjugation that works for most common verbs.
_BARE_VERB_TO_PAST_PARTICIPLE = {
    "win": "won",
    "lose": "lost",
    "beat": "beaten",
    "be": "been",
    "do": "done",
    "go": "gone",
    "have": "had",
    "make": "made",
    "take": "taken",
    "give": "given",
    "see": "seen",
    "come": "come",
    "leave": "left",
    "say": "said",
    "find": "found",
    "hold": "held",
    "fall": "fallen",
    "rise": "risen",
    "hit": "hit",
    "exceed": "exceeded",
    "reach": "reached",
    "pass": "passed",
    "fail": "failed",
    "resign": "resigned",
    "step-down": "stepped down",
    "approve": "approved",
    "default": "defaulted",
    "ban": "banned",
    "veto": "vetoed",
    "indict": "indicted",
    "convict": "convicted",
}


def _to_claim(question: str) -> str:
    """Convert "Will <subject> <verb> <rest>?" → "<subject> has <verb-ed> <rest>".

    Polymarket questions are dominantly future-tense yes/no questions. NLI
    models score these terribly when left bare. Past-perfect conjugation
    ("X has won", "Y has resigned") gives the best zero-shot entailment
    behaviour we measured. Falls back gracefully when the pattern doesn't
    match (the question is just stripped of "?").
    """
    q = (question or "").strip()
    q = _QMARK.sub("", q)
    m = _WILL_PREFIX.match(q)
    if not m:
        return q
    rest = q[m.end():].strip()
    # Find first verb-like token. We split on whitespace and look up a
    # past-participle form in the lookup table; if we find one, we
    # rebuild as "<everything before verb> has <verb-pp> <everything after>".
    tokens = rest.split()
    for i, tok_ in enumerate(tokens):
        bare = tok_.lower().strip(",.")
        pp = _BARE_VERB_TO_PAST_PARTICIPLE.get(bare)
        if pp is None:
            # Also try -e suffixed verb (e.g. "exceede" not in lookup but "exceed" is)
            if bare.endswith("e"):
                pp = _BARE_VERB_TO_PAST_PARTICIPLE.get(bare[:-1])
        if pp is not None:
            before = " ".join(tokens[:i])
            after = " ".join(tokens[i+1:])
            piece = f"{before} has {pp} {after}".strip()
            return re.sub(r"\s+", " ", piece)
    # No known verb found — return bare statement
    return rest


@dataclass
class NLIResult:
    direction: str  # "yes" | "no" | "unknown"
    confidence: float  # max(p_entail, p_contradict)
    p_entail_yes: float  # P(news entails claim) → "yes"
    p_entail_no: float   # P(news contradicts claim) → "no"
    margin: float  # |p_entail - p_contradict|
    elapsed_ms: float
    yes_hypothesis: str  # the claim used
    no_hypothesis: str   # alias, kept for backwards-compat


def _label_indices(model) -> dict[str, int]:
    """Map the model's id2label → name for entailment / contradiction / neutral."""
    id2label = getattr(model.config, "id2label", {}) or {}
    out = {}
    for i, name in id2label.items():
        n = str(name).lower().replace("_", "")
        if n.startswith("entail"):
            out["entailment"] = int(i)
        elif n.startswith("contradict"):
            out["contradiction"] = int(i)
        elif n.startswith("neutral") or n == "":
            out["neutral"] = int(i)
    # cross-encoder/nli-deberta-v3-small uses ["contradiction", "entailment", "neutral"]
    return out


def verify(headline: str, question: str, body: str | None = None) -> Optional[NLIResult]:
    """Run zero-shot NLI on (news premise, market claim).

    Single forward pass: read entailment AND contradiction probs.
      p_yes  = P(news entails the claim)
      p_no   = P(news contradicts the claim)
    The remaining probability mass is on "neutral" (claim is unrelated).

    Returns None if the model isn't loaded or input is empty.
    """
    if not headline or not question:
        return None
    model, tok = _get_model()
    if model is None or tok is None:
        return None
    import torch
    t0 = time.time()
    premise = headline
    if body:
        premise = f"{headline}. {body[:500]}"
    claim = _to_claim(question)

    enc = tok(
        [premise], [claim],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
    label_idx = _label_indices(model)
    ent_idx = label_idx.get("entailment", 1)
    con_idx = label_idx.get("contradiction", 0)
    p_entail = float(probs[ent_idx])
    p_contradict = float(probs[con_idx])

    margin = abs(p_entail - p_contradict)
    direction = "unknown"
    confidence = max(p_entail, p_contradict)
    if margin >= NLI_MIN_MARGIN and confidence >= NLI_MIN_CONF:
        direction = "yes" if p_entail > p_contradict else "no"

    elapsed_ms = (time.time() - t0) * 1000.0
    return NLIResult(
        direction=direction,
        confidence=confidence,
        p_entail_yes=p_entail,
        p_entail_no=p_contradict,
        margin=margin,
        elapsed_ms=elapsed_ms,
        yes_hypothesis=claim,
        no_hypothesis=claim,  # same claim; "no" comes from contradiction prob
    )
