"""Consistency-loss training scaffold (Karkare/Paleka arXiv 2412.18544;
Outcome-RL arXiv 2505.17989).

Direct implementation of the doc's Problem-6 fix #3: turn the
arbitrage-based consistency metric from a detector into a *training
loss*. Karkare & Paleka 2024 showed that a forecaster trained with a
consistency-loss term has held-out Brier improvements correlated with
the consistency metric itself — i.e. consistency is both a diagnostic
and a useful auxiliary objective.

The existing `polyagent/signals/consistency_check.py` is a detector
only (NegRisk-aware sum-to-1 check). This module exposes the same
constraint as a **differentiable loss** suitable for use during model
fine-tuning.

Three loss terms
----------------
1. **MEE-sum consistency**: for a NegRisk-eligible cluster
   {q_1, …, q_n}, loss = (Σ p(q_i) − 1)².

2. **Monotonicity consistency**: for a pair (A ⊆ B), loss =
   max(0, p(A) − p(B))² — hinge on the violation direction.

3. **Negation invariance**: for a question q with the YES token,
   p(q) + p(¬q) should equal 1; loss = (p(q) + p(¬q) − 1)².

All three are squared L2 hinges on standard constraint violations;
each can be added as an auxiliary term to the primary cross-entropy
loss with a per-term weight.

Usage
-----
This module is **framework-agnostic** — it returns Python floats /
numpy arrays. To plug into a real training loop:

  - PyTorch: convert the inputs to tensors and re-derive in torch ops
    (a few-liner). Keep the formulas the same.
  - Outcome-RL specifically: the consistency loss goes alongside the
    outcome-prediction reward in the RL update.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class ConsistencyLossBundle:
    """Bundle of the three consistency loss terms; downstream code
    combines them with primary cross-entropy via per-term weights."""
    mee_sum_loss: float
    monotonicity_loss: float
    negation_invariance_loss: float

    def total(
        self,
        w_mee: float = 1.0,
        w_mono: float = 1.0,
        w_neg: float = 1.0,
    ) -> float:
        return (
            w_mee * self.mee_sum_loss
            + w_mono * self.monotonicity_loss
            + w_neg * self.negation_invariance_loss
        )

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def mee_sum_loss(probs_per_cluster: list[float]) -> float:
    """L2 hinge on (Σp − 1) for a mutually-exclusive exhaustive
    cluster. Zero when constraint is satisfied."""
    if not probs_per_cluster:
        return 0.0
    s = sum(float(p) for p in probs_per_cluster)
    return float((s - 1.0) ** 2)


def monotonicity_loss(p_subset: float, p_superset: float) -> float:
    """Hinge: max(0, p(A) − p(B))² where A ⊆ B. Zero when satisfied."""
    diff = float(p_subset) - float(p_superset)
    return float(max(0.0, diff) ** 2)


def negation_invariance_loss(p_yes: float, p_no: float) -> float:
    """L2 hinge on (p_yes + p_no − 1). Zero when calibrated."""
    return float((float(p_yes) + float(p_no) - 1.0) ** 2)


def compute_consistency_loss(
    *,
    mee_clusters: list[list[float]] | None = None,
    monotone_pairs: list[tuple[float, float]] | None = None,
    yes_no_pairs: list[tuple[float, float]] | None = None,
) -> ConsistencyLossBundle:
    """Compute the three consistency-loss terms over batches.

    `mee_clusters`: list of clusters, each a list of P(YES) for the
                    cluster's questions.
    `monotone_pairs`: list of (p_subset, p_superset).
    `yes_no_pairs`:   list of (p_yes_token, p_no_token).
    """
    mee_loss = 0.0
    if mee_clusters:
        mee_loss = sum(mee_sum_loss(c) for c in mee_clusters) / max(1, len(mee_clusters))
    mono_loss = 0.0
    if monotone_pairs:
        mono_loss = sum(monotonicity_loss(*p) for p in monotone_pairs) / max(1, len(monotone_pairs))
    neg_loss = 0.0
    if yes_no_pairs:
        neg_loss = sum(negation_invariance_loss(*p) for p in yes_no_pairs) / max(1, len(yes_no_pairs))
    return ConsistencyLossBundle(
        mee_sum_loss=mee_loss,
        monotonicity_loss=mono_loss,
        negation_invariance_loss=neg_loss,
    )


def consistency_score(bundle: ConsistencyLossBundle) -> float:
    """Map total loss to a [0, 1] score where 1 = perfectly consistent.

    Useful as a single-number diagnostic during training:
        score = exp(−total_loss / softness)
    """
    softness = 0.1
    return float(math.exp(-bundle.total() / softness))
