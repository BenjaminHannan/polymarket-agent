"""Map a market question to a single category label.

Reuses the same regex set as features.py (for consistency between training
features and the per-category combiner key). Picks the category with the
most keyword hits; ties broken by priority order. Returns "other" if no
category matches.
"""

from __future__ import annotations

from polyagent.models.features import _CATEGORY_RES


def categorize(question: str) -> str:
    if not question:
        return "other"
    counts: dict[str, int] = {}
    for name, rx in _CATEGORY_RES:
        n = len(rx.findall(question))
        if n > 0:
            counts[name] = n
    if not counts:
        return "other"
    # Sort by count desc, then by appearance order in _CATEGORY_RES (priority).
    order = {name: i for i, (name, _) in enumerate(_CATEGORY_RES)}
    return max(counts.items(), key=lambda kv: (kv[1], -order.get(kv[0], 99)))[0]
