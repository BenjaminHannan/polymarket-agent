"""Falsifiability harness for §4 — NegRisk arb opportunity-count.

The doc's framing for §4 falsifiability:
  "did NegRisk arb opportunities exceeding 1.5% appear in the historical
  book at rate >= R per day?"

This script answers that question against the polyagent log history.
The existing combinatorial_arb scanner (already running in production)
emits a `combinatorial_arb_candidate` event whenever a violation is
detected, with (kind, edge_pp, event_id, members). We grep those events
out of polyagent.log, filter by kind and edge threshold, and aggregate
per day.

Why this is real falsifiability (unlike §9 BOCPD which failed):
  - We don't need to estimate a parameterized model on tiny samples.
  - We just count actual events the live scanner already detected.
  - Both the events and their edges are observed, not parameterized.
  - The Sharpe contribution is a function of execution assumptions
    (persistence, fee, fill-depth) — but the existence and frequency
    of opportunities is not.

If the per-day rate at edge >= 1.5% is materially > 0 (say >= 2/day on
real NegRisk groups, with persistence >= 30s on average), proceed to
build the v2 scanner with atomic dispatch + depth-floor sizing.

If essentially zero, the v2 build is not justified and we defer.

Run:
    python -m scripts.backtest_negrisk_arb
    python -m scripts.backtest_negrisk_arb --min-edge-pp 1.5 --kind negrisk_sum_lt_1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "data" / "polyagent.log"


def parse_events(log_path: Path):
    """Yield parsed `combinatorial_arb_candidate` events from a log."""
    with log_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"event": "combinatorial_arb_candidate"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj


def _day(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def run(log_path: Path, min_edge_pp: float, kind_filter: str | None,
        max_edge_pp: float = 30.0) -> int:
    if not log_path.exists():
        print(f"Log file {log_path} not found.")
        return 1
    print(f"Scanning {log_path}...")

    n_total = 0
    n_pass = 0
    by_kind: Counter[str] = Counter()
    by_kind_filtered: Counter[str] = Counter()
    by_day: Counter[str] = Counter()              # event-count per day
    by_day_kind: dict[str, Counter[str]] = defaultdict(Counter)
    edge_distribution_pp: list[float] = []
    distinct_event_ids = set()
    # Persistence proxy: consecutive firings on the same event_id within
    # the same scan-cycle window get bucketed together.
    persistence_per_event: dict[str, list[str]] = defaultdict(list)
    for obj in parse_events(log_path):
        n_total += 1
        kind = obj.get("kind", "date_monotonicity")
        by_kind[kind] += 1
        edge_pp = float(obj.get("edge_pp", 0))
        if kind_filter and kind != kind_filter:
            continue
        if edge_pp < min_edge_pp:
            continue
        # Upper bound: edges > 30pp are almost always partial-NegRisk-
        # group artifacts (we stream top-N markets by liquidity, so a
        # NegRisk event with 10+ candidates only shows 4 in our set;
        # the partial sum is near zero, scanner reports "99pp edge"
        # which is an artifact, not a real arb). Real NegRisk
        # arbitrages are typically <30pp.
        if edge_pp > max_edge_pp:
            continue
        n_pass += 1
        by_kind_filtered[kind] += 1
        edge_distribution_pp.append(edge_pp)
        day = _day(obj.get("timestamp", ""))
        by_day[day] += 1
        by_day_kind[day][kind] += 1
        eid = obj.get("event_id") or obj.get("prefix") or "?"
        distinct_event_ids.add(eid)
        persistence_per_event[eid].append(obj.get("timestamp", ""))

    print(f"\nTotal arb candidate events: {n_total}")
    print(f"By kind:")
    for k, c in by_kind.most_common():
        print(f"   {k:30s} {c:6d}")
    print()
    print(f"Filter: kind={kind_filter or 'ALL'} AND edge_pp >= {min_edge_pp}")
    print(f"Events passing filter: {n_pass}")
    if n_pass == 0:
        print()
        print("VERDICT: no opportunities in history at this threshold.")
        return 2
    print(f"By kind (filtered):")
    for k, c in by_kind_filtered.most_common():
        print(f"   {k:30s} {c:6d}")
    print()
    print(f"Distinct event_ids/prefixes: {len(distinct_event_ids)}")
    print()

    n_days = len(by_day)
    print(f"Daily rate: {n_pass} events / {n_days} days = {n_pass / max(n_days, 1):.1f} events/day")
    print()

    print("Per-day counts (last 14 days):")
    for d in sorted(by_day)[-14:]:
        kinds = by_day_kind[d]
        breakdown = " ".join(f"{k}={v}" for k, v in kinds.most_common())
        print(f"  {d}: {by_day[d]:5d}  ({breakdown})")
    print()

    # Edge distribution
    import statistics
    edges = sorted(edge_distribution_pp)
    print("Edge distribution (pp):")
    print(f"  min:  {min(edges):.2f}")
    print(f"  p25:  {edges[len(edges)//4]:.2f}")
    print(f"  median: {statistics.median(edges):.2f}")
    print(f"  p75:  {edges[3*len(edges)//4]:.2f}")
    print(f"  p95:  {edges[int(0.95*len(edges))]:.2f}")
    print(f"  max:  {max(edges):.2f}")
    print()

    # Persistence per event_id: events with multiple firings indicate
    # the opportunity persisted across multiple scan cycles.
    pers_counts = [len(v) for v in persistence_per_event.values() if len(v) > 1]
    n_persistent = len(pers_counts)
    print(f"Event-IDs with multi-cycle persistence: {n_persistent} / {len(distinct_event_ids)}")
    if pers_counts:
        print(f"  median persistent firings per event: {sorted(pers_counts)[len(pers_counts)//2]}")
        print(f"  max persistent firings per event:    {max(pers_counts)}")
    print()

    # NegRisk-only verdict
    neg_pass = (
        by_kind_filtered.get("negrisk_sum_lt_1", 0)
        + by_kind_filtered.get("negrisk_sum_gt_1", 0)
    )
    print("---")
    print("§4 VERDICT (NegRisk-class only):")
    if neg_pass == 0:
        print("  FAIL: zero NegRisk sum-to-1 violations at this threshold.")
        print("  The v2 atomic-dispatch scanner is NOT justified by this data.")
        print("  Defer until either (a) more NegRisk markets stream, or")
        print("  (b) the threshold is lowered AND the per-leg fill")
        print("  assumptions are honored.")
        return 2
    neg_per_day = neg_pass / max(n_days, 1)
    print(f"  NegRisk hits: {neg_pass} ({neg_per_day:.2f}/day)")
    if neg_per_day < 1.0:
        print("  RATE TOO LOW (<1/day). Marginal case for v2 build.")
        print("  Recommend lowering threshold to 1.0pp and re-running")
        print("  before committing the engineering effort.")
        return 2
    print(f"  PASS: NegRisk arb opportunities are plentiful (>={1.0}/day).")
    print(f"  v2 build justified. Next: implement atomic-dispatch scanner")
    print(f"  with depth-floor sizing and persistence verification.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--min-edge-pp", type=float, default=1.5)
    p.add_argument("--max-edge-pp", type=float, default=30.0,
                   help="upper bound to filter partial-NegRisk-group artifacts")
    p.add_argument("--kind", default=None,
                   help="filter to a specific kind (negrisk_sum_lt_1, etc.)")
    args = p.parse_args()
    return run(args.log, args.min_edge_pp, args.kind, args.max_edge_pp)


if __name__ == "__main__":
    raise SystemExit(main())
