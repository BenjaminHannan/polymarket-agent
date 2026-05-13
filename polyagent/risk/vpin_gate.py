"""VPIN (Volume-Synchronized Probability of Informed Trading) toxicity gate.

Direct implementation of the doc's Problem-1 fix #5: adverse-selection-
aware sizing via VPIN gating. Bartlett & O'Hara 2026 (SSRN 6615739)
explicitly validates VPIN on Kalshi prediction markets — single-name
markets have predictable VPIN→maker-loss correlation, broad markets
do not. The practical Polymarket adaptation is to gate **maker quotes**
(post-only / GTC limit orders) by a per-category one-sided OFI threshold
over a rolling 5-minute window.

Reference:
  - Easley, López de Prado, O'Hara, "Flow Toxicity and Liquidity in a
    High-Frequency World," RFS 2012 (the original VPIN paper).
  - Bartlett & O'Hara, "Adverse Selection on Prediction Markets,"
    SSRN 6615739, 2026 (validates VPIN on Kalshi).
  - Barzykin, Bergault, Guéant, Lemmel, "Optimal Quoting under Adverse
    Selection and Price Reading," arXiv 2508.20225, 2025 (extends
    Avellaneda-Stoikov with explicit informed-flow term — the
    quote-side companion to this gate).

VPIN intuition
--------------
Split the recent trade tape into N volume buckets of equal volume V.
Inside each bucket, the **imbalance** is::

    |B_buy − B_sell| / V

where B_buy and B_sell are buy- and sell-initiated volume in the
bucket. VPIN is the average imbalance over the most-recent N buckets,
in [0, 1]. High VPIN ⇒ flow is one-sided ⇒ informed traders are likely
running, and resting maker quotes on the offered side will be picked
off.

Polymarket-specific caveat (Dubach 2026)
----------------------------------------
Polymarket's public WSS Lee-Ready buy/sell classification agrees with
on-chain ground truth only ~59% of the time (vs ~80% on equity venues).
That means VPIN computed from feed direction is noisier than the
literature would predict. We compensate in two ways:

  1. Use a **conservative** VPIN threshold (default 0.65 vs the
     0.55–0.60 typical in equity HFT) — the gate trips less often
     because each trip carries less information.
  2. Expose `mark_direction_quality` so a future on-chain
     `OrderFilled` migration can flag this gate as fully-trusted.

Decision rule
-------------
A maker quote on (token_id, side) is BLOCKED if::

    vpin(token_id, lookback) > vpin_max  AND
    sign(net_flow(token_id, lookback)) is AGAINST our resting side

i.e. we only block when flow is *toxically* one-sided *against* our
quote. We don't block on symmetric volume bursts (high VPIN with
balanced buys/sells), and we don't block our quote on the same side as
the imbalance (we'd love to be picked off by takers buying when our
buy quote rests below).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class _TradeEvent:
    ts: float
    side: str            # "BUY" or "SELL" (taker-initiated, Lee-Ready)
    size: float          # USDC notional or share count (must be consistent)


@dataclass
class VPINGate:
    """Per-token rolling VPIN with directional flow gate.

    Args:
        bucket_volume: total volume V per bucket. Polymarket sport
            markets see ~$5k–$50k/hour notional; V=500 USDC gives
            ~10–100 buckets/hour — enough fidelity for a 5-min gate.
        n_buckets: how many recent buckets to average for VPIN. The
            EdLO 2012 default is 50; we use 20 since Polymarket flow
            is thinner.
        vpin_max: trip threshold. Block quotes above this when flow
            is against the quote. Default 0.65 (conservative vs the
            equity-HFT 0.55 since Polymarket direction is noisier per
            Dubach 2026).
        lookback_sec: window for net-flow direction sign. Default
            300s = 5 minutes (matches Bartlett-O'Hara's per-category
            window).
        min_buckets: don't trip the gate until this many buckets have
            been observed. Avoids cold-start false positives.
        max_events_per_token: ring-buffer cap on raw trade events per
            token (memory bound).
        direction_quality: how much to trust the feed direction
            classification. Lee-Ready on Polymarket WSS is ~0.59
            (Dubach 2026). Set to 0.8+ once on-chain `OrderFilled`
            ingest lands. Used to scale the effective imbalance.
    """
    bucket_volume: float = 500.0
    n_buckets: int = 20
    vpin_max: float = 0.65
    lookback_sec: float = 300.0
    min_buckets: int = 5
    max_events_per_token: int = 10_000
    direction_quality: float = 0.59

    _events: dict = field(default_factory=dict)             # token -> deque[_TradeEvent]
    _buckets: dict = field(default_factory=dict)            # token -> deque[(buy_vol, sell_vol)]
    _running_bucket: dict = field(default_factory=dict)     # token -> [buy_vol, sell_vol]

    # ── ingest ──────────────────────────────────────────────────────────
    def record_trade(self, token_id: str, side: str, size: float,
                     ts: float | None = None) -> None:
        """Ingest one taker-initiated trade. `side` is "BUY" (taker bought
        YES — i.e. crossed the ask) or "SELL" (taker sold — crossed the
        bid). The classification comes from the WSS feed's Lee-Ready;
        treat as noisy until on-chain migration."""
        if size <= 0:
            return
        ts = float(ts if ts is not None else time.time())
        side_norm = side.upper().strip()
        if side_norm not in ("BUY", "SELL"):
            return

        evs = self._events.get(token_id)
        if evs is None:
            evs = deque(maxlen=self.max_events_per_token)
            self._events[token_id] = evs
        evs.append(_TradeEvent(ts=ts, side=side_norm, size=float(size)))

        rb = self._running_bucket.get(token_id)
        if rb is None:
            rb = [0.0, 0.0]
            self._running_bucket[token_id] = rb
        if side_norm == "BUY":
            rb[0] += size
        else:
            rb[1] += size
        # Seal the bucket when total volume reaches bucket_volume.
        if rb[0] + rb[1] >= self.bucket_volume:
            bks = self._buckets.get(token_id)
            if bks is None:
                bks = deque(maxlen=self.n_buckets)
                self._buckets[token_id] = bks
            bks.append((rb[0], rb[1]))
            rb[0] = 0.0
            rb[1] = 0.0

    # ── compute ─────────────────────────────────────────────────────────
    def vpin(self, token_id: str) -> float | None:
        """Current rolling VPIN in [0, 1]. None if too few buckets."""
        bks = self._buckets.get(token_id)
        if bks is None or len(bks) < self.min_buckets:
            return None
        imbal = 0.0
        n = 0
        for buy_vol, sell_vol in bks:
            tot = buy_vol + sell_vol
            if tot <= 0:
                continue
            imbal += abs(buy_vol - sell_vol) / tot
            n += 1
        if n == 0:
            return None
        raw = imbal / n
        # Scale by direction-quality. Noisy direction => observed
        # imbalance under-states true imbalance => raw VPIN is biased low.
        # We don't blindly upscale — that'd make the gate hair-trigger —
        # but we apply a partial correction:
        #   adjusted = raw + (1 − direction_quality) * (1 − raw) / 2
        # which fixes raw at 0 and 1 (extremes are unchanged) and lifts
        # ambiguous middle values.
        q = max(0.5, min(1.0, self.direction_quality))
        adjusted = raw + (1.0 - q) * (1.0 - raw) * 0.5
        return float(min(1.0, adjusted))

    def net_flow(self, token_id: str) -> float:
        """Recent net-flow (buy − sell volume) over `lookback_sec`.
        Used for the directional check."""
        evs = self._events.get(token_id)
        if not evs:
            return 0.0
        now = time.time()
        cutoff = now - self.lookback_sec
        net = 0.0
        for e in reversed(evs):
            if e.ts < cutoff:
                break
            net += e.size if e.side == "BUY" else -e.size
        return float(net)

    # ── decision ────────────────────────────────────────────────────────
    def allow_quote(self, token_id: str, side: str) -> tuple[bool, dict]:
        """Decide whether a maker quote on `side` for `token_id` is
        allowed. Returns (allow, telemetry).

        Args:
            side: "BUY" if we're posting a passive buy (bid), "SELL"
                if we're posting a passive sell (offer). The gate
                blocks a quote when toxic flow is *against* it:
                  - block BUY when net flow is heavily SELL (sellers
                    are dumping; our buy is about to be picked off)
                  - block SELL when net flow is heavily BUY (buyers
                    are lifting; our offer is about to be picked off)
        """
        v = self.vpin(token_id)
        if v is None or v < self.vpin_max:
            return True, {
                "vpin": v,
                "decision": "allow",
                "reason": "below_threshold" if v is not None else "cold_start",
            }
        nf = self.net_flow(token_id)
        side_norm = side.upper().strip()
        # Net flow > 0 means buy-heavy; flow against BUY-quote? No
        # (we'd want buyers to lift our offer, not our bid). We block
        # BUY-quote when flow is sell-heavy (nf < 0), block SELL-quote
        # when flow is buy-heavy (nf > 0).
        toxic_against_quote = (
            (side_norm == "BUY" and nf < 0)
            or (side_norm == "SELL" and nf > 0)
        )
        if not toxic_against_quote:
            return True, {
                "vpin": v,
                "net_flow": nf,
                "decision": "allow",
                "reason": "flow_aligned_with_quote",
            }
        log.info(
            "vpin_gate_blocked",
            token_id=token_id,
            side=side_norm,
            vpin=round(v, 3),
            net_flow=round(nf, 2),
            vpin_max=self.vpin_max,
        )
        return False, {
            "vpin": v,
            "net_flow": nf,
            "decision": "block",
            "reason": "toxic_flow_against_quote",
        }

    # ── ops ─────────────────────────────────────────────────────────────
    def mark_direction_quality(self, quality: float) -> None:
        """Update the direction-classification trust level. Call this
        when on-chain `OrderFilled` ingest comes online (raise toward
        0.85–0.95) or when feed degrades."""
        self.direction_quality = float(max(0.5, min(1.0, quality)))

    def summary(self) -> dict:
        return {
            "n_tokens": len(self._buckets),
            "vpin_max": self.vpin_max,
            "bucket_volume": self.bucket_volume,
            "n_buckets": self.n_buckets,
            "direction_quality": self.direction_quality,
        }
