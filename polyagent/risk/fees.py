"""Per-category fee + maker rebate accounting (pmwhy.md A7, B1).

Polymarket runs a tiered fee curve by category, with the headline rates
documented in their public help articles and changelog:

  Crypto:    1.80% peak (most volatile category)
  Sports:    0.75%
  Politics:  1.00%
  Entertainment / AI / Other: 1.00% (default)

Makers pay zero fees and receive 20–25% of the realized taker fees as
USDC rebates, accrued daily per market. We default to 22% mid-band.

Modeling choices for paper-mode:

  - Fees are charged on the *taker* side of each fill at order entry
    (taker pays, maker doesn't).
  - The maker's rebate is credited as a positive ledger entry — paper
    only; in real-money mode this would be a daily USDC payout.
  - We do NOT model the dynamic "peak" rate adjustments. The
    flat-per-category model overstates fees during low-vol regimes and
    understates them during election spikes; for paper P&L the order
    of magnitude is what matters.
  - This module is pure (no I/O); persistence happens in PaperBroker.

Override per-category rates via env vars FEE_RATE_<CATEGORY>
(e.g. FEE_RATE_SPORTS_GLOBAL=0.0075 for 75 bps).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Default per-category taker fee rates (decimal fractions, not bps).
# These are intentionally conservative; tune via env vars when
# Polymarket publishes a curve update.
_DEFAULT_TAKER_FEE: dict[str, float] = {
    "crypto":          0.018,   # 1.80%
    "sports_us":       0.0075,  # 0.75%
    "sports_global":   0.0075,
    "politics_us":     0.010,   # 1.00%
    "politics_global": 0.010,
    "geopolitics":     0.010,
    "weather":         0.010,
    "ai":              0.010,
    "entertainment":   0.010,
    "economy":         0.010,
    "other":           0.010,
}

# Maker rebate as a fraction of the realized taker fee on the same fill.
# Polymarket docs: 20–25% per market, USDC daily payouts. We default
# mid-band; override via env MAKER_REBATE_SHARE.
DEFAULT_REBATE_SHARE = float(os.getenv("MAKER_REBATE_SHARE", "0.22"))


def _category_rate(category: str | None) -> float:
    """Resolve the taker fee rate for a category, with env-var override.

    Env override pattern: FEE_RATE_<CATEGORY_UPPER> e.g. FEE_RATE_CRYPTO=0.018.
    Falls through to the "other" default when nothing matches.
    """
    cat = (category or "other").lower().strip()
    env_key = f"FEE_RATE_{cat.upper().replace(' ', '_')}"
    env_val = os.getenv(env_key)
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return _DEFAULT_TAKER_FEE.get(cat, _DEFAULT_TAKER_FEE["other"])


@dataclass
class FillFees:
    """Result of fee accounting for a single fill."""
    taker_fee_rate: float       # decimal fraction
    taker_fee_paid: float       # in $ (notional × rate, or 0 for maker)
    maker_rebate_rate: float    # decimal fraction (% of taker fee)
    maker_rebate_credited: float  # in $ (credited to maker; 0 for taker)
    is_maker: bool
    notional: float
    category: str | None


def compute_fees(
    *,
    notional: float,
    category: str | None,
    is_maker: bool,
    rebate_share: float = DEFAULT_REBATE_SHARE,
) -> FillFees:
    """Compute fee and (if maker) rebate for one fill.

    Convention:
      - Takers PAY: `taker_fee_paid = notional × taker_fee_rate`
      - Makers RECEIVE: `maker_rebate_credited = notional × taker_fee_rate × rebate_share`
        (the implicit modelled assumption is that *somewhere* a counterparty
        taker paid the fee; the maker rebate is a portion of that fee
        flowing back. In paper-mode we credit the rebate without debiting
        the counterparty.)
      - The flow is mutually exclusive: one fill is either a taker fill or
        a maker fill.
    """
    rate = _category_rate(category)
    if is_maker:
        return FillFees(
            taker_fee_rate=rate,
            taker_fee_paid=0.0,
            maker_rebate_rate=rebate_share,
            maker_rebate_credited=float(notional * rate * rebate_share),
            is_maker=True,
            notional=float(notional),
            category=category,
        )
    return FillFees(
        taker_fee_rate=rate,
        taker_fee_paid=float(notional * rate),
        maker_rebate_rate=rebate_share,
        maker_rebate_credited=0.0,
        is_maker=False,
        notional=float(notional),
        category=category,
    )


def cancel_latency_penalty(
    fill_price: float,
    side: str,
    realized_vol_per_sec: float | None,
    *,
    block_sec: float = 2.0,
) -> float:
    """Adverse drift on a resting limit during the cancel-latency window.

    Polymarket runs on Polygon (~2 s block time). A passive limit cannot
    dodge an adverse mid-tick faster than the next block, so on every
    maker fill we discount the recorded price by σ × √block_sec to
    reflect the expected adverse drift the limit ate before getting
    canceled or filled.

    For BUY (we paid `fill_price`): effective price = fill_price + drift
      (we paid more than the post price effectively because we held
      through adverse motion before getting hit).
    For SELL: effective price = fill_price − drift.

    Returns the EFFECTIVE FILL PRICE adjusted for the penalty; do not
    re-apply elsewhere.
    """
    sigma = realized_vol_per_sec if (realized_vol_per_sec and realized_vol_per_sec > 0) else 0.005
    import math
    drift = sigma * math.sqrt(block_sec)
    if side == "BUY":
        return min(1.0, fill_price + drift)
    else:
        return max(0.0, fill_price - drift)
