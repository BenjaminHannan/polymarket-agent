"""agent-next/polymarket-paper-trader-compatible fee model
(pmwhybetter.md Problem-4 #2).

Direct implementation of the `agent-next/polymarket-paper-trader`
(MCP-server) fee formula. The doc flags this as a "honest paper
broker" upgrade; our existing `polyagent/risk/fees.py` uses a per-
category bps rate but doesn't have the Polymarket-specific
`min(p, 1−p)` correction:

  fee_usd = bps_rate × min(p, 1−p) × shares

This matters because Polymarket's actual fee scales with the *less
likely* side of the trade — buying a 0.05 YES pays fees on the
0.05 leg, not 1.00. Our existing model over-charges on longshots
and under-charges on near-50/50 markets.

Reference
---------
- `agent-next/polymarket-paper-trader` GitHub
- Polymarket docs: https://docs.polymarket.com/#fees

API
---
- `agent_next_fee(price, side, shares, category)` → FillFees
  Same shape as `polyagent.risk.fees.compute_fees` for drop-in
  replacement.
- `effective_taker_fee_bps(price)` → float
  Side-agnostic effective rate in bps, useful for sanity-checking
  the existing fee model.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# Polymarket fee schedule (verify against docs.polymarket.com periodically).
# These are the *nominal* bps rates applied to min(p, 1-p) * shares.
DEFAULT_FEE_RATES_BPS = {
    "crypto": 180.0,         # 1.80%
    "sports": 75.0,           # 0.75%
    "sports_global": 75.0,
    "politics": 100.0,        # 1.00%
    "election": 100.0,
    "econ": 100.0,
    "entertainment": 100.0,
    "other": 100.0,
}

DEFAULT_MAKER_REBATE_SHARE = 0.22   # 22% of taker fees returned to maker


@dataclass
class FillFees:
    """Drop-in compatible with polyagent.risk.fees.FillFees."""
    taker_fee_usd: float
    maker_rebate_usd: float
    effective_fee_usd: float       # taker − rebate for the (maker,taker) round-trip
    rate_bps_used: float           # the *effective* bps applied to notional


def effective_taker_fee_bps(
    price: float, category: str = "other",
) -> float:
    """The Polymarket fee rate, expressed in bps of *notional*.

    Because fee = bps × min(p, 1−p) × shares = bps × min(p, 1−p) × N/p
                 = bps × min(p, 1−p) / p × N,
    the bps-of-notional rate is `bps_rate × min(p, 1−p) / p`.

    For p=0.5 this equals the nominal bps. For p=0.05 (longshot)
    it's bps × 0.05/0.05 = bps. For p=0.95 it's bps × 0.05/0.95
    ≈ 5% of nominal — buying a heavy favourite pays much less fee
    per dollar than the same notional in a near-coin-flip.
    """
    if price <= 0 or price >= 1:
        return 0.0
    nominal = DEFAULT_FEE_RATES_BPS.get(
        (category or "other").lower(), DEFAULT_FEE_RATES_BPS["other"],
    )
    min_side = min(price, 1 - price)
    return float(nominal * min_side / price)


def agent_next_fee(
    price: float,
    side: str,
    shares: float,
    category: str = "other",
    *,
    is_maker: bool = False,
    rebate_share: float = DEFAULT_MAKER_REBATE_SHARE,
) -> FillFees:
    """Exact Polymarket-style fee per the agent-next/polymarket-paper-trader
    formula.

    Args:
        price: fill price in (0, 1).
        side: "BUY" or "SELL" (informational; the formula is side-symmetric).
        shares: filled size in shares.
        category: market category for the bps lookup.
        is_maker: if True, returns negative effective_fee_usd (we
            collected the rebate). If False, returns positive cost.

    The formula: fee = bps × min(p, 1−p) × shares.
    """
    if price <= 0 or price >= 1 or shares <= 0:
        return FillFees(
            taker_fee_usd=0.0, maker_rebate_usd=0.0,
            effective_fee_usd=0.0, rate_bps_used=0.0,
        )
    nominal_bps = DEFAULT_FEE_RATES_BPS.get(
        (category or "other").lower(), DEFAULT_FEE_RATES_BPS["other"],
    )
    min_side = min(price, 1 - price)
    taker_fee_usd = (nominal_bps / 10_000.0) * min_side * float(shares)
    maker_rebate_usd = taker_fee_usd * float(rebate_share)
    if is_maker:
        # Maker collects the rebate, pays no taker fee.
        effective = -maker_rebate_usd
    else:
        # Taker pays the full fee; rebate goes to the counterparty.
        effective = taker_fee_usd
    return FillFees(
        taker_fee_usd=float(taker_fee_usd),
        maker_rebate_usd=float(maker_rebate_usd),
        effective_fee_usd=float(effective),
        rate_bps_used=float(nominal_bps),
    )


def reconcile_with_existing_fee_model(
    price: float, side: str, shares: float, category: str,
) -> dict:
    """Side-by-side comparison: agent-next-formula vs the existing
    `polyagent.risk.fees.compute_fees`. Useful as a one-off sanity
    check when migrating."""
    new = agent_next_fee(price, side, shares, category)
    try:
        from polyagent.risk.fees import compute_fees
        notional = price * shares
        old = compute_fees(notional=notional, category=category, is_maker=False)
        return {
            "agent_next_taker_fee_usd": new.taker_fee_usd,
            "existing_taker_fee_usd": old.taker_fee_paid,
            "delta_usd": new.taker_fee_usd - old.taker_fee_paid,
            "agent_next_effective_bps_of_notional":
                effective_taker_fee_bps(price, category),
        }
    except Exception:
        return {
            "agent_next_taker_fee_usd": new.taker_fee_usd,
            "agent_next_effective_bps_of_notional":
                effective_taker_fee_bps(price, category),
        }
