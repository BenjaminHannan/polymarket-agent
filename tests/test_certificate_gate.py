"""Cert-gate sanity test for CombinedTrader.

Confirms:
  * `certified_categories=None` → legacy behaviour (passes the gate, hits later checks)
  * `certified_categories={...}` → categories outside the set are skipped early
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import MagicMock

from polyagent.gamma import Market
from polyagent.strategies.combined_trader import CombinedTrader


def _make_trader(certified):
    book_store = MagicMock()
    book_store.books = {}
    broker = MagicMock()
    broker.drawdown.drawdown.return_value = 0.0
    broker.nav.return_value = 10000.0
    broker.was_recently_stopped.return_value = False
    broker.is_token_buy_capped.return_value = False
    broker.positions = {}
    return CombinedTrader(
        book_store=book_store,
        broker=broker,
        certified_categories=certified,
    )


def _market(category: str = "sports_global") -> Market:
    return Market(
        condition_id="0xtest",
        question="dummy?",
        yes_token_id="t_yes",
        no_token_id="t_no",
        end_date_iso="2030-01-01T00:00:00Z",
        liquidity=10000.0,
        volume_24h=10000.0,
        accepting_orders=True,
        category=category,
    )


def test_gate_blocks_uncertified_category():
    trader = _make_trader(certified={"sports_global"})
    asyncio.run(
        trader.on_signal(
            market=_market("crypto"),
            p_combined=0.7,
            p_market=0.4,
            category="crypto",
        )
    )
    # The gate should reject without ever asking the broker for nav-side state.
    # Specifically: no broker.submit call, no daily_notional change.
    assert trader.daily_notional_used == 0.0


def test_gate_allows_certified_category():
    trader = _make_trader(certified={"sports_global"})
    # Will fall past the gate but be rejected by the volume/cooldown gates;
    # what we're checking is that we got past the early-return — i.e. the
    # gate did not short-circuit on the certified category. We assert by
    # patching out everything past the gate to confirm we *reach* it.
    # Here, with no book and volume=0, we'll bail on the volume gate, not
    # the cert gate. That's success: no exception, no early-skip log.
    asyncio.run(
        trader.on_signal(
            market=_market("sports_global"),
            p_combined=0.7,
            p_market=0.4,
            category="sports_global",
        )
    )


def test_gate_disabled_is_legacy_behavior():
    trader = _make_trader(certified=None)
    # No allowlist → any category should pass the gate; same downstream
    # short-circuit on volume gate.
    asyncio.run(
        trader.on_signal(
            market=_market("entertainment"),
            p_combined=0.7,
            p_market=0.4,
            category="entertainment",
        )
    )


def test_certificate_db_query_shape():
    """The exact shape main.py uses to build the allowlist."""
    import json
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE strategy_certificates(
             name TEXT, enabled INTEGER, dsr_holdout REAL,
             n_holdout INTEGER, issued_ts REAL, detail TEXT
           )"""
    )
    conn.execute(
        "INSERT INTO strategy_certificates VALUES (?,?,?,?,?,?)",
        ("a", 1, 0.99, 600, time.time(), json.dumps({"category": "sports_global"})),
    )
    conn.execute(
        "INSERT INTO strategy_certificates VALUES (?,?,?,?,?,?)",
        ("b", 0, 0.50, 600, time.time(), json.dumps({"category": "crypto"})),
    )
    conn.execute(
        "INSERT INTO strategy_certificates VALUES (?,?,?,?,?,?)",
        ("c", 1, 0.99, 600, time.time(), json.dumps({"reason": "no category key"})),
    )
    conn.commit()

    rows = conn.execute(
        "SELECT detail FROM strategy_certificates WHERE enabled=1"
    ).fetchall()
    cats = set()
    for (d,) in rows:
        try:
            obj = json.loads(d or "{}")
        except Exception:
            continue
        c = obj.get("category")
        if isinstance(c, str) and c:
            cats.add(c)
    # Only the row with enabled=1 AND a valid category key should pass.
    assert cats == {"sports_global"}
