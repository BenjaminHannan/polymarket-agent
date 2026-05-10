"""Tests for the data-api.polymarket.com trade ingest.

The HTTP fetcher is exercised via integration with the running
data-api in a few env-flagged tests; the offline-safe tests cover
table schema, deduplication, and the smart-money aggregation.
"""
from __future__ import annotations

import sqlite3
import time

from polyagent.data.polymarket_trades import (
    ensure_table,
    insert_trades,
    top_volume_wallets,
)


def _t(**overrides) -> dict:
    """Build a sample trade dict matching the data-api response shape."""
    base = {
        "transactionHash": "0xabc123",
        "proxyWallet": "0xfeedbeef",
        "asset": "12345",
        "conditionId": "0xcond1",
        "side": "BUY",
        "outcome": "Yes",
        "outcomeIndex": 0,
        "size": 100.0,
        "price": 0.50,
        "timestamp": 1000.0,
        "title": "Will it rain?",
        "slug": "will-it-rain",
        "eventSlug": "weather",
        "name": "Alice",
        "pseudonym": "Quick-Fox",
    }
    base.update(overrides)
    return base


def test_ensure_table_creates_schema():
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    cols = [r[1] for r in c.execute("PRAGMA table_info(historical_trades)")]
    for required in ("tx_hash", "wallet", "asset", "condition_id", "side",
                     "size", "price", "ts"):
        assert required in cols


def test_insert_trades_basic():
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    n_ins, n_dup = insert_trades(c, [_t(), _t(price=0.51)])
    assert n_ins == 2
    assert n_dup == 0


def test_insert_trades_dedupe_by_pk():
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    insert_trades(c, [_t()])
    n_ins, n_dup = insert_trades(c, [_t()])  # same trade again
    assert n_ins == 0
    assert n_dup == 1


def test_insert_trades_handles_missing_optional_fields():
    """The data-api response sometimes omits name/bio/etc. — should
    insert cleanly without crashing."""
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    minimal = {
        "transactionHash": "0xtest",
        "proxyWallet": "0xtestwallet",
        "asset": "999",
        "conditionId": "0xcond2",
        "side": "SELL",
        "size": 50.0,
        "price": 0.30,
        "timestamp": 2000.0,
    }
    n_ins, _ = insert_trades(c, [minimal])
    assert n_ins == 1


def test_wallet_addresses_lowercased():
    """Avoid duplicates on case-mismatch by normalizing wallets."""
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    insert_trades(c, [_t(proxyWallet="0xABCDEF"), _t(proxyWallet="0xabcdef", size=200)])
    rows = c.execute("SELECT DISTINCT wallet FROM historical_trades").fetchall()
    # Both should land under the lowercase wallet
    assert all(r[0] == "0xabcdef" for r in rows)


def test_top_volume_wallets_returns_descending():
    c = sqlite3.connect(":memory:")
    ensure_table(c)
    now = time.time()
    # Three wallets with different total volumes over the last 5 days
    insert_trades(c, [
        _t(transactionHash="0xa", proxyWallet="0xwhale", size=1000, price=0.50, timestamp=now - 100),
        _t(transactionHash="0xb", proxyWallet="0xwhale", size=500, price=0.50, timestamp=now - 200),
        _t(transactionHash="0xc", proxyWallet="0xmid", size=300, price=0.50, timestamp=now - 300),
        _t(transactionHash="0xd", proxyWallet="0xsmall", size=50, price=0.50, timestamp=now - 400),
    ])
    # Hack: top_volume_wallets reads from a path; build a temp file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        c2 = sqlite3.connect(path)
        ensure_table(c2)
        insert_trades(c2, [
            _t(transactionHash="0xa", proxyWallet="0xwhale", size=1000, price=0.50, timestamp=now - 100),
            _t(transactionHash="0xb", proxyWallet="0xwhale", size=500, price=0.50, timestamp=now - 200),
            _t(transactionHash="0xc", proxyWallet="0xmid", size=300, price=0.50, timestamp=now - 300),
            _t(transactionHash="0xd", proxyWallet="0xsmall", size=50, price=0.50, timestamp=now - 400),
        ])
        c2.close()
        top = top_volume_wallets(path, days=30, top_k=10, min_usdc_volume=100.0)
        assert len(top) == 2  # whale + mid pass min_usdc_volume; small filtered out
        assert top[0]["wallet"] == "0xwhale"
        assert top[0]["volume"] == 750.0  # (1000 + 500) × 0.50
        assert top[0]["n_trades"] == 2
        assert top[1]["wallet"] == "0xmid"
    finally:
        os.unlink(path)


def test_top_volume_wallets_respects_min_usdc_filter():
    import tempfile, os
    now = time.time()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        c = sqlite3.connect(path)
        ensure_table(c)
        # Below-threshold wallet — should be filtered out
        insert_trades(c, [_t(proxyWallet="0xtiny", size=10, price=0.50, timestamp=now)])
        c.close()
        top = top_volume_wallets(path, days=30, min_usdc_volume=100.0)
        assert top == []
    finally:
        os.unlink(path)
