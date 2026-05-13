"""Tests for on-chain OrderFilled ingester scaffold."""
from __future__ import annotations

import asyncio
import os
import sqlite3

from polyagent.data.onchain_orderfilled import (
    OnchainFill, ensure_columns, insert_onchain_fills, run_onchain_ingester,
    _decode_log,
)


def test_ensure_columns_idempotent(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        "CREATE TABLE trades (tx_hash TEXT, wallet TEXT, asset TEXT, "
        "side TEXT, size REAL, price REAL, timestamp REAL)"
    )
    ensure_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    assert "direction_source" in cols
    assert "counterparty_wallet" in cols
    # Re-run to confirm idempotence.
    ensure_columns(conn)


def test_insert_onchain_fills(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        "CREATE TABLE trades (tx_hash TEXT PRIMARY KEY, wallet TEXT, asset TEXT, "
        "side TEXT, size REAL, price REAL, timestamp REAL)"
    )
    fills = [
        OnchainFill(
            tx_hash="0xabc", block_number=100, block_timestamp=1000.0,
            maker_wallet="0xMaker", taker_wallet="0xTaker",
            maker_asset_id="tokA", taker_asset_id="tokB",
            maker_amount=100.0, taker_amount=50.0, fee=0.5,
            price=0.5, side="BUY",
        ),
    ]
    n = insert_onchain_fills(conn, fills)
    assert n == 1
    row = conn.execute(
        "SELECT direction_source, counterparty_wallet FROM trades WHERE tx_hash='0xabc'"
    ).fetchone()
    assert row[0] == "onchain"
    assert row[1] == "0xMaker"


def test_decode_log_handles_garbage():
    # Malformed log (no topics) should return None, not raise.
    decoded = _decode_log({"transactionHash": "0xfoo", "blockNumber": "0x10"})
    assert decoded is None


def test_run_disabled_without_rpc():
    """No ALCHEMY_RPC_URL or POLYGON_RPC_URL ⇒ ingester is a no-op."""
    # Both fallbacks must be unset for the no-op path. Save and restore so
    # we don't clobber the developer's .env-loaded env between tests.
    saved_alchemy = os.environ.pop("ALCHEMY_RPC_URL", None)
    saved_polygon = os.environ.pop("POLYGON_RPC_URL", None)
    try:
        # Should return immediately without hanging or opening any db.
        asyncio.run(run_onchain_ingester("/dev/null", rpc_url=None))
    finally:
        if saved_alchemy is not None:
            os.environ["ALCHEMY_RPC_URL"] = saved_alchemy
        if saved_polygon is not None:
            os.environ["POLYGON_RPC_URL"] = saved_polygon
