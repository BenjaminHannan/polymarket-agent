"""Tests for OrderFilled log decoding (web3-free path)."""
from __future__ import annotations

from polyagent.data.onchain_orderfilled import (
    _decode_order_filled, _hex_to_int, _hex_to_address, _split_data,
    insert_onchain_fills,
)
import sqlite3


def test_hex_to_int_basic():
    assert _hex_to_int("0x0") == 0
    assert _hex_to_int("0xff") == 255
    assert _hex_to_int("0x10") == 16


def test_hex_to_address_left_pads_to_40():
    # 64-hex topic, low 40 = address
    topic = "0x000000000000000000000000abcdef1234567890abcdef1234567890abcdef12"
    addr = _hex_to_address(topic)
    assert addr == "0xabcdef1234567890abcdef1234567890abcdef12"


def test_split_data_n_words():
    data = "0x" + ("11" * 32) + ("22" * 32) + ("33" * 32)
    words = _split_data(data, 3)
    assert len(words) == 3
    assert words[0].endswith("11")
    assert words[1].endswith("22")
    assert words[2].endswith("33")


def test_split_data_pads_short_blobs():
    """Short data should pad with zero-words."""
    data = "0x" + "ff" * 32
    words = _split_data(data, 3)
    assert len(words) == 3
    assert words[0] == "0x" + "ff" * 32
    # Remaining words are zero-padded
    assert words[1] == "0x" + "0" * 64
    assert words[2] == "0x" + "0" * 64


def _word(n: int) -> str:
    """64-hex word for a uint256."""
    return format(n, "064x")


def test_decode_order_filled_sell_side_when_maker_asset_is_usdc():
    """Maker offering USDC (asset id 0) is BIDDING for the outcome,
    so the taker who fills the order is SELLING the outcome ⇒ taker SELL."""
    raw = {
        "transactionHash": "0xabc",
        "blockNumber": "0x10",
        "topics": [
            "0xtopic0",
            "0x" + "00" * 32,                            # orderHash
            "0x" + "00" * 12 + "aa" * 20,                # maker
            "0x" + "00" * 12 + "bb" * 20,                # taker
        ],
        "data": (
            "0x"
            + _word(0)            # maker_asset = 0 (USDC) → maker is bidding
            + _word(1)            # taker_asset = 1 (CTF token)
            + _word(1_000_000)    # maker_amount = 1.0 USDC (6 decimals)
            + _word(2_000_000)    # taker_amount = 2.0 shares
            + _word(0)            # fee = 0
        ),
    }
    fill = _decode_order_filled(raw)
    assert fill is not None
    assert fill.side == "SELL"
    assert fill.maker_wallet.endswith("aa" * 10)
    assert fill.taker_wallet.endswith("bb" * 10)
    assert abs(fill.maker_amount - 1.0) < 1e-6
    assert abs(fill.taker_amount - 2.0) < 1e-6
    # price = taker_amount / maker_amount
    assert abs(fill.price - 2.0) < 1e-6


def test_decode_order_filled_buy_side_when_maker_asset_is_ctf():
    """Maker offering a CTF token (non-zero asset) is ASKING; taker who
    fills is BUYING the outcome ⇒ taker BUY."""
    raw = {
        "transactionHash": "0xdef",
        "blockNumber": "0x20",
        "topics": [
            "0xtopic0",
            "0x" + "00" * 32,
            "0x" + "00" * 12 + "cc" * 20,
            "0x" + "00" * 12 + "dd" * 20,
        ],
        "data": (
            "0x"
            + _word(1)            # maker_asset = 1 (CTF token) → maker is asking
            + _word(0)            # taker_asset = 0 (USDC)
            + _word(1_000_000)    # maker_amount
            + _word(1_000_000)    # taker_amount
            + _word(0)            # fee
        ),
    }
    fill = _decode_order_filled(raw)
    assert fill is not None
    assert fill.side == "BUY"


def test_decode_handles_malformed_returns_none():
    """Missing topics ⇒ None, no crash."""
    fill = _decode_order_filled({"topics": [], "data": "0x"})
    assert fill is None


def test_insert_fills_uses_taker_asset_for_buy(tmp_path):
    """For a BUY (maker sold USDC), the recorded asset is taker_asset."""
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.execute(
        "CREATE TABLE trades (tx_hash TEXT PRIMARY KEY, wallet TEXT, "
        "asset TEXT, side TEXT, size REAL, price REAL, timestamp REAL)"
    )
    from polyagent.data.onchain_orderfilled import OnchainFill
    fill = OnchainFill(
        tx_hash="0x1", block_number=1, block_timestamp=10.0,
        maker_wallet="0xMaker", taker_wallet="0xTaker",
        maker_asset_id="0",         # USDC
        taker_asset_id="tokenX",
        maker_amount=10.0, taker_amount=20.0, fee=0.0,
        price=0.5, side="BUY",
    )
    n = insert_onchain_fills(conn, [fill])
    assert n == 1
    row = conn.execute("SELECT asset, size FROM trades WHERE tx_hash='0x1'").fetchone()
    # BUY ⇒ asset = taker_asset_id, size = taker_amount
    assert row[0] == "tokenX"
    assert row[1] == 20.0
