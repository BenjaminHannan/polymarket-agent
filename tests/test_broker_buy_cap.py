"""Confirms the broker-level per-token BUY cap (PROJECT.md gate #3).

The cap was previously enforced only at strategy level; the
broker-side check ensures it fires even when a strategy forgets the
gate. Arb strategies legitimately need multi-fill same-token activity
for basket trades and are excluded.
"""

import asyncio

from polyagent.orderbook import BookStore, OrderBook
from polyagent.paper_broker import PaperBroker


def _make_book(token_id: str, ask_price: float = 0.50, ask_size: float = 100.0) -> OrderBook:
    b = OrderBook(token_id=token_id)
    b.asks = {ask_price: ask_size}
    b.bids = {ask_price - 0.01: ask_size}
    import time as _t
    b.last_update_ts = _t.time()
    b._mid_history.append((b.last_update_ts, ask_price - 0.005))
    return b


def _patch_db(tmp_path, name):
    import polyagent.config as cfg
    object.__setattr__(cfg.settings, "db_path", str(tmp_path / name))


def test_broker_blocks_3rd_buy_for_non_arb(tmp_path):
    """Two BUYs through, third blocked at broker level."""
    import polyagent.config as cfg
    saved = cfg.settings.db_path
    try:
        _patch_db(tmp_path, "paper_test.db")

        async def run():
            bs = BookStore()
            tok = "0x" + "a" * 64
            bs.books[tok] = _make_book(tok)
            broker = PaperBroker(book_store=bs, nav_start=10_000.0)
            await broker.open()
            try:
                f1 = await broker.submit(
                    strategy="combined_trader", condition_id="C1", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="t1",
                )
                f2 = await broker.submit(
                    strategy="combined_trader", condition_id="C1", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="t2",
                )
                f3 = await broker.submit(
                    strategy="combined_trader", condition_id="C1", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="t3-should-block",
                )
            finally:
                await broker.close()
            return f1, f2, f3

        f1, f2, f3 = asyncio.run(run())
        assert f1 > 0
        assert f2 > 0
        assert f3 == 0.0
    finally:
        object.__setattr__(cfg.settings, "db_path", saved)


def test_broker_allows_arb_strategy_to_exceed_cap(tmp_path):
    """Arb strategies bypass the cap by design (multi-leg basket trades)."""
    import polyagent.config as cfg
    saved = cfg.settings.db_path
    try:
        _patch_db(tmp_path, "paper_test_arb.db")

        async def run():
            bs = BookStore()
            tok = "0x" + "b" * 64
            bs.books[tok] = _make_book(tok)
            broker = PaperBroker(book_store=bs, nav_start=10_000.0)
            await broker.open()
            try:
                f1 = await broker.submit(
                    strategy="arb_negrisk", condition_id="C2", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="a1",
                )
                f2 = await broker.submit(
                    strategy="arb_negrisk", condition_id="C2", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="a2",
                )
                f3 = await broker.submit(
                    strategy="arb_negrisk", condition_id="C2", token_id=tok,
                    side="BUY", max_size=5.0, max_price=0.99, reason="a3-still-allowed",
                )
            finally:
                await broker.close()
            return f1, f2, f3

        f1, f2, f3 = asyncio.run(run())
        assert f1 > 0 and f2 > 0 and f3 > 0
    finally:
        object.__setattr__(cfg.settings, "db_path", saved)
