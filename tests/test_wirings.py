"""Integration tests for the new wirings (VPIN/LR/microprice/conformal-Kelly).

These verify the strategies actually consult the new gates rather than
silently bypassing them. Each test instantiates the strategy with a
fake gate that records calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from polyagent.risk.vpin_gate import VPINGate
from polyagent.risk.likelihood_ratio_gate import LikelihoodRatioGate


@dataclass
class _RecordingVPIN:
    """Stand-in VPIN gate that records every call and forces block."""
    calls: list[tuple[str, str]] = field(default_factory=list)
    decision: bool = True

    def allow_quote(self, token_id, side):
        self.calls.append((token_id, side))
        return (self.decision, {"decision": "test"})

    def record_trade(self, *a, **kw):
        pass


def test_vpin_gate_consulted_when_attached():
    """The wiring in passive_poster_v2 routes _vpin_allow through the
    attached gate. We don't run the full async cycle here; we just check
    the helper method behavior directly."""
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2
    rec = _RecordingVPIN(decision=True)
    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
        vpin_gate=rec,
    )
    assert p._vpin_allow("tok1", "BUY") is True
    assert rec.calls == [("tok1", "BUY")]


def test_vpin_gate_block_returns_false():
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2
    rec = _RecordingVPIN(decision=False)
    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
        vpin_gate=rec,
    )
    assert p._vpin_allow("tok2", "SELL") is False


def test_vpin_gate_absent_allows_all():
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2
    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
    )
    # No gate attached ⇒ always allow.
    assert p._vpin_allow("tok3", "BUY") is True


def test_vpin_gate_error_recovers():
    """A gate that raises should not crash the strategy."""
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2

    class _BoomGate:
        def allow_quote(self, *a, **kw):
            raise RuntimeError("boom")

        def record_trade(self, *a, **kw):
            pass

    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
        vpin_gate=_BoomGate(),
    )
    # On error, default to allow (open).
    assert p._vpin_allow("tok4", "BUY") is True


def test_effective_quote_size_without_wash_conn():
    """No wash_graph_conn ⇒ return quote_size unchanged."""
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2
    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
        quote_size=25.0,
    )
    assert p._effective_quote_size("tok") == 25.0


def test_effective_quote_size_with_wash_conn(tmp_path):
    """With a wash-graph DB showing high wash share, quote should shrink."""
    import sqlite3
    from polyagent.risk.wash_graph import ensure_tables
    from polyagent.strategies.passive_poster_v2 import PassivePosterV2
    conn = sqlite3.connect(tmp_path / "t.db")
    ensure_tables(conn)
    # Inject a market wash score of 0.6 for tok
    conn.execute(
        "INSERT INTO market_wash_score (asset, wash_share, n_trades, last_updated) "
        "VALUES ('tok', 0.6, 100, 0)"
    )
    conn.commit()
    p = PassivePosterV2(
        book_store=None,                  # type: ignore
        broker=None,                      # type: ignore
        markets_by_token={},
        quote_size=25.0,
        wash_graph_conn=conn,
    )
    # 25 × (1 − 0.6) = 10
    assert p._effective_quote_size("tok") == 10.0


def test_vpin_gate_record_trade_via_book_store():
    """When BookStore has a vpin_gate attached, last_trade_price events
    forward to record_trade with Lee-Ready direction inference."""
    from polyagent.orderbook import BookStore
    gate = VPINGate(bucket_volume=1000, n_buckets=5, min_buckets=2)
    bs = BookStore()
    bs.vpin_gate = gate
    # Seed a book so the prior mid is known.
    bs.handle({
        "event_type": "book",
        "asset_id": "tok1",
        "bids": [{"price": "0.49", "size": "100"}],
        "asks": [{"price": "0.51", "size": "100"}],
    })
    # Trade at 0.52 (above the 0.50 mid) ⇒ taker BUY
    bs.handle({
        "event_type": "last_trade_price",
        "asset_id": "tok1",
        "price": "0.52",
        "size": "50",
    })
    # With bucket_volume=1000 and size=50 the bucket hasn't sealed yet —
    # the running bucket's BUY side should hold 50.
    rb = gate._running_bucket.get("tok1")
    assert rb is not None and rb[0] >= 50.0  # buy bucket


def test_vpin_gate_records_sell_side():
    """Trade below the prior mid is classified as SELL."""
    from polyagent.orderbook import BookStore
    gate = VPINGate(bucket_volume=1000, n_buckets=5, min_buckets=2)
    bs = BookStore()
    bs.vpin_gate = gate
    bs.handle({
        "event_type": "book",
        "asset_id": "tok1",
        "bids": [{"price": "0.49", "size": "100"}],
        "asks": [{"price": "0.51", "size": "100"}],
    })
    # Trade at 0.48 (below mid 0.50) ⇒ taker SELL
    bs.handle({
        "event_type": "last_trade_price",
        "asset_id": "tok1",
        "price": "0.48",
        "size": "30",
    })
    rb = gate._running_bucket.get("tok1")
    assert rb is not None and rb[1] >= 30.0  # sell bucket


def test_lr_gate_admits_during_burn_in():
    """LR gate should admit everything before fit is ready."""
    g = LikelihoodRatioGate(feature_dim=4, burn_in=200)
    assert g.admit([0.5, 0.5, 0.0, 100.0]) is True
