"""In-memory order book reconstruction from Polymarket WSS messages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class OrderBook:
    token_id: str
    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)
    last_trade_price: float | None = None
    last_trade_ts: float | None = None
    tick_size: float = 0.01
    last_update_ts: float | None = None
    # Rolling history of (ts, mid) pairs for momentum / vol features.
    _mid_history: list[tuple[float, float]] = field(default_factory=list)

    def apply_snapshot(self, msg: dict) -> None:
        self.bids.clear()
        self.asks.clear()
        for lvl in msg.get("bids", []) or []:
            p, s = float(lvl["price"]), float(lvl["size"])
            if s > 0:
                self.bids[p] = s
        for lvl in msg.get("asks", []) or []:
            p, s = float(lvl["price"]), float(lvl["size"])
            if s > 0:
                self.asks[p] = s
        ts = msg.get("tick_size")
        if ts is not None:
            try:
                self.tick_size = float(ts)
            except (TypeError, ValueError):
                pass
        self._touch()

    def apply_price_change(self, msg: dict) -> None:
        for chg in msg.get("changes", []) or []:
            try:
                p = float(chg["price"])
                s = float(chg["size"])
            except (KeyError, TypeError, ValueError):
                continue
            side = (chg.get("side") or "").upper()
            book = self.bids if side == "BUY" else self.asks if side == "SELL" else None
            if book is None:
                continue
            if s == 0:
                book.pop(p, None)
            else:
                book[p] = s
        self._touch()

    def apply_trade(self, msg: dict) -> None:
        try:
            self.last_trade_price = float(msg["price"])
            self.last_trade_ts = time.time()
        except (KeyError, TypeError, ValueError):
            pass

    def _touch(self) -> None:
        """Mark this book as just-updated and record a mid sample."""
        self.last_update_ts = time.time()
        m = self.mid()
        if m is not None:
            self._mid_history.append((self.last_update_ts, m))
            # Keep only last 4 hours of mids; bounded memory.
            cutoff = self.last_update_ts - 4 * 3600
            self._mid_history = [t for t in self._mid_history if t[0] >= cutoff]

    def imbalance(self, depth_levels: int = 5) -> float | None:
        """Bid-side share of top-N total depth, in [0, 1]. >0.5 means bid-heavy."""
        if not self.bids or not self.asks:
            return None
        bids_top = sorted(self.bids.items(), reverse=True)[:depth_levels]
        asks_top = sorted(self.asks.items())[:depth_levels]
        bid_sum = sum(s for _, s in bids_top)
        ask_sum = sum(s for _, s in asks_top)
        total = bid_sum + ask_sum
        if total <= 0:
            return None
        return bid_sum / total

    def microprice(self) -> float | None:
        """Stoikov 2018 microprice: bid×ask_size + ask×bid_size, normalized.
        Provably better short-horizon predictor than mid for CLOBs."""
        if not self.bids or not self.asks:
            return None
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        bp, bs = bid
        ap, as_ = ask
        denom = bs + as_
        if denom <= 0:
            return None
        return (bp * as_ + ap * bs) / denom

    def ofi(self, depth_levels: int = 5) -> float | None:
        """Order Flow Imbalance (Cont/Kukanov/Stoikov 2014, multi-level).

        Approximation using current snapshot (we don't track event-level
        change rates here): sum across top-N levels of
            (bid_size_i - ask_size_i) × (1 / (i+1))
        normalized by total depth. Captures depth concentration on one side.
        """
        if not self.bids or not self.asks:
            return None
        bids_top = sorted(self.bids.items(), reverse=True)[:depth_levels]
        asks_top = sorted(self.asks.items())[:depth_levels]
        if not bids_top or not asks_top:
            return None
        n = max(len(bids_top), len(asks_top))
        weighted = 0.0
        total = 0.0
        for i in range(n):
            b = bids_top[i][1] if i < len(bids_top) else 0.0
            a = asks_top[i][1] if i < len(asks_top) else 0.0
            w = 1.0 / (i + 1)
            weighted += w * (b - a)
            total += w * (b + a)
        if total <= 0:
            return None
        return weighted / total

    def momentum(self, window_sec: float) -> float | None:
        """Log-return of mid over the lookback window. None if insufficient history."""
        if not self._mid_history:
            return None
        now = self.last_update_ts or time.time()
        target = now - window_sec
        # Find the oldest mid within window
        old = None
        for t, m in self._mid_history:
            if t >= target:
                old = m
                break
        if old is None or old <= 0:
            return None
        cur_mid = self.mid()
        if cur_mid is None or cur_mid <= 0:
            return None
        import math
        return math.log(cur_mid / old)

    def realized_vol(self, window_sec: float) -> float | None:
        """Std-dev of log-returns within the window."""
        if len(self._mid_history) < 3:
            return None
        now = self.last_update_ts or time.time()
        target = now - window_sec
        pts = [(t, m) for t, m in self._mid_history if t >= target and m > 0]
        if len(pts) < 3:
            return None
        import math
        rets = [math.log(pts[i][1] / pts[i - 1][1]) for i in range(1, len(pts))]
        if not rets:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        p = max(self.bids)
        return p, self.bids[p]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        p = min(self.asks)
        return p, self.asks[p]

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b[0] + a[0]) / 2.0

    def spread(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a[0] - b[0]


class BookStore:
    """Keeps a book per token_id. Routes WSS messages by event_type."""

    def __init__(self) -> None:
        self.books: dict[str, OrderBook] = {}
        # Optional wash-trade filter — set externally. When provided, every
        # book change / trade event is forwarded so the filter can compute
        # per-token wash share and blacklist anomalous markets.
        self.wash_filter = None

    def get(self, token_id: str) -> OrderBook:
        b = self.books.get(token_id)
        if b is None:
            b = OrderBook(token_id=token_id)
            self.books[token_id] = b
        return b

    def invalidate(self, token_ids: list[str]) -> None:
        """Clear book state for a list of token_ids (e.g. on WSS disconnect).

        The next `book` snapshot from the server will repopulate. Until then,
        strategies that check best_bid()/best_ask() see no quotes and skip.
        """
        for tid in token_ids:
            tid = str(tid)
            b = self.books.get(tid)
            if b is None:
                continue
            b.bids.clear()
            b.asks.clear()
            # last_trade_price is fine to retain (informational only).

    def handle(self, msg: dict) -> str | None:
        """Apply a single WSS event. Returns token_id if known."""
        event_type = msg.get("event_type") or msg.get("type")
        token_id = msg.get("asset_id") or msg.get("token_id") or msg.get("market")
        if not token_id:
            return None
        book = self.get(str(token_id))

        if event_type == "book":
            book.apply_snapshot(msg)
            if self.wash_filter is not None:
                self.wash_filter.on_book_change(str(token_id))
        elif event_type == "price_change":
            book.apply_price_change(msg)
            if self.wash_filter is not None:
                self.wash_filter.on_book_change(str(token_id))
        elif event_type == "last_trade_price":
            book.apply_trade(msg)
            if self.wash_filter is not None:
                self.wash_filter.on_trade(str(token_id))
        elif event_type == "tick_size_change":
            try:
                book.tick_size = float(msg.get("new_tick_size", book.tick_size))
            except (TypeError, ValueError):
                pass
        return str(token_id)
