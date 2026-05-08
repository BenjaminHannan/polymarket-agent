"""TradeHunter — continuously scans every subscribed market for trade
candidates, ranks by edge × confidence × throttle headroom, and dispatches
the best ones to the combined trader.

Where CombinedSignaler runs every 120s and emits one signal per market when
edge > min_edge, TradeHunter runs every 30s and:
  - Computes combined probability for every quoted market in one batched
    GPU pass (LGBM predict_batch)
  - Filters/ranks by gate-weighted score
  - Logs the top N candidates each cycle (visible "thinking" activity)
  - Dispatches the top-K through the trader callback so they're acted on

Net effect: the bot is searching the whole opportunity surface 4x more
often, with structured ranking instead of first-past-the-post.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import joblib
import structlog
from pathlib import Path

from polyagent.gamma import Market, days_to_resolution as _ttr
from polyagent.models.categorize import categorize
from polyagent.models.lgbm import Predictor
from polyagent.news_store import NewsStore
from polyagent.orderbook import BookStore
from polyagent.signals.combiner import log_pool

log = structlog.get_logger()

TraderCallback = Callable[..., Awaitable[None]]


@dataclass
class TradeHunter:
    book_store: BookStore
    markets: list[Market]
    predictor: Predictor
    news_store: NewsStore
    combiner_path: str
    poll_sec: float = 30.0
    min_abs_edge: float = 0.08      # raw edge gate (before fee buffer)
    max_abs_edge: float = 0.50      # cap: anything beyond this is almost
                                    # certainly model error on a question-only
                                    # signal — skip it rather than over-trade
    log_top_n: int = 5              # how many candidates to surface each cycle
    dispatch_top_n: int = 3          # how many to actually send to the trader
    trader: Optional[TraderCallback] = None

    _bundle: dict | None = None
    _bundle_mtime: float = 0.0

    def _maybe_reload_bundle(self) -> None:
        try:
            mtime = Path(self.combiner_path).stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= self._bundle_mtime:
            return
        try:
            self._bundle = joblib.load(self.combiner_path)
            self._bundle_mtime = mtime
        except Exception as e:
            log.warning("hunter_combiner_reload_failed", err=str(e))

    def _weights_for(self, question: str) -> tuple[list[float], list[str], str]:
        bundle = self._bundle or {}
        cat = categorize(question)
        by_cat = bundle.get("by_category") or {}
        if cat in by_cat:
            entry = by_cat[cat]
            return entry["weights"], entry["expert_names"], cat
        default = bundle.get("default") or {
            "weights": bundle.get("weights"),
            "expert_names": bundle.get("expert_names"),
        }
        return default["weights"], default["expert_names"], "_default"

    async def run(self) -> None:
        if not Path(self.combiner_path).exists():
            log.warning("trade_hunter_no_combiner", path=self.combiner_path)
            await asyncio.Event().wait()
            return
        self._maybe_reload_bundle()
        log.info(
            "trade_hunter_start",
            poll_sec=self.poll_sec,
            n_markets=len(self.markets),
            min_abs_edge=self.min_abs_edge,
        )

        while True:
            await asyncio.sleep(self.poll_sec)
            t_start = time.time()
            self._maybe_reload_bundle()

            # Filter to markets with current quotes (best ask)
            quoted: list[tuple[Market, float, list[float], list[str], str]] = []
            for m in self.markets:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    continue
                ask = book.best_ask()
                if ask is None or ask[0] <= 0 or ask[0] >= 1.0:
                    continue
                weights, expert_names, used_cat = self._weights_for(m.question)
                if not weights or not expert_names:
                    continue
                quoted.append((m, ask[0], weights, expert_names, used_cat))

            if not quoted:
                log.info("trade_hunter_cycle_empty")
                continue

            # Single batched LGBM predict for all quoted markets
            features = [
                (m.question, m.liquidity, m.volume_24h, _ttr(m.end_date_iso))
                for (m, *_rest) in quoted
            ]
            try:
                preds = await asyncio.to_thread(self.predictor.predict_batch, features)
            except Exception as e:
                log.warning("trade_hunter_predict_failed", err=str(e))
                continue

            # Score every market
            ranked: list[dict] = []
            for (m, p_market, weights, expert_names, used_cat), pred in zip(quoted, preds):
                # news_match expert (cheap async query)
                news_p = None
                if "news_match" in expert_names:
                    try:
                        news_p = await self.news_store.news_match_p_yes(m.condition_id)
                    except Exception:
                        news_p = None
                expert_probs: dict[str, float] = {}
                for name in expert_names:
                    if name == "stat_lgbm":
                        expert_probs[name] = pred["calibrated"]
                    elif name.startswith("p_market") or name == "market":
                        expert_probs[name] = p_market
                    elif name == "news_match":
                        expert_probs[name] = news_p if news_p is not None else p_market
                    else:
                        expert_probs[name] = 0.5
                ordered = [expert_probs[n] for n in expert_names]
                p_combined = log_pool(ordered, weights)
                edge = p_combined - p_market
                if abs(edge) < self.min_abs_edge:
                    continue
                # Skip implausibly large edges — these are almost always
                # model overconfidence on familiar question patterns rather
                # than genuine mispricings. The model can't see live state,
                # so 50pp+ disagreement with the book is a red flag.
                if abs(edge) > self.max_abs_edge:
                    continue
                ranked.append(
                    {
                        "market": m,
                        "p_combined": p_combined,
                        "p_market": p_market,
                        "edge": edge,
                        "category": used_cat,
                        "abs_edge": abs(edge),
                    }
                )

            ranked.sort(key=lambda d: -d["abs_edge"])

            elapsed = time.time() - t_start
            log.info(
                "trade_hunter_cycle",
                scanned=len(quoted),
                candidates=len(ranked),
                elapsed_sec=round(elapsed, 2),
            )

            # Surface top candidates to logs (visibility)
            for c in ranked[: self.log_top_n]:
                log.info(
                    "trade_hunter_candidate",
                    question=c["market"].question[:90],
                    category=c["category"],
                    p_combined=round(c["p_combined"], 3),
                    p_market=round(c["p_market"], 3),
                    edge=round(c["edge"], 3),
                    direction=("yes" if c["edge"] > 0 else "no"),
                )

            # Dispatch top to trader
            if self.trader is not None:
                for c in ranked[: self.dispatch_top_n]:
                    try:
                        await self.trader(
                            market=c["market"],
                            p_combined=c["p_combined"],
                            p_market=c["p_market"],
                            category=c["category"],
                        )
                    except Exception as e:
                        log.warning("trade_hunter_dispatch_error", err=str(e))
