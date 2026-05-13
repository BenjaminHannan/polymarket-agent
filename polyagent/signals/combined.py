"""Runtime combined signal: per-category log-pool over multiple experts.

Loads a v2 combiner bundle:

    {
      "version": 2,
      "horizon": "p_market_6h",
      "default": {"weights": [...], "expert_names": ["stat_lgbm", "p_market_6h"]},
      "by_category": {cat: {"weights": [...], "expert_names": [...]}, ...}
    }

For each market: categorize question -> pick weights from `by_category` if
present, else `default`. Combine stat_lgbm prediction + current market mid
via log-pool, log a `combined_signal` row when |edge| ≥ threshold.

Falls back to v1 bundles (single weights/expert_names) for compatibility.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import joblib
import structlog

from polyagent.gamma import Market
from polyagent.models.categorize import categorize
from polyagent.models.chronos import ChronosForecaster
from polyagent.models.lgbm import Predictor
from polyagent.news_store import NewsStore
from polyagent.orderbook import BookStore
from polyagent.config import settings
from polyagent.models.article_retriever import HybridArticleRetriever
from polyagent.models.llm_forecaster import LLMForecaster
from polyagent.signals.combiner import LogPoolCombiner, log_pool


# (market, p_combined, p_market, category) -> awaitable
TraderCallback = Callable[..., Awaitable[None]]

log = structlog.get_logger()


@dataclass
class CombinedSignaler:
    book_store: BookStore
    markets: list[Market]
    news_store: NewsStore
    predictor: Predictor
    combiner_path: str
    poll_sec: float = 120.0
    min_edge: float = 0.10
    trader: Optional[TraderCallback] = None
    llm_forecaster: LLMForecaster | None = None
    # Optional Chronos-Bolt forecaster. Off by default (ENABLE_CHRONOS=0).
    # When set, the live loop computes `p_chronos` per market from the
    # mid-history and includes it in `expert_probs`. It is only POOLED
    # into the combined probability when the loaded combiner bundle's
    # `expert_names` already contains "chronos" — wiring without retrain
    # per the user's "wire the integration path" rule. When unpooled, the
    # forecast still appears in the logged `detail.expert_probs` for
    # offline analysis (correlation studies before a retrain).
    chronos: ChronosForecaster | None = None
    # Karkare consistency check state (event_id -> {deviation, ts}). When
    # populated, large deviations downweight the llm_forecaster expert.
    consistency_state: dict | None = None
    # Cache (condition_id -> (p_llm, ts)) so we don't pay LLM cost every poll
    _llm_cache: dict = None  # type: ignore

    _bundle: dict | None = None
    _bundle_mtime: float = 0.0

    def __post_init__(self):
        if self._llm_cache is None:
            self._llm_cache = {}

    def _weights_for(self, question: str) -> tuple[list[float], list[str], str]:
        """Return (weights, expert_names, category_used). Falls back to default."""
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

    def _maybe_reload_bundle(self) -> None:
        """Reload combiner.joblib if its mtime advanced since last load."""
        try:
            mtime = Path(self.combiner_path).stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= self._bundle_mtime:
            return
        try:
            self._bundle = joblib.load(self.combiner_path)
            self._bundle_mtime = mtime
            n_cats = len(self._bundle.get("by_category") or {})
            log.info(
                "combiner_reloaded",
                horizon=self._bundle.get("horizon"),
                n_categories=n_cats,
                n_full_rows=self._bundle.get("n_full_rows"),
            )
        except Exception as e:
            log.warning("combiner_reload_failed", err=str(e))

    async def run(self) -> None:
        if not Path(self.combiner_path).exists():
            log.warning("combined_signal_no_combiner", path=self.combiner_path)
            await asyncio.Event().wait()
            return
        self._maybe_reload_bundle()
        if self._bundle is None:
            log.warning("combiner_initial_load_failed", path=self.combiner_path)
            await asyncio.Event().wait()
            return
        version = self._bundle.get("version", 1)
        n_cats = len(self._bundle.get("by_category") or {})
        log.info(
            "combined_signal_start",
            version=version,
            horizon=self._bundle.get("horizon"),
            n_categories=n_cats,
            n_markets=len(self.markets),
        )

        while True:
            await asyncio.sleep(self.poll_sec)
            self._maybe_reload_bundle()

            # Pre-filter to markets with current quotes; we want the BEST ASK as
            # the comparison price (that's what the trader would pay), not the
            # mid — using mid systematically overstates edge by half the spread.
            candidates: list[tuple[Market, float, list[float], list[str], str]] = []
            for m in self.markets:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    continue
                ask_yes = book.best_ask()
                if ask_yes is None:
                    continue
                p_market = ask_yes[0]  # what we'd actually pay if buying YES
                weights, expert_names, used_cat = self._weights_for(m.question)
                if not weights or not expert_names:
                    continue
                candidates.append((m, p_market, weights, expert_names, used_cat))

            if not candidates:
                continue

            # Batched predict OFF the event loop (sync LightGBM otherwise blocks).
            from polyagent.gamma import days_to_resolution as _ttr
            features = [
                (m.question, m.liquidity, m.volume_24h, _ttr(m.end_date_iso))
                for (m, *_) in candidates
            ]
            preds = await asyncio.to_thread(self.predictor.predict_batch, features)

            for (m, p_market, weights, expert_names, used_cat), pred in zip(candidates, preds):
                # Resolve news_match P(YES) live from the signals table; None if
                # no recent matcher signals exist for this market.
                news_p = None
                if "news_match" in expert_names:
                    news_p = await self.news_store.news_match_p_yes(
                        m.condition_id,
                        window_sec=settings.news_match_window_sec,
                        half_life_sec=settings.news_match_half_life_sec or None,
                    )
                expert_probs: dict[str, float] = {}
                # Optionally fetch LLM forecast for this market (cached
                # for 1h; only triggered when news_velocity is elevated)
                p_llm = None
                if (
                    self.llm_forecaster is not None
                    and self.llm_forecaster.is_enabled()
                    and "llm_forecaster" in expert_names
                ):
                    cached = self._llm_cache.get(m.condition_id)
                    if cached and (asyncio.get_running_loop().time() - cached[1] < 3600):
                        p_llm = cached[0]
                    else:
                        # Trigger only when there's news activity worth reasoning over
                        nv = await self.news_store.news_velocity(m.condition_id) if hasattr(self.news_store, "news_velocity") else None
                        if nv and nv[0] >= 1:
                            # Asof-clean + hybrid BM25/dense retrieval.
                            # Pull a wider candidate pool (200 most-recent
                            # within 7 days), then rank by RRF(BM25,
                            # dense-cosine) against the question. This gives
                            # the LLM topic-relevant articles instead of
                            # whatever happened to be most recent.
                            articles = []
                            if hasattr(self.news_store, "db") and self.news_store.db is not None:
                                import time as _t
                                asof = _t.time()
                                try:
                                    async with self.news_store.db.execute(
                                        "SELECT title FROM news WHERE ts <= ? AND ts >= ? "
                                        "ORDER BY ts DESC LIMIT 200",
                                        (asof, asof - 7 * 86400),
                                    ) as cur:
                                        cand = [
                                            row[0] async for row in cur if row and row[0]
                                        ]
                                except Exception:
                                    cand = []
                                if cand:
                                    try:
                                        retriever = HybridArticleRetriever(top_k=8)
                                        articles = retriever.retrieve(
                                            m.question, cand
                                        )
                                    except Exception as e:
                                        log.warning("hybrid_retrieval_failed", err=str(e))
                                        articles = cand[:8]
                            res = await self.llm_forecaster.forecast_async(
                                m.question, articles or []
                            )
                            if res:
                                p_llm = res["p"]
                                self._llm_cache[m.condition_id] = (
                                    p_llm,
                                    asyncio.get_running_loop().time(),
                                )

                # Optional Chronos forecast — runs when ENABLE_CHRONOS=1 and a
                # forecaster object was passed in. Computed once per candidate
                # whether or not "chronos" is in expert_names, so the detail
                # row carries the forecast for offline correlation analysis
                # ahead of any retrain. Cost is bounded by the min-history
                # gate and the chronos module's own internal load/predict
                # short-circuits when not installed.
                p_chronos = None
                if settings.enable_chronos and self.chronos is not None:
                    try:
                        book_yes = self.book_store.books.get(m.yes_token_id)
                        hist = (
                            [pt[1] for pt in book_yes._mid_history]
                            if book_yes is not None
                            and getattr(book_yes, "_mid_history", None)
                            else []
                        )
                        if len(hist) >= settings.chronos_min_history:
                            fc = self.chronos.predict(
                                prices=hist[-max(64, settings.chronos_horizon * 2):],
                                horizon=settings.chronos_horizon,
                            )
                            if fc:
                                # Take the terminal forecast; clamp to a
                                # legal probability range. Chronos is
                                # forecasting the next mid (which is a price
                                # in [0, 1] for binary tokens), so a direct
                                # use as P(YES) is reasonable but coarse.
                                p_chronos = max(0.01, min(0.99, float(fc[-1])))
                    except Exception as e:
                        log.warning("chronos_predict_error", err=str(e))

                for name in expert_names:
                    if name == "stat_lgbm":
                        expert_probs[name] = pred["calibrated"]
                    elif name.startswith("p_market") or name == "market":
                        expert_probs[name] = p_market
                    elif name == "news_match":
                        expert_probs[name] = news_p if news_p is not None else p_market
                    elif name == "llm_forecaster":
                        expert_probs[name] = p_llm if p_llm is not None else p_market
                    elif name == "chronos":
                        expert_probs[name] = p_chronos if p_chronos is not None else p_market
                    else:
                        expert_probs[name] = 0.5

                ordered = [expert_probs[n] for n in expert_names]
                # Karkare consistency downweight: if this market's NegRisk
                # event has a large LLM consistency deviation, scale down
                # the llm_forecaster weight before pooling. Large deviation
                # (|sum_yes - 1| ~ 0.3+) → near-zero llm weight; small
                # deviation passes through unchanged.
                effective_weights = list(weights)
                # LOCAL-EDGE TILT: if a market has been flagged as having
                # a non-English / insider information edge, upweight the
                # news_match expert (Telegram + non-English wires can
                # actually move it) and downweight stat_lgbm (which has
                # no view on it). One-shot LLM call cached per question.
                local_edge_clf = getattr(self.news_store, "local_edge_classifier", None)
                if local_edge_clf is not None:
                    try:
                        le = local_edge_clf.classify(m.question)
                    except Exception:
                        le = None
                    if le is not None and le.has_local_edge and le.confidence >= 0.6:
                        # Multiplicative tilt before the market-prior floor.
                        for nm_idx, nm in enumerate(expert_names):
                            if nm == "news_match":
                                effective_weights[nm_idx] *= 1.5
                            elif nm == "llm_forecaster":
                                effective_weights[nm_idx] *= 1.5
                            elif nm == "stat_lgbm":
                                effective_weights[nm_idx] *= 0.6
                        # renormalize so weights still sum to 1
                        s = sum(effective_weights)
                        if s > 0:
                            effective_weights = [w / s for w in effective_weights]

                # MARKET-PRIOR SHRINKAGE (Della Vedova 2026, Akey 2026,
                # Whelan 2024). Empirical finding: question-only ML models
                # have Brier ~0.13-0.18 while the market price has Brier
                # ~0.05-0.10 on resolved markets. The model's apparent
                # "edge" is mostly its residual error projected onto the
                # better-calibrated market price. Defense: enforce a
                # MINIMUM weight on p_market in the runtime log-pool,
                # regardless of what the SLSQP weights say. λ-shrinkage
                # in the literature uses λ=0.10-0.15 on the model side;
                # we set the market-floor to 0.60 as a conservative cap.
                MARKET_WEIGHT_FLOOR = 0.60
                market_idx = None
                for i, n in enumerate(expert_names):
                    if n.startswith("p_market") or n == "market":
                        market_idx = i
                        break
                if market_idx is not None and effective_weights[market_idx] < MARKET_WEIGHT_FLOOR:
                    target = MARKET_WEIGHT_FLOOR
                    # Re-allocate: market gets at least `target`, the rest
                    # of the weight (1 - target) is distributed across the
                    # other experts in proportion to their original weights.
                    others_total = sum(
                        w for j, w in enumerate(effective_weights) if j != market_idx
                    )
                    if others_total > 0:
                        scale = (1.0 - target) / others_total
                        new_weights = [
                            (target if j == market_idx else w * scale)
                            for j, w in enumerate(effective_weights)
                        ]
                        effective_weights = new_weights
                if (
                    self.consistency_state
                    and "llm_forecaster" in expert_names
                    and getattr(m, "event_id", None)
                    and getattr(m, "neg_risk", False)
                ):
                    rec = self.consistency_state.get(m.event_id)
                    if rec is not None:
                        dev = float(rec.get("deviation", 0.0))
                        # Smooth multiplicative damping: 1.0 at dev=0, 0.1 at dev>=0.4
                        damp = max(0.1, 1.0 - min(1.0, dev / 0.4) * 0.9)
                        idx_llm = expert_names.index("llm_forecaster")
                        effective_weights[idx_llm] = effective_weights[idx_llm] * damp
                p_combined = log_pool(ordered, effective_weights)
                edge = p_combined - p_market
                if abs(edge) < self.min_edge:
                    continue
                # Conformal lower bound on the stat_lgbm output (idea #9):
                # propagate the Venn-Abers / cell-calibrator interval so the
                # trader can size off a worst-case probability AND so the
                # selective-abstention gate (§1) can compute width. If we
                # don't have an interval (cell didn't get enough samples),
                # pass the point estimate through — strategies become no-ops.
                p_low_stat = pred.get("calibrated_low")
                p_high_stat = pred.get("calibrated_high")
                # Reapply the same log-pool to the lower-bound and upper-
                # bound stat experts to get bounded combined probabilities
                # — only when stat is in the expert list.
                p_combined_low = p_combined
                p_combined_high = p_combined
                if p_low_stat is not None and "stat_lgbm" in expert_names:
                    idx_stat = expert_names.index("stat_lgbm")
                    ordered_low = list(ordered)
                    ordered_low[idx_stat] = float(p_low_stat)
                    try:
                        p_combined_low = log_pool(ordered_low, effective_weights)
                    except Exception:
                        p_combined_low = p_combined
                if p_high_stat is not None and "stat_lgbm" in expert_names:
                    idx_stat = expert_names.index("stat_lgbm")
                    ordered_high = list(ordered)
                    ordered_high[idx_stat] = float(p_high_stat)
                    try:
                        p_combined_high = log_pool(ordered_high, effective_weights)
                    except Exception:
                        p_combined_high = p_combined
                direction = "yes" if edge > 0 else "no"
                # Microstructure context (informational, not yet in the
                # log-pool — but useful for downstream traders to gate on)
                book = self.book_store.books.get(m.yes_token_id)
                imb = book.imbalance(5) if book else None
                mom_5m = book.momentum(300) if book else None
                mom_1h = book.momentum(3600) if book else None
                # News context (velocity + sentiment z-score)
                news_vel = await self.news_store.news_velocity(m.condition_id)
                news_z = await self.news_store.sentiment_zscore(m.condition_id)
                detail = {
                    "p_combined": round(p_combined, 4),
                    "p_market": round(p_market, 4),
                    "edge": round(edge, 4),
                    "category_used": used_cat,
                    "weights": dict(zip(expert_names, [round(w, 3) for w in weights])),
                    "expert_probs": {k: round(v, 4) for k, v in expert_probs.items()},
                    # Surface p_chronos in the detail even when it isn't an
                    # active expert in the loaded combiner — gives the
                    # offline retrain pipeline a feature column to correlate.
                    "p_chronos": (
                        round(p_chronos, 4)
                        if (settings.enable_chronos and p_chronos is not None)
                        else None
                    ),
                    "news_match_present": news_p is not None,
                    "question": m.question[:160],
                    "source": "combined",
                    # Microstructure
                    "book_imbalance": round(imb, 3) if imb is not None else None,
                    "mom_5m": round(mom_5m, 4) if mom_5m is not None else None,
                    "mom_1h": round(mom_1h, 4) if mom_1h is not None else None,
                    # News context
                    "news_velocity_short": news_vel[0] if news_vel else 0,
                    "news_velocity_ratio": round(news_vel[2], 2) if news_vel else None,
                    "sentiment_zscore": round(news_z, 3) if news_z is not None else None,
                }
                await self.news_store.insert_signal(
                    strategy="combined",
                    condition_id=m.condition_id,
                    direction=direction,
                    score=abs(edge),
                    news_hash="",
                    detail=detail,
                )
                log.info(
                    "combined_signal",
                    condition_id=m.condition_id,
                    question=m.question[:80],
                    category=used_cat,
                    p_combined=round(p_combined, 3),
                    p_market=round(p_market, 3),
                    edge=round(edge, 3),
                    direction=direction,
                )

                if self.trader is not None:
                    try:
                        await self.trader(
                            market=m,
                            p_combined=p_combined,
                            p_market=p_market,
                            category=used_cat,
                            p_combined_low=p_combined_low,
                            p_combined_high=p_combined_high,
                        )
                    except Exception as e:
                        log.warning("combined_trader_error", err=str(e))
