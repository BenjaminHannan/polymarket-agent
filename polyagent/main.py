"""Async entrypoint for the paper-trading agent."""

from __future__ import annotations

import asyncio
import signal

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.data import alchemy as ingest_alchemy
from polyagent.data import bls as ingest_bls
from polyagent.data import bluesky as ingest_bluesky
from polyagent.data import congress as ingest_congress
from polyagent.data import courtlistener as ingest_courtlistener
from polyagent.data import fred as ingest_fred
from polyagent.data import resolution_watcher as resolution_watcher_mod
from polyagent.data import rss as ingest_rss
from polyagent.data import sec_edgar as ingest_sec_edgar
from polyagent.data import telegram as ingest_telegram
from polyagent.data.resolution_watcher import HeldMarketTracker
from polyagent.gamma import Market, fetch_active_markets
from polyagent.news_store import NewsEvent, NewsStore
from polyagent.orderbook import BookStore
from polyagent.paper_broker import PaperBroker
from polyagent.models.lgbm import Predictor
from polyagent.models.retrain_loop import RetrainLoop
from polyagent.risk.throttle import StrategyThrottler
from polyagent.dashboard import Dashboard
from polyagent.models.llm_forecaster import LLMForecaster
from polyagent.models.news_embed_matcher import SemanticMarketIndex
from polyagent.models.psi_monitor import PSIMonitor
from polyagent.eval.harness import SharpeHarness, StrategyCertRegistry
from polyagent.risk.adverse_selection import AdverseSelectionFilter
from polyagent.risk.bocpd_gate import BOCPDGate
from polyagent.risk.exit_policy import KaminskiLoStopGate, NearResolutionLockIn
from polyagent.risk.live_ece import LiveECEMonitor
from polyagent.risk.selective_gate import SelectiveGate
from polyagent.risk.smart_money import SmartMoneyRegistry
from polyagent.risk.wash_filter import WashFilter
from polyagent.signals.combinatorial_arb import CombinatorialArb
from polyagent.signals.combined import CombinedSignaler
from polyagent.signals.consistency_check import ConsistencyCheck
from polyagent.signals.local_edge import LocalEdgeClassifier
from polyagent.signals.message_market_match import MessageMarketMatcher
from polyagent.signals.news_match import MarketIndex, NewsMatcher
from polyagent.signals.stat import StatSignaler
from polyagent.strategies.combined_trader import CombinedTrader
from polyagent.strategies.news_trader import NewsTrader
from polyagent.strategies.passive_poster import PassivePoster
from polyagent.supervisor import supervised
from polyagent.tasks_extra import signal_prune_loop, status_digest_loop, stop_loss_loop
from polyagent.trade_hunter import TradeHunter
from polyagent.ws_polymarket import drain, stream_markets

log = logging_setup.configure()


async def status_loop(broker: PaperBroker, book_store: BookStore, interval: float = 30.0) -> None:
    while True:
        await asyncio.sleep(interval)
        await broker.snapshot_nav()
        books_with_quotes = sum(
            1 for b in book_store.books.values() if b.best_bid() and b.best_ask()
        )
        log.info(
            "status",
            books_total=len(book_store.books),
            books_quoted=books_with_quotes,
            **broker.summary(),
        )


async def market_event_loop(
    queue: asyncio.Queue,
    book_store: BookStore,
) -> None:
    """Pull WSS events and apply them to the book store. (No per-event
    strategy callouts — combined_signal polls the books on its own
    schedule.)"""
    async for evt in drain(queue):
        book_store.handle(evt)


async def news_event_loop(
    queue: asyncio.Queue,
    store: NewsStore,
    matcher: NewsMatcher,
    mm_matcher: MessageMarketMatcher | None = None,
) -> None:
    """Drain news events from the queue, dedup, match → trade.

    Records ingest-to-decision latency on every news event so we can see how
    quickly news flows from the wire to the trader. The 2025 microstructure
    paper found median ingest delay sub-50ms but multi-second tail; if our
    p99 is in the seconds, latency-edge plays are unwinnable but structural
    edges still work.

    Also fans each event into the message→market LLM matcher (when
    enabled) for structured (confidence, direction, reason_short) signals.
    """
    import time as _time
    while True:
        evt: NewsEvent = await queue.get()
        t_dequeue = _time.time()
        ingest_latency = t_dequeue - evt.ts if evt.ts > 0 else None
        is_new = await store.insert(evt)
        if not is_new:
            continue
        await matcher.on_event(evt)
        if mm_matcher is not None:
            try:
                await mm_matcher.on_event(evt)
            except Exception as e:
                log.warning("mmm_event_error", err=str(e))
        decision_latency = _time.time() - t_dequeue
        if ingest_latency is not None and ingest_latency > 0:
            log.debug(
                "news_pipeline_latency",
                ingest_sec=round(ingest_latency, 3),
                decision_sec=round(decision_latency, 3),
                source=evt.source,
            )


async def news_stats_loop(store: NewsStore, interval: float = 60.0) -> None:
    import time as _time

    while True:
        await asyncio.sleep(interval)
        if store.db is None:
            continue
        cutoff = _time.time() - 300
        async with store.db.execute(
            "SELECT source, COUNT(*) FROM news WHERE ts > ? GROUP BY source ORDER BY 2 DESC",
            (cutoff,),
        ) as cur:
            rows = [tuple(r) async for r in cur]
        async with store.db.execute("SELECT COUNT(*) FROM signals") as cur:
            sig_count = (await cur.fetchone() or [0])[0]
        log.info(
            "news_stats",
            last_5min_by_source=rows,
            signals_total=sig_count,
        )


async def run() -> None:
    log.info("startup", starting_nav=settings.starting_nav, max_markets=settings.max_markets)

    markets: list[Market] = await fetch_active_markets(
        limit=settings.max_markets, min_liquidity=settings.min_liquidity
    )
    if not markets:
        log.error("no_active_markets")
        return

    asset_ids: list[str] = []
    markets_by_token: dict[str, Market] = {}
    for m in markets:
        asset_ids.extend([m.yes_token_id, m.no_token_id])
        markets_by_token[m.yes_token_id] = m
        markets_by_token[m.no_token_id] = m

    log.info(
        "markets_selected",
        n_markets=len(markets),
        n_tokens=len(asset_ids),
        sample=[m.question[:60] for m in markets[:5]],
    )

    book_store = BookStore()
    if settings.enable_wash_filter:
        book_store.wash_filter = WashFilter(max_wash_share=settings.wash_filter_max_share)
    broker = PaperBroker(book_store=book_store)
    await broker.open()

    held_tracker = HeldMarketTracker()
    held_tracker.add_many(markets)

    throttler = StrategyThrottler(
        db_path=settings.db_path,
        nav_reference=settings.starting_nav,
    )
    adverse_filter = AdverseSelectionFilter(db_path=settings.db_path)
    smart_money = SmartMoneyRegistry.load()
    psi_monitor = PSIMonitor(db_path=settings.db_path)
    live_ece = LiveECEMonitor(db_path=settings.db_path)
    # §12 — live Sharpe-honesty harness (DSR/PSR/MTRL nightly).
    sharpe_harness = SharpeHarness(db_path=settings.db_path) if settings.enable_sharpe_harness else None
    cert_registry = StrategyCertRegistry(db_path=settings.db_path) if settings.enable_strategy_cert_gate else None
    # §1 — selective-abstention gate. One instance shared across both
    # combined_trader and passive_poster so the width buffer captures
    # the full distribution of candidate signals.
    selective_gate = (
        SelectiveGate(
            target_coverage=settings.selective_gate_coverage,
            burn_in=settings.selective_gate_burn_in,
        )
        if settings.enable_selective_gate
        else None
    )
    # §10 — Kaminski-Lo stop gate + near-resolution lock-in.
    kl_gate = (
        KaminskiLoStopGate(db_path=settings.db_path)
        if settings.enable_kaminski_lo_gate
        else None
    )
    near_resolution = (
        NearResolutionLockIn(
            min_unrealized_pct=settings.near_resolution_min_unrealized_pct,
            max_hours_to_resolution=settings.near_resolution_max_hours,
            min_lock_value_usd=settings.near_resolution_min_lock_usd,
        )
        if settings.enable_near_resolution_lock_in
        else None
    )

    # §9 — BOCPD changepoint gate. Fed by the resolution_watcher with
    # win/loss outcomes; consulted by both traders for a global size
    # multiplier during detected regime changes.
    bocpd_gate = (
        BOCPDGate(
            hazard=settings.bocpd_hazard,
            cp_threshold=settings.bocpd_cp_threshold,
            deleverage_trades=settings.bocpd_deleverage_trades,
            deleverage_mult=settings.bocpd_deleverage_mult,
        )
        if settings.enable_bocpd_gate
        else None
    )

    # Shared predictor: used by stat signaler, combined signaler, and the
    # resolution watcher's outcome materializer. Loaded once.
    from pathlib import Path as _P
    model_path = settings.lgbm_model_path or str(_P(settings.db_path).parent / "lgbm_model.joblib")
    shared_predictor: Predictor | None = None
    if _P(model_path).exists():
        shared_predictor = Predictor(model_path=model_path)
        try:
            shared_predictor.load()
        except Exception as e:
            log.warning("predictor_load_failed", err=str(e))
            shared_predictor = None

    news_store = NewsStore()
    await news_store.open()

    market_idx = MarketIndex.build(markets)
    # Semantic index for news → market matching (GPU embeddings).
    semantic_idx: SemanticMarketIndex | None
    try:
        semantic_idx = SemanticMarketIndex.build(markets)
    except Exception as e:
        log.warning("semantic_index_build_failed", err=str(e))
        semantic_idx = None

    news_trader = NewsTrader(
        book_store=book_store,
        broker=broker,
        per_trade_notional=settings.news_trade_per_trade_notional,
        max_per_market_notional=settings.news_trade_max_per_market,
        max_daily_notional=settings.news_trade_max_daily,
        max_ask_price=settings.news_trade_max_ask,
        cooldown_sec=settings.news_trade_cooldown_sec,
        throttler=throttler,
    )

    matcher = NewsMatcher(
        index=market_idx,
        store=news_store,
        trade_min_overlap=settings.news_trade_min_overlap,
        trade_min_confidence=settings.news_trade_min_confidence,
        trade_min_score=settings.news_trade_min_score,
        trader=news_trader.on_signal if settings.enable_news_trader else None,
        semantic_index=semantic_idx,
        semantic_min_sim=settings.news_semantic_min_sim,
        semantic_top_k=settings.news_semantic_top_k,
    )
    # Per-message → market matcher (LLM-shaped (confidence, reason)
    # signals on top of the cosine matcher). Only active if the local
    # LLM is enabled AND a semantic index built. No-op otherwise.
    mm_matcher: MessageMarketMatcher | None = None
    if settings.enable_message_market_matcher and semantic_idx is not None:
        mm_matcher = MessageMarketMatcher(
            news_store=news_store,
            semantic_index=semantic_idx,
        )
    # Local-edge classifier exposed to downstream signalers via
    # attribute on the news_store (cheap shared registry; classify-once,
    # cache forever).
    local_edge_clf = LocalEdgeClassifier()
    setattr(news_store, "local_edge_classifier", local_edge_clf)

    market_queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)
    news_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        log.info("shutdown_requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    # Helper: wrap a coroutine factory in the supervisor and create a task.
    def _spawn(name: str, factory):
        return asyncio.create_task(supervised(name, factory), name=name)

    tasks = [
        _spawn(
            "ws_market_stream",
            lambda: stream_markets(asset_ids, market_queue, on_disconnect=book_store.invalidate),
        ),
        _spawn(
            "market_event_loop",
            lambda: market_event_loop(market_queue, book_store),
        ),
        _spawn(
            "news_event_loop",
            lambda: news_event_loop(news_queue, news_store, matcher, mm_matcher),
        ),
        _spawn("status_loop", lambda: status_loop(broker, book_store)),
        _spawn("news_stats_loop", lambda: news_stats_loop(news_store)),
        _spawn(
            "resolution_watcher",
            lambda: resolution_watcher_mod.run(
                broker, held_tracker, predictor=shared_predictor, bocpd_gate=bocpd_gate,
            ),
        ),
        _spawn(
            "throttler",
            lambda: throttler.run(interval_sec=settings.throttle_interval_sec),
        ),
        _spawn("adverse_selection", lambda: adverse_filter.run()),
        _spawn("smart_money", lambda: smart_money.run()),
        _spawn("psi_monitor", lambda: psi_monitor.run()),
        _spawn("live_ece", lambda: live_ece.run()),
        *(
            [_spawn("sharpe_harness", lambda: sharpe_harness.run())]
            if sharpe_harness is not None else []
        ),
        _spawn(
            "dashboard",
            lambda: Dashboard(
                broker=broker,
                book_store=book_store,
                markets=markets,
                db_path=settings.db_path,
                port=settings.dashboard_port,
            ).run(),
        ),
        _spawn(
            "stop_loss",
            lambda: stop_loss_loop(
                broker=broker,
                book_store=book_store,
                threshold_pct=settings.stop_loss_threshold_pct,
                interval_sec=settings.stop_loss_interval_sec,
                kaminski_lo_gate=kl_gate,
                near_resolution=near_resolution,
                held_tracker=held_tracker,
            ),
        ),
        _spawn(
            "status_digest",
            lambda: status_digest_loop(broker, book_store, interval_sec=300.0),
        ),
        _spawn(
            "signal_prune",
            lambda: signal_prune_loop(settings.db_path, keep_days=30),
        ),
    ]

    if settings.enable_stat_signal:
        stat = StatSignaler(
            book_store=book_store,
            markets=markets,
            news_store=news_store,
            model_path=model_path,
            poll_sec=settings.stat_poll_sec,
            min_edge=settings.stat_min_edge,
        )
        tasks.append(_spawn("stat_signal", lambda: stat.run()))

    combiner_path = settings.combiner_path or str(_P(settings.db_path).parent / "combiner.joblib")
    if settings.enable_combined_signal and shared_predictor is not None and _P(combiner_path).exists():
        combined_trader: CombinedTrader | None = None
        if settings.enable_combined_trader:
            # Build the certified-category allowlist from strategy_certificates
            # when the gate is enabled. Empty/None disables the gate (legacy
            # behaviour). One-line rollback: UPDATE strategy_certificates
            # SET enabled=0 WHERE name='...'.
            certified_cats: set[str] | None = None
            if settings.enable_certificate_gate:
                import sqlite3 as _sql
                import json as _json
                _conn = _sql.connect(settings.db_path)
                try:
                    _rows = _conn.execute(
                        "SELECT detail FROM strategy_certificates WHERE enabled=1"
                    ).fetchall()
                finally:
                    _conn.close()
                certified_cats = set()
                for (_detail,) in _rows:
                    try:
                        _d = _json.loads(_detail or "{}")
                    except Exception:
                        continue
                    _cat = _d.get("category")
                    if isinstance(_cat, str) and _cat:
                        certified_cats.add(_cat)
                log.info(
                    "certificate_gate_active",
                    n_certs=len(_rows),
                    allowed_categories=sorted(certified_cats),
                )
            combined_trader = CombinedTrader(
                book_store=book_store,
                broker=broker,
                kelly_mult=settings.combined_kelly_mult,
                max_per_trade_kelly=settings.combined_max_per_trade_kelly,
                max_per_trade_notional=settings.combined_max_per_trade_notional,
                max_per_market_notional=settings.combined_max_per_market_notional,
                max_daily_notional=settings.combined_max_daily_notional,
                max_ask=settings.combined_max_ask,
                cooldown_sec=settings.combined_cooldown_sec,
                theta_min_default=settings.combined_theta_default,
                min_ask=settings.combined_min_ask,
                fee_buffer=settings.combined_fee_buffer,
                throttler=throttler,
                adverse_filter=adverse_filter,
                smart_money=smart_money,
                news_store=news_store,
                selective_gate=selective_gate,
                bocpd_gate=bocpd_gate,
                certified_categories=certified_cats,
            )
        # Shared LLM forecaster + consistency-check state. The runtime task
        # populates `consistency.state[event_id]` and CombinedSignaler reads
        # the same dict to downweight the llm_forecaster expert when an
        # event's outcomes don't sum to ~1.
        _llm = LLMForecaster()
        consistency = ConsistencyCheck(
            markets=markets,
            news_store=news_store,
            llm_forecaster=_llm,
        )
        # Maker-side passive poster (paper-mode). Runs alongside the taker
        # path; receives every combined-signal candidate at a lower edge bar
        # and posts virtual passive limits.
        passive_poster: PassivePoster | None = None
        if settings.enable_passive_poster:
            passive_poster = PassivePoster(
                book_store=book_store,
                broker=broker,
                markets_by_token=markets_by_token,
                poll_sec=settings.passive_poster_poll_sec,
                min_edge=settings.passive_poster_min_edge,
                max_concurrent=settings.passive_poster_max_concurrent,
                per_post_notional=settings.passive_poster_per_post_notional,
                max_total_notional=settings.passive_poster_max_total_notional,
                aggression=settings.passive_poster_aggression,
                ttl_sec=settings.passive_poster_ttl_sec,
                max_spread=settings.passive_poster_max_spread,
                min_spread=settings.passive_poster_min_spread,
                smart_money=smart_money,
                selective_gate=selective_gate,
                bocpd_gate=bocpd_gate,
            )

        async def _on_combined_signal(
            *, market, p_combined, p_market, category,
            p_combined_low=None, p_combined_high=None,
        ):
            # Fan a single signal out to taker (combined_trader) and maker
            # (passive_poster). Both apply their own gates independently.
            if combined_trader is not None:
                try:
                    await combined_trader.on_signal(
                        market=market,
                        p_combined=p_combined,
                        p_market=p_market,
                        category=category,
                        p_combined_low=p_combined_low,
                        p_combined_high=p_combined_high,
                    )
                except Exception as e:
                    log.warning("combined_trader_fanout_error", err=str(e))
            if passive_poster is not None:
                try:
                    await passive_poster.on_signal(
                        market=market, p_combined=p_combined, p_market=p_market,
                        category=category,
                        p_combined_low=p_combined_low,
                        p_combined_high=p_combined_high,
                    )
                except Exception as e:
                    log.warning("passive_poster_fanout_error", err=str(e))

        combined = CombinedSignaler(
            book_store=book_store,
            markets=markets,
            news_store=news_store,
            predictor=shared_predictor,
            combiner_path=combiner_path,
            poll_sec=settings.stat_poll_sec,
            min_edge=settings.stat_min_edge,
            trader=(_on_combined_signal if (combined_trader is not None or passive_poster is not None) else None),
            llm_forecaster=_llm,
            consistency_state=consistency.state,
        )
        tasks.append(_spawn("combined_signal", lambda: combined.run()))
        tasks.append(_spawn("consistency_check", lambda: consistency.run()))
        if passive_poster is not None:
            tasks.append(_spawn("passive_poster", lambda: passive_poster.run()))

        # TradeHunter — aggressive 30s scanner that ranks every quoted market
        # by edge and dispatches the top candidates to the trader. Runs in
        # parallel with combined_signal (which is the slower, log-only path).
        if settings.enable_trade_hunter:
            hunter = TradeHunter(
                book_store=book_store,
                markets=markets,
                predictor=shared_predictor,
                news_store=news_store,
                combiner_path=combiner_path,
                poll_sec=settings.trade_hunter_poll_sec,
                min_abs_edge=settings.trade_hunter_min_abs_edge,
                max_abs_edge=settings.trade_hunter_max_abs_edge,
                log_top_n=settings.trade_hunter_log_top_n,
                dispatch_top_n=settings.trade_hunter_dispatch_top_n,
                trader=(_on_combined_signal if (combined_trader is not None or passive_poster is not None) else None),
            )
            tasks.append(_spawn("trade_hunter", lambda: hunter.run()))

        # Combinatorial arb scanner — finds nested-date violations across
        # related markets and (optionally) routes them to the trader.
        if settings.enable_combinatorial_arb:
            comb = CombinatorialArb(
                book_store=book_store,
                markets=markets,
                poll_sec=settings.combinatorial_arb_poll_sec,
                min_violation=settings.combinatorial_arb_min_violation,
                trader=(_on_combined_signal if (combined_trader is not None or passive_poster is not None) else None),
            )
            tasks.append(_spawn("combinatorial_arb", lambda: comb.run()))

        if settings.enable_retrain_loop:
            retrain = RetrainLoop(
                out_path=combiner_path,
                experts=settings.retrain_experts.split(","),
                horizon=settings.retrain_horizon,
                min_rows=settings.retrain_min_rows,
                increment=settings.retrain_increment,
                check_interval_sec=settings.retrain_check_interval_sec,
                regression_tolerance=settings.retrain_regression_tolerance,
                allow_regression=settings.retrain_allow_regression,
                forward_holdout_k=settings.retrain_forward_holdout_k,
            )
            tasks.append(_spawn("retrain_loop", lambda: retrain.run()))

    if settings.enable_ingest:
        tasks.extend(
            [
                _spawn("ingest_rss", lambda: ingest_rss.run(news_queue)),
                _spawn("ingest_bluesky", lambda: ingest_bluesky.run(news_queue)),
                _spawn("ingest_fred", lambda: ingest_fred.run(news_queue)),
                _spawn("ingest_bls", lambda: ingest_bls.run(news_queue)),
                _spawn("ingest_congress", lambda: ingest_congress.run(news_queue)),
                _spawn("ingest_courtlistener", lambda: ingest_courtlistener.run(news_queue)),
                _spawn("ingest_sec_edgar", lambda: ingest_sec_edgar.run(news_queue)),
                _spawn("ingest_alchemy", lambda: ingest_alchemy.run(news_queue)),
                _spawn("ingest_telegram", lambda: ingest_telegram.run(news_queue)),
            ]
        )

    try:
        # Only stop on explicit signal. If a task crashes we log and keep going;
        # if it returns voluntarily that's also fine (it just leaves).
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await broker.snapshot_nav()
        log.info("final_summary", **broker.summary())
        await broker.close()
        await news_store.close()


def cli() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
