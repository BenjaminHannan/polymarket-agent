import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Polymarket endpoints (no auth)
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    market_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Storage
    db_path: str = str(ROOT / "data" / "paper.db")
    log_path: str = str(ROOT / "data" / "polyagent.log")

    # Paper-trading sizing
    starting_nav: float = float(os.getenv("STARTING_NAV", "10000"))
    max_markets: int = int(os.getenv("MAX_MARKETS", "50"))
    min_liquidity: float = float(os.getenv("MIN_LIQUIDITY", "5000"))
    arb_threshold: float = float(os.getenv("ARB_THRESHOLD", "0.99"))
    per_trade_size: float = float(os.getenv("PER_TRADE_SIZE", "100"))
    max_per_market: float = float(os.getenv("MAX_PER_MARKET", "500"))

    # API keys
    alchemy_rpc_url: str = os.getenv("ALCHEMY_RPC_URL", "")
    alchemy_api_key: str = os.getenv("ALCHEMY_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    bls_api_key: str = os.getenv("BLS_API_KEY", "")
    congress_api_key: str = os.getenv("CONGRESS_API_KEY", "")
    courtlistener_api_key: str = os.getenv("COURTLISTENER_API_KEY", "")
    sec_edgar_user_agent: str = os.getenv(
        "SEC_EDGAR_USER_AGENT", "polyagent contact@example.com"
    )
    bluesky_app_password: str = os.getenv("BLUESKY_APP_PASSWORD", "")

    # Ingest cadences (seconds)
    rss_poll_sec: int = int(os.getenv("RSS_POLL_SEC", "60"))
    fred_poll_sec: int = int(os.getenv("FRED_POLL_SEC", "1800"))
    bls_poll_sec: int = int(os.getenv("BLS_POLL_SEC", "1800"))
    congress_poll_sec: int = int(os.getenv("CONGRESS_POLL_SEC", "900"))
    courtlistener_poll_sec: int = int(os.getenv("COURTLISTENER_POLL_SEC", "300"))
    sec_edgar_poll_sec: int = int(os.getenv("SEC_EDGAR_POLL_SEC", "180"))

    # News matcher
    news_match_min_overlap: int = int(os.getenv("NEWS_MATCH_MIN_OVERLAP", "2"))
    news_semantic_min_sim: float = float(os.getenv("NEWS_SEMANTIC_MIN_SIM", "0.40"))
    news_semantic_top_k: int = int(os.getenv("NEWS_SEMANTIC_TOP_K", "5"))
    enable_ingest: bool = os.getenv("ENABLE_INGEST", "1") == "1"

    # News-driven paper trader
    enable_news_trader: bool = os.getenv("ENABLE_NEWS_TRADER", "1") == "1"
    news_trade_per_trade_notional: float = float(os.getenv("NEWS_TRADE_PER_TRADE_NOTIONAL", "25"))
    news_trade_max_per_market: float = float(os.getenv("NEWS_TRADE_MAX_PER_MARKET", "75"))
    news_trade_max_daily: float = float(os.getenv("NEWS_TRADE_MAX_DAILY", "500"))
    news_trade_min_overlap: int = int(os.getenv("NEWS_TRADE_MIN_OVERLAP", "3"))
    news_trade_min_confidence: float = float(os.getenv("NEWS_TRADE_MIN_CONFIDENCE", "0.4"))
    news_trade_min_score: float = float(os.getenv("NEWS_TRADE_MIN_SCORE", "0.20"))
    news_trade_max_ask: float = float(os.getenv("NEWS_TRADE_MAX_ASK", "0.85"))
    news_trade_cooldown_sec: float = float(os.getenv("NEWS_TRADE_COOLDOWN_SEC", "300"))

    # Statistical layer (LightGBM)
    enable_stat_signal: bool = os.getenv("ENABLE_STAT_SIGNAL", "1") == "1"
    lgbm_model_path: str = os.getenv("LGBM_MODEL_PATH", "")
    stat_poll_sec: float = float(os.getenv("STAT_POLL_SEC", "120"))
    stat_min_edge: float = float(os.getenv("STAT_MIN_EDGE", "0.10"))

    # Combined (log-pool) signal
    enable_combined_signal: bool = os.getenv("ENABLE_COMBINED_SIGNAL", "1") == "1"
    combiner_path: str = os.getenv("COMBINER_PATH", "")

    # News aggregator window + exponential-decay half-life. Signals older than
    # `window_sec` are dropped entirely; within the window, weight = exp(-age/tau)
    # where tau = half_life_sec / ln(2). Set half_life to 0 to disable decay
    # (uniform mean over the window).
    news_match_window_sec: float = float(os.getenv("NEWS_MATCH_WINDOW_SEC", str(7 * 86400)))
    news_match_half_life_sec: float = float(os.getenv("NEWS_MATCH_HALF_LIFE_SEC", str(2 * 86400)))

    # Combined-signal paper trader
    enable_combined_trader: bool = os.getenv("ENABLE_COMBINED_TRADER", "1") == "1"
    # Halved Kelly + halved per-trade cap after diagnostic showed 22%
    # of notional eaten by spread/queue burn on a longshot-heavy book.
    combined_kelly_mult: float = float(os.getenv("COMBINED_KELLY_MULT", "0.075"))
    combined_max_per_trade_kelly: float = float(os.getenv("COMBINED_MAX_PER_TRADE_KELLY", "0.025"))
    combined_max_per_trade_notional: float = float(os.getenv("COMBINED_MAX_PER_TRADE_NOTIONAL", "30"))
    combined_max_per_market_notional: float = float(os.getenv("COMBINED_MAX_PER_MARKET_NOTIONAL", "100"))
    combined_max_daily_notional: float = float(os.getenv("COMBINED_MAX_DAILY_NOTIONAL", "500"))
    combined_max_ask: float = float(os.getenv("COMBINED_MAX_ASK", "0.85"))
    combined_cooldown_sec: float = float(os.getenv("COMBINED_COOLDOWN_SEC", "600"))
    combined_theta_default: float = float(os.getenv("COMBINED_THETA_DEFAULT", "0.15"))
    # Refuse to trade tokens below this price — Della Vedova's 650-900 bps
    # half-spread on lowest-probability decile means model "edge" is
    # systematically eaten by spread on cheap longshots. Diagnostic showed
    # 52% of fills were under $0.10 with avg 22% slippage burn.
    combined_min_ask: float = float(os.getenv("COMBINED_MIN_ASK", "0.10"))
    # Fee buffer raised from 0.5pp to 2pp — must clear half-spread cleanly
    # before counting edge.
    combined_fee_buffer: float = float(os.getenv("COMBINED_FEE_BUFFER", "0.02"))

    # Per-strategy auto-throttle
    throttle_interval_sec: float = float(os.getenv("THROTTLE_INTERVAL_SEC", "300"))

    # Arb scanner visibility task
    arb_scan_poll_sec: float = float(os.getenv("ARB_SCAN_POLL_SEC", "30"))

    # Web dashboard
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8080"))

    # Stop-loss task
    stop_loss_threshold_pct: float = float(os.getenv("STOP_LOSS_THRESHOLD_PCT", "0.40"))
    stop_loss_interval_sec: float = float(os.getenv("STOP_LOSS_INTERVAL_SEC", "60"))

    # Trade hunter (aggressive 30s scanner)
    enable_trade_hunter: bool = os.getenv("ENABLE_TRADE_HUNTER", "1") == "1"
    trade_hunter_poll_sec: float = float(os.getenv("TRADE_HUNTER_POLL_SEC", "30"))
    trade_hunter_min_abs_edge: float = float(os.getenv("TRADE_HUNTER_MIN_ABS_EDGE", "0.08"))
    trade_hunter_max_abs_edge: float = float(os.getenv("TRADE_HUNTER_MAX_ABS_EDGE", "0.50"))

    # Combinatorial arbitrage scanner
    enable_combinatorial_arb: bool = os.getenv("ENABLE_COMBINATORIAL_ARB", "1") == "1"
    combinatorial_arb_poll_sec: float = float(os.getenv("COMBINATORIAL_ARB_POLL_SEC", "60"))
    combinatorial_arb_min_violation: float = float(os.getenv("COMBINATORIAL_ARB_MIN_VIOLATION", "0.05"))
    trade_hunter_log_top_n: int = int(os.getenv("TRADE_HUNTER_LOG_TOP_N", "5"))
    trade_hunter_dispatch_top_n: int = int(os.getenv("TRADE_HUNTER_DISPATCH_TOP_N", "3"))

    # Passive (maker-side) limit poster — paper-mode simulation of the
    # "post inside the spread, get filled by takers" strategy. Uses the
    # queue model to estimate fill probability per cycle.
    enable_passive_poster: bool = os.getenv("ENABLE_PASSIVE_POSTER", "1") == "1"
    passive_poster_poll_sec: float = float(os.getenv("PASSIVE_POSTER_POLL_SEC", "20"))
    passive_poster_min_edge: float = float(os.getenv("PASSIVE_POSTER_MIN_EDGE", "0.04"))
    passive_poster_max_concurrent: int = int(os.getenv("PASSIVE_POSTER_MAX_CONCURRENT", "8"))
    passive_poster_per_post_notional: float = float(os.getenv("PASSIVE_POSTER_PER_POST_NOTIONAL", "25"))
    passive_poster_max_total_notional: float = float(os.getenv("PASSIVE_POSTER_MAX_TOTAL_NOTIONAL", "300"))
    passive_poster_aggression: float = float(os.getenv("PASSIVE_POSTER_AGGRESSION", "0.4"))
    passive_poster_ttl_sec: float = float(os.getenv("PASSIVE_POSTER_TTL_SEC", "180"))
    passive_poster_max_spread: float = float(os.getenv("PASSIVE_POSTER_MAX_SPREAD", "0.10"))
    passive_poster_min_spread: float = float(os.getenv("PASSIVE_POSTER_MIN_SPREAD", "0.005"))

    # Per-message → market LLM matcher (structured (confidence, direction,
    # reason_short) signals). Idempotent under news dedup; no-op without
    # local LLM. Cheap to leave on.
    enable_message_market_matcher: bool = (
        os.getenv("ENABLE_MESSAGE_MARKET_MATCHER", "1") == "1"
    )

    # Wash-trade hygiene filter (Dubach 2026). Markets with anomalous wash
    # share are skipped by all entry strategies. Computed runtime from the
    # last-trade pattern (small positive evidence; cheap heuristic).
    enable_wash_filter: bool = os.getenv("ENABLE_WASH_FILTER", "1") == "1"
    wash_filter_max_share: float = float(os.getenv("WASH_FILTER_MAX_SHARE", "0.30"))

    # Closed-loop combiner retraining
    enable_retrain_loop: bool = os.getenv("ENABLE_RETRAIN_LOOP", "1") == "1"
    retrain_experts: str = os.getenv("RETRAIN_EXPERTS", "stat_lgbm,news_match,p_market_6h")
    retrain_horizon: str = os.getenv("RETRAIN_HORIZON", "p_market_6h")
    retrain_min_rows: int = int(os.getenv("RETRAIN_MIN_ROWS", "150"))
    retrain_increment: int = int(os.getenv("RETRAIN_INCREMENT", "50"))
    retrain_check_interval_sec: float = float(os.getenv("RETRAIN_CHECK_INTERVAL_SEC", "600"))
    retrain_regression_tolerance: float = float(os.getenv("RETRAIN_REGRESSION_TOLERANCE", "0.0"))
    retrain_allow_regression: bool = os.getenv("RETRAIN_ALLOW_REGRESSION", "0") == "1"
    retrain_forward_holdout_k: int = int(os.getenv("RETRAIN_FORWARD_HOLDOUT_K", "50"))


settings = Settings()
