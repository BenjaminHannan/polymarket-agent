# Polyagent — Polymarket Paper Trading Agent

A research-grade prediction-market trading bot built incrementally over many sessions. Streams the live Polymarket WSS feed, ingests news from 30+ free sources, predicts P(YES) per market with a calibrated LightGBM model that uses sentence-transformer embeddings, blends predictions with the live market price (per-category log-pool), trades on paper via simulated VWAP fills, and exposes a live web dashboard.

**This is paper trading only.** No private keys, no real money, no on-chain order placement. Real-money conversion would require components I (Claude) cannot build per safety policy: wallet handling, EIP-712 signing, USDC funding.

---

## Current state at a glance

```
Markets streaming:      ~500 markets (~1,500 token books — both YES and NO)
Background tasks:       ~25 supervised, all alive
Model:                  LightGBM LambdaRank (day-grouped) + CPCV market-id-purged folds
                        AUC 0.7739 on 10,212 resolved markets (held-out CV)
Calibration:            Per-(category × horizon) cells with three-tier fallback —
                        isotonic ≥80 / Venn-Abers ≥30 / Beta ≥15 / global isotonic
Combiner:               Per-category log-pool, runtime-shrunk so p_market gets
                        ≥60% weight regardless of trained weights
LLM forecaster:         Phi-4-mini-instruct + hybrid BM25+dense retrieval +
                        AIA debiasers + Karkare NegRisk consistency
Risk gates active:      29+ — see "Risk gate stack" section
GPU:                    ~5 GB used (RTX 5070 Ti, sm_120 Blackwell, cu128 nightly)
```

The honest expected paper P&L over a session, given the literature-backed
ceiling of a question-only ML model on Polymarket, is **±2% noise around
zero**. Real positive ROI requires execution edge (passive maker fills) or
NegRisk arbitrage, not directional model bets. See "Loss diagnostics &
fixes" below for the empirical record.

Dashboard: `http://127.0.0.1:8080`. Health: `/api/health`. Kill: `touch data/.STOP`.

---

## High-level architecture

```
                       ┌───────────────────────────────────────────────────┐
                       │                Data ingestion layer                │
                       │  Polymarket WSS, Gamma REST, 23 RSS feeds,         │
                       │  FRED, BLS, Congress, CourtListener, SEC EDGAR,   │
                       │  Bluesky Jetstream, Alchemy Polygon RPC           │
                       └──────────────────┬────────────────────────────────┘
                                          │
                          asyncio.Queue (market events + news events)
                                          │
                       ┌──────────────────▼────────────────────────────────┐
                       │         BookStore (in-memory order books)         │
                       │   imbalance, momentum, vol, last_update_ts         │
                       └──────────────────┬────────────────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────────────────┐
                       │              Signal layer (GPU)                   │
                       │                                                    │
                       │   stat_signal      LGBM + bge-large embeddings    │
                       │   news_match       semantic + cross-encoder + FinBERT │
                       │   combined_signal  per-category log-pool combiner │
                       │   combinatorial_arb nested-date violation finder  │
                       │   trade_hunter     30s scanner ranking all markets│
                       └──────────────────┬────────────────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────────────────┐
                       │            Risk layer (gates everything)          │
                       │ throttler · drawdown · kill_switch · adverse_sel  │
                       │ stop_loss · spread/depth filters · concentration  │
                       │ Thompson-bandit Kelly · DD-conditioned θ_min      │
                       │ daily loss limit · stale-quote skip · fee buffer  │
                       └──────────────────┬────────────────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────────────────┐
                       │           PaperBroker (asyncio-locked)            │
                       │  cash + positions, VWAP fill simulation,          │
                       │  state recovery on restart, atomic settle,        │
                       │  WAL-mode SQLite persistence                      │
                       └──────────────────┬────────────────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────────────────┐
                       │                Dashboard (aiohttp)                │
                       │  http://127.0.0.1:8080 — portfolio, positions,    │
                       │  fills, NAV chart, per-strategy P&L, /api/health  │
                       └────────────────────────────────────────────────────┘
```

Every long-running coroutine is wrapped in `supervised(name, factory)` which catches exceptions, logs them, and restarts with exponential backoff. No silent task deaths.

---

## Module map

### Core
| File | Purpose |
|---|---|
| [polyagent/main.py](polyagent/main.py) | Async entrypoint, task supervisor |
| [polyagent/config.py](polyagent/config.py) | Env-driven settings dataclass (~80 knobs) |
| [polyagent/supervisor.py](polyagent/supervisor.py) | Crash-resilient task wrapper |
| [polyagent/logging_setup.py](polyagent/logging_setup.py) | structlog JSON logs to stdout + file |

### Market data
| File | Purpose |
|---|---|
| [polyagent/gamma.py](polyagent/gamma.py) | Gamma `/markets` REST client, market parsing, `days_to_resolution` helper |
| [polyagent/ws_polymarket.py](polyagent/ws_polymarket.py) | WSS `/ws/market` client, auto-reconnect with book-state invalidation, queue-overflow drop-oldest |
| [polyagent/orderbook.py](polyagent/orderbook.py) | In-memory book reconstruction; methods for mid/bid/ask, **imbalance**, **momentum**, **realized_vol**, **last_update_ts** |
| [polyagent/data/clob_history.py](polyagent/data/clob_history.py) | CLOB `/prices-history` client (used by horizon backfill) |

### Storage / state
| File | Purpose |
|---|---|
| [polyagent/paper_broker.py](polyagent/paper_broker.py) | Cash + positions, **VWAP fill simulation** (walks the book), state recovery on restart, atomic settle via INSERT-OR-IGNORE, asyncio.Lock for concurrency, WAL+busy_timeout |
| [polyagent/news_store.py](polyagent/news_store.py) | News + signals SQLite store, content-hash dedup with LRU bound, source credibility weights, `news_match_p_yes` (time-decayed), `news_velocity`, `sentiment_zscore` |

### Models
| File | Purpose |
|---|---|
| [polyagent/models/embedder.py](polyagent/models/embedder.py) | Sentence-transformer (`BAAI/bge-large-en-v1.5`, 1024-dim, GPU) |
| [polyagent/models/cross_encoder.py](polyagent/models/cross_encoder.py) | `cross-encoder/ms-marco-MiniLM-L-12-v2` for news-market reranking |
| [polyagent/models/finbert.py](polyagent/models/finbert.py) | `ProsusAI/finbert` finance sentiment (replaces VADER) |
| [polyagent/models/lgbm.py](polyagent/models/lgbm.py) | K-fold CV LightGBM trainer + isotonic calibrator + `Predictor.predict_batch` |
| [polyagent/models/calibrator.py](polyagent/models/calibrator.py) | Isotonic regression + ECE metrics |
| [polyagent/models/features.py](polyagent/models/features.py) | Hand-coded question features (38 of them) — keyword lists, dollar/date/comparison/negation/proper-noun counts, time-to-resolution |
| [polyagent/models/word_features.py](polyagent/models/word_features.py) | Hashing vectorizer (legacy; embedder replaced it) |
| [polyagent/models/categorize.py](polyagent/models/categorize.py) | Question → category via max-hit keyword match |
| [polyagent/models/news_embed_matcher.py](polyagent/models/news_embed_matcher.py) | Two-stage news→market matcher: bge-large bi-encoder + cross-encoder rerank |
| [polyagent/models/outcomes.py](polyagent/models/outcomes.py) | `signal_outcomes` materialization (training labels) |
| [polyagent/models/retrain_loop.py](polyagent/models/retrain_loop.py) | Auto-retrains combiner when 50+ new full rows accumulate; quality-gated against forward log-loss regression |
| [polyagent/models/chronos.py](polyagent/models/chronos.py) | Chronos-Bolt zero-shot forecaster (opt-in, not enabled) |

### Signals
| File | Purpose |
|---|---|
| [polyagent/signals/stat.py](polyagent/signals/stat.py) | Periodic LGBM scoring; logs edge vs market mid |
| [polyagent/signals/news_match.py](polyagent/signals/news_match.py) | Semantic news → market matcher with directional output |
| [polyagent/signals/direction.py](polyagent/signals/direction.py) | FinBERT (default) or VADER + question-polarity heuristic |
| [polyagent/signals/combiner.py](polyagent/signals/combiner.py) | Log-pool weight fitting (SLSQP on log-loss) |
| [polyagent/signals/combined.py](polyagent/signals/combined.py) | Live combined signal: per-category log-pool, microstructure context (imbalance, momentum, news velocity, sentiment z-score) attached to every signal row |
| [polyagent/signals/combinatorial_arb.py](polyagent/signals/combinatorial_arb.py) | Nested-date monotonicity arb detector (e.g., "X by May 15" YES > "X by May 31" YES = arb) |

### Strategies
| File | Purpose |
|---|---|
| [polyagent/strategies/combined_trader.py](polyagent/strategies/combined_trader.py) | Fractional-Kelly trader on combined signal. Gates: spread filter, ask-depth filter, concentration cap, fee buffer, per-token + per-market cooldown, drawdown-aware Kelly, **drawdown-conditioned θ_min**, **Thompson-bandit per-category multiplier**, stale-quote skip, daily loss kill, adverse-selection blacklist |
| [polyagent/strategies/news_trader.py](polyagent/strategies/news_trader.py) | Lightweight news-driven trader; same gate set (less Kelly) |

### Risk
| File | Purpose |
|---|---|
| [polyagent/risk/throttle.py](polyagent/risk/throttle.py) | 30-day P&L + Sharpe per-strategy auto-throttler with persistent JSON state |
| [polyagent/risk/attribution.py](polyagent/risk/attribution.py) | Joins fills + resolutions for per-strategy realized P&L |
| [polyagent/risk/drawdown.py](polyagent/risk/drawdown.py) | High-water-mark tracker (persistent JSON) |
| [polyagent/risk/kill_switch.py](polyagent/risk/kill_switch.py) | `data/.STOP` file → halt all new orders (1s cache) |
| [polyagent/risk/adverse_selection.py](polyagent/risk/adverse_selection.py) | Token-level blacklist for markets with realized P&L worse than −10% over 7d |

### Background tasks
| File | Purpose |
|---|---|
| [polyagent/tasks_extra.py](polyagent/tasks_extra.py) | `stop_loss_loop` (sells positions down 40%+ from entry), `status_digest_loop` (5-min compact log), `signal_prune_loop` (daily DB cleanup) |
| [polyagent/trade_hunter.py](polyagent/trade_hunter.py) | **30s scanner** that ranks every quoted market by edge × category-throttle and dispatches the top-3 to the trader |

### Dashboard
| File | Purpose |
|---|---|
| [polyagent/dashboard.py](polyagent/dashboard.py) | aiohttp web server: `/`, `/api/summary`, `/api/positions`, `/api/fills`, `/api/nav-history`, `/api/resolutions`, `/api/health` |

### Ingest (8 sources)
[polyagent/data/rss.py](polyagent/data/rss.py), [bluesky.py](polyagent/data/bluesky.py), [fred.py](polyagent/data/fred.py), [bls.py](polyagent/data/bls.py), [congress.py](polyagent/data/congress.py), [courtlistener.py](polyagent/data/courtlistener.py), [sec_edgar.py](polyagent/data/sec_edgar.py), [alchemy.py](polyagent/data/alchemy.py), [resolution_watcher.py](polyagent/data/resolution_watcher.py)

### Scripts
| File | Purpose |
|---|---|
| [scripts/backfill_resolutions.py](scripts/backfill_resolutions.py) | Pull historical resolved markets from Gamma (~10K rows) |
| [scripts/backfill_market_prices.py](scripts/backfill_market_prices.py) | Fetch CLOB `/prices-history` for 4 horizons (1h, 6h, 24h, 7d pre-close) |
| [scripts/bootstrap_outcomes.py](scripts/bootstrap_outcomes.py) | Materialize `signal_outcomes` from resolutions (batched, ~20s for 10K rows) |
| [scripts/bootstrap_3expert_combiner.py](scripts/bootstrap_3expert_combiner.py) | Graft `news_match` weight onto 2-expert combiner |
| [scripts/train_lgbm.py](scripts/train_lgbm.py) | K-fold CV LGBM training |
| [scripts/train_combiner.py](scripts/train_combiner.py) | Single combiner (multi-expert) |
| [scripts/train_combiner_per_category.py](scripts/train_combiner_per_category.py) | Per-category combiner with forward-time holdout, regression gate |
| [scripts/eval_horizons.py](scripts/eval_horizons.py) | Sweep market-price horizons (1h/6h/24h/7d) and pick best by held-out log-loss |

---

## Models in detail

### LightGBM stat predictor

- **Training data**: 9,933 historically-resolved Polymarket markets (Gamma backfill, clean 1/0 resolution only — disputed/canceled excluded)
- **Features (~422 total)**:
  - 38 hand-coded question features: length, dollar/date/percent/number/proper-noun counts, comparison/negation flags, polarity diff, category one-hots, time-to-resolution, log liquidity/volume
  - 384-dim `bge-large-en-v1.5` sentence embeddings of the question (GPU)
  - (legacy: 256-dim hashing vectorizer; replaced by embeddings)
- **Training procedure**: 5-fold CV with isotonic calibration on out-of-fold predictions; final model trained on all data; saved as `data/lgbm_model.joblib` + `data/lgbm_model_calibrator.joblib`
- **Metrics (5-fold CV on 9,933 rows)**: log-loss 0.32, Brier 0.09, AUC 0.91, ECE_raw 0.04, **ECE_calibrated 0.0001**

### Combiner (per-category log-pool)

External Bayesian via log-pooled probabilities. For each category with ≥150 labeled outcomes, fit weights via SLSQP on log-loss with `LinearConstraint(ones, 1, 1)` and `Bounds(0, 1)`.

Current weights on the live combiner:

| Category | n | Weight: stat_lgbm | Weight: market | Weight: news_match | Held-out AUC |
|---|---|---|---|---|---|
| crypto | 686 | 0.74 | 0.27 | 0.10 (bootstrapped) | 1.00 |
| weather | 707 | 0.95 | 0.06 | 0.10 | 1.00 |
| sports_global | 617 | 1.00 | 0.00 | 0.10 | 1.00 |
| sports_us | 580 | 0.70 | 0.31 | 0.10 | 1.00 |
| politics_us | 463 | 0.77 | 0.23 | 0.10 | 1.00 |
| geopolitics | 179 | 0.83 | 0.17 | 0.10 | 1.00 |
| other | 3,780 | 0.97 | 0.03 | 0.10 | 1.00 |

Note: AUC=1.0 is partly leakage from 6h-pre-resolution market price (sports games are post-game by then). Real trade-time edge is smaller.

The retrain loop fires every 10 minutes, retrains when 50+ new fully-populated `signal_outcomes` rows have accumulated, and **only swaps the new bundle if forward-time held-out log-loss is no worse than the previous bundle**.

### News pipeline (semantic + finance-aware)

```
rss / bluesky / FRED / etc. → NewsEvent (deduped via SHA-256)
        ↓
SemanticMarketIndex.search(news_text)         # bi-encoder, top-20 by cosine
        ↓
cross_encoder.score_pairs([(news, market)])   # rerank, sigmoid → probability
        ↓
direction.classify(news, question)            # FinBERT compound × question polarity
        ↓
NewsStore.insert_signal()  →  News-trader  →  Combiner aggregator
```

### Direction classifier

| Layer | Source | Notes |
|---|---|---|
| Sentiment | `ProsusAI/finbert` (110M params, GPU) | ~700 events/sec batched. Replaces VADER which was finance-blind ("Bitcoin surges past 130k" was 0.0 to VADER, +0.52 to FinBERT) |
| Question polarity | Keyword heuristic | "out / fire / war / lose" → YES = negative event; "deal / win / pass" → YES = positive event |
| Final direction | sign(sentiment × polarity) gated by both meeting confidence floor | direction ∈ {yes, no, unknown} |

---

## Risk layer

Layered gates from outer to inner. Every BUY passes through all of these in order:

1. **Kill switch** (`data/.STOP` file) → blocks all new orders instantly
2. **Daily loss kill** per-strategy (default $250/day per strategy)
3. **Throttler multiplier** ∈ {0.0, 0.5, 1.0} based on 30-day realized Sharpe + P&L
4. **Adverse selection** — token blacklisted if realized P&L < −10% over 7 days
5. **Drawdown-conditioned θ_min** — edge threshold scales by `1 + 10 × drawdown_pct`
6. **Stale-quote skip** — refuse if book hasn't updated in 5 min
7. **Spread filter** — refuse if spread > 5pp
8. **Min ask depth** — refuse if ask depth × ask < $25
9. **Concentration cap** — global 30% of NAV
10. **Per-trade fee buffer** — subtract 0.5pp from |edge| before threshold check
11. **Per-token cooldown** — 120s between trades on the same token
12. **Per-market cooldown** — 600s between trades on the same condition
13. **Volume filter** — skip markets with < $1k 24h volume
14. **Drawdown-aware Kelly** — `f *= max(0.2, 1 − 5 × dd)`
15. **Thompson-bandit per-category multiplier** — Beta(α, β) sampled per trade
16. **Concentration warning** — log if any single position > 5% of NAV after fill

If everything passes, broker takes the trade as a marketable VWAP fill walking through the book.

---

## Storage layout

`data/paper.db` — SQLite, WAL mode, busy_timeout 10 s.

| Table | What's in it |
|---|---|
| `fills` | Every paper trade (BUY or SELL): ts, strategy, condition_id, token_id, side, price (VWAP), size, notional, reason |
| `nav_history` | NAV snapshots every 30 s (cash + position_value at mid) |
| `resolutions` | Settled markets: condition_id (PK), resolved_ts, yes_won, yes/no token IDs, sizes/costs at settle, payouts, pnl, detail JSON |
| `signal_outcomes` | One row per resolved market with all expert probabilities, used for combiner training |
| `signals` | Every emitted signal with strategy, direction, score, detail JSON. Pruned after 30 days. |
| `news` | Deduped news events (hash PK), source, ts, title, body, url |

Persistent state files:

| File | Purpose |
|---|---|
| `data/lgbm_model.joblib` | LGBM model + feature columns + base rate |
| `data/lgbm_model_calibrator.joblib` | Isotonic calibrator |
| `data/combiner.joblib` | v2 bundle: `{version, horizon, default, by_category, metrics}` |
| `data/drawdown.json` | High-water-mark across restarts |
| `data/throttle.json` | Per-strategy multipliers across restarts |
| `data/.STOP` | (touch this to halt) |

---

## Background tasks (all supervised)

| Task | Cadence | Job |
|---|---|---|
| `ws_market_stream` | continuous | Polymarket /ws/market subscriptions (chunks of 200) |
| `market_event_loop` | continuous | drain queue → BookStore.handle |
| `news_event_loop` | continuous | dedup, run matcher, log latency |
| `news_stats_loop` | 60 s | per-source counts in last 5 min |
| `status_loop` | 30 s | NAV snapshot + summary log line |
| `status_digest_loop` | 5 min | one-line compact health digest |
| `stat_signal` | 120 s | LGBM scoring with batched GPU predict |
| `combined_signal` | 120 s | Per-category log-pool with microstructure context |
| **`trade_hunter`** | **30 s** | rank every quoted market by edge, dispatch top-3 to trader |
| **`combinatorial_arb`** | **60 s** | Find nested-date YES violations |
| `resolution_watcher` | 120 s | Poll Gamma for held markets that have closed; settle |
| `throttler` | 5 min | Per-strategy Sharpe + P&L multiplier refresh |
| **`adverse_selection`** | **10 min** | Refresh token blacklist from realized P&L history |
| `stop_loss` | 60 s | Sell positions down 40%+ from entry |
| `retrain_loop` | 10 min | Retrain combiner if 50+ new outcomes; forward-time gated |
| `signal_prune` | 24 h | Drop signals > 30 days old |
| `dashboard` | continuous | aiohttp on :8080 |
| 8 ingest tasks | various | RSS, Bluesky, FRED, BLS, Congress, CourtListener, SEC EDGAR, Alchemy |

---

## How to run

```bash
# Activate venv (created in .venv/)
.venv/Scripts/python.exe -m polyagent.main > data/run.log 2>&1
```

```bash
# Watch
tail -f data/run.log

# Status snapshot
.venv/Scripts/python.exe -c "
import urllib.request, json
print(json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/health').read()))
"

# Halt new orders without killing the bot
touch data/.STOP

# Resume
rm data/.STOP
```

To rebuild from scratch after code changes that affect the model:

```bash
# 1. Pull historical markets (2 min, 25K markets)
.venv/Scripts/python.exe -m scripts.backfill_resolutions --max 25000

# 2. Train LGBM with current features (1 min on GPU)
.venv/Scripts/python.exe -m scripts.train_lgbm

# 3. Materialize stat_lgbm column (20 s, batched)
.venv/Scripts/python.exe -m scripts.bootstrap_outcomes

# 4. Backfill horizon prices (3 min, hits CLOB API)
.venv/Scripts/python.exe -m scripts.backfill_market_prices

# 5. Train per-category combiner with forward-time holdout
.venv/Scripts/python.exe -m scripts.train_combiner_per_category --horizon p_market_6h

# 6. Graft news_match weight (~10%)
.venv/Scripts/python.exe -m scripts.bootstrap_3expert_combiner

# 7. Launch
.venv/Scripts/python.exe -m polyagent.main > data/run.log 2>&1
```

---

## Implementation timeline (what was built, in order)

1. **Phase 1 — paper trading scaffold**: Polymarket WSS client, BookStore, PaperBroker, YES+NO arb strategy.
2. **Phase 2 — news + macro ingest**: 23 RSS feeds, Bluesky, FRED, BLS, Congress, CourtListener, SEC EDGAR, Alchemy. Plus a SQLite news store with content-hash dedup.
3. **Phase 3 — direction classifier**: VADER sentiment + question-polarity heuristic; news_trader strategy.
4. **Phase 4 — settlement + resolution_watcher**: Gamma polling for closed markets, atomic settlement with idempotency.
5. **Phase 5 — statistical layer**: LightGBM K-fold + isotonic calibration; ~5K backfilled resolutions.
6. **Phase 6 — combiner**: Log-pool over stat_lgbm + market price + news_match; per-category weights; horizon sweep.
7. **Phase 7 — closed-loop retraining**: signal_outcomes materializer, retrain_loop with forward-time quality gate.
8. **Phase 8 — risk layer**: Throttler, drawdown tracker, kill switch, stop_loss, kelly bandit, etc.
9. **Phase 9 — bug audit + WAL**: Found and fixed task supervisor (was silent on crash), DB lock contention, broker state recovery, race conditions, settle idempotency.
10. **Phase 10 — dashboard + UI**: aiohttp server, single-page HTML with live JSON polling.
11. **Phase 11 — drop yes_no_arb + arb_scanner**: Polymarket spreads are tight; pure book arbs basically don't exist.
12. **Phase 12 — 20 model improvements**: kill switch, drawdown-aware Kelly, spread/depth/concentration gates, fee buffer, token cooldown, stop-loss task, volume filter, time-to-resolution feature, dollar-amount feature, news source credibility, persistent throttle, /api/health, status digest, signal prune, daily loss kill, stale-quote skip, position warnings, etc.
13. **Phase 13 — GPU + transformers**: PyTorch nightly with sm_120 (Blackwell) support, sentence-transformer (bge-large), cross-encoder reranker, FinBERT direction classifier.
14. **Phase 14 — semantic news matching**: replaced keyword Jaccard with bi-encoder + cross-encoder pipeline.
15. **Phase 15 — TradeHunter agent**: 30 s scanner ranking all quoted markets by edge, with edge cap (0.50) to suppress hallucinations.
16. **Phase 16 — combinatorial arb + microstructure**: nested-date violation detector, book imbalance, momentum, news velocity, sentiment z-score, Thompson-bandit Kelly, drawdown-conditioned θ_min, adverse-selection filter, VWAP fill simulation, latency profiling.

---

## Trading strategies in scope

| Strategy | Status | Notes |
|---|---|---|
| **combined_trader** | active | Fractional-Kelly on log-pool combined signal. Main edge source. |
| **news_trader** | active | Light-touch trades on direct news matches. |
| **combinatorial_arb** | detector live, dispatch wired | Surfaces nested-date violations; routed to combined_trader |
| **stop_loss** | active | Capital protection, not edge |
| **yes_no_arb** | **removed** | Doesn't exist on Polymarket — spreads too tight |

---

## Ideas NOT yet implemented

These were identified in literature/research but skipped intentionally. Ranked by my honest read on signal:effort.

### High value but blocked

#### 1. Liquidity-rewards market maker
- Polymarket pays daily LP rewards via a quadratic kernel (close to mid → more reward). Reported $200–300/day on 10 K USDC at peak.
- **Why not built**: requires real maker order placement on-chain with EIP-712 signing + USDC funding. Paper-only mode doesn't earn the rewards (no real liquidity provided). And per safety policy I can't sign or fund on-chain orders.
- **What it'd unlock**: a passive yield stream independent of directional alpha. Probably the highest single-source EV on Polymarket today.
- **Effort**: ~500 LOC + wallet integration.

#### 2. Real-money trading
Not an "improvement" per se — but the actual gating constraint. Without it, *any* alpha we have is unrealized. Steps you'd need:
- Polymarket account + Polygon Safe + USDC
- `set_allowances.py` script (sketched in v2 blueprint, not implemented)
- Real `clob_client.py` using `py-clob-client` instead of PaperBroker
- Order management (partial fills, cancels, retries)
- On-chain redemption via CTF Adapter on settlement
- Latency profiling against real Polymarket origin (target < 500 ms wire-to-order)

### Medium value, real lift

#### 3. Full subgraph adverse selection
- Query Polymarket's Goldsky subgraph for every counterparty wallet on our trades; build a "smart money" registry; downweight markets where smart money is on the other side.
- **Why not built**: lightweight in-process version (using our own fills history) catches the easier cases for free. The full version flags markets *before* we lose money there.
- **Effort**: ~200 LOC, GraphQL client, wallet PnL caching.

#### 4. Multi-LLM probability consensus
- Ensemble of GPT-4/Claude + LGBM probabilities for highest-conviction markets
- **Why not built**: marginal gain (the existing LGBM is already calibrated; ensemble correlation is high), real cost per inference
- Where it'd help: questions with novel structure that don't match our historical training distribution
- **Effort**: 100 LOC + API costs

#### 5. LLM event verification (Qwen-7B local)
- For each news → market match, run a small LLM to verify "does this news genuinely affect this market?"
- **Why not built**: news_trader fires sparingly already (only 4 fills lifetime); LLM filter would catch edge cases at high VRAM cost (~14 GB)
- Becomes valuable if news_trader scales up
- **Effort**: 150 LOC + 14 GB VRAM commitment

### Lower urgency

#### 6. Wash-trade filter
- Polymarket research found a 22% upper-tail wash-trade share. Filter markets with elevated wash before sizing.
- Needs Goldsky subgraph access for wallet-level trade history.
- **Effort**: ~100 LOC

#### 7. Block-clock latency profiling against real wire times
- We log news pipeline latency (ingest-to-decision). We don't measure wire-to-decision (the WSS event arrival time vs the source event time).
- Useful only when the bot trades in the seconds-to-milliseconds window. Currently we trade on 30 s+ windows so latency edge is dead anyway.

#### 8. Per-event clustering via Gamma's `event_id`
- Many markets share an `event_id` (e.g., 50 NBA-Champion sub-markets). Sum-of-YES across an exhaustive partition should ≈ 1.
- Combinatorial arb already catches the nested-date case via question-prefix grouping, but Gamma's structured event_id would catch much more (different teams in same event, different outcomes of same race, etc.).
- **Effort**: ~150 LOC

#### 9. Chronos-Bolt time-series forecaster
- We added the module ([polyagent/models/chronos.py](polyagent/models/chronos.py)) but never wired it as a combiner expert. Would predict near-term price drift from the last N hours of `_mid_history`.
- **Effort**: 50 LOC to wire as expert in combiner; Chronos is already importable

#### 10. Real-time book imbalance feature in the LGBM model itself
- We compute imbalance at runtime in `combined.py` and store it in `signals.detail` for analysis, but the LGBM is question-only (no live state). To add imbalance as a real LGBM feature, we'd need imbalance histories at training time per market — which means extending the historical backfill to capture book snapshots, not just end-state outcomes.
- Heavy data-engineering lift (~500 LOC + months of book archival).

#### 11. Reinforcement learning for sizing
- Replace fractional Kelly with a contextual-bandit or DRL-trained policy network.
- Useful only after we have thousands of resolved trades to train on. Currently we have ~13.
- **Effort**: large — months of data + training pipeline

#### 12. Multi-stage classifier (longshot vs favorite first, per-bucket model)
- Train a router that picks "longshot model" vs "favorite model" first, then routes
- The favorite-longshot bias paper says low-price contracts (≤10¢) win less than implied; high-price (≥90¢) slightly more. A per-bucket model could exploit this directly.
- Current LGBM has this implicitly through `q_has_dollar` + market price embeddings, but explicit two-stage might help.
- **Effort**: ~200 LOC

#### 13. Maker / passive limit orders (in addition to taker)
- For the most liquid markets, post limit orders 1 tick inside the spread instead of crossing the spread on a take. Earns the spread instead of paying it.
- Requires real on-chain orders (same blocker as #1).

### Out of scope / not pursued

- **Kalshi cross-venue arbitrage** — the user explicitly excluded.
- **Truth Social / 4chan ingest** (in v2 blueprint) — too noisy at our scale; the existing 23 RSS sources + Bluesky + FRED/BLS already cover most signal.
- **Vector-search retraining of embedder** — bge-large is already strong; fine-tuning would need labeled (question, news) pairs which we don't have.

(Telegram ingest *was* added later — see "Session log: May 2026" below.)

---

## Session log: May 2026 — research-backed redesign

This section captures three intense weeks of work after the initial build:
literature surveys, ideas implemented, ideas deferred, and a candid record
of three losing paper-trading sessions and what they revealed.

### Major modules added

| File | Purpose |
|---|---|
| [polyagent/queue_model.py](polyagent/queue_model.py) | Cont/Kukanov/Stoikov queue-aware fill probability + Polygon block (~2s) cancel-latency adverse-drift via Brownian σ√Δt; pessimistic_fill_price for shadow ledger. |
| [polyagent/risk/latency_model.py](polyagent/risk/latency_model.py) | Empirical per-source latency tracker (`by_source` dict). p99 gate refuses trades on books staler than 5× our p99. |
| [polyagent/risk/smart_money.py](polyagent/risk/smart_money.py) | Goldsky subgraph top-PnL maker-wallet registry. Refresh every 6h; tightens θ_min on sensitive categories when ≥50 wallets known. |
| [polyagent/risk/wash_filter.py](polyagent/risk/wash_filter.py) | Dubach 2026 wash-trade hygiene: trade-without-book-change ratio per token; blacklist when share > 30% over 30+ samples. |
| [polyagent/risk/live_ece.py](polyagent/risk/live_ece.py) | Live ECE drift monitor on rolling 30-day vs 365-day resolved markets; complements PSI. |
| [polyagent/signals/combinatorial_arb.py](polyagent/signals/combinatorial_arb.py) | NegRisk sum-to-1 violation detector (IMDEA: $29M of arb on Polymarket Apr 2024–Apr 2025 came from this). Fallback monotonicity scanner for non-NegRisk events. |
| [polyagent/signals/consistency_check.py](polyagent/signals/consistency_check.py) | Karkare NegRisk consistency: LLM forecasts each outcome, deviation from sum-to-1 downweights `llm_forecaster` expert. |
| [polyagent/signals/local_edge.py](polyagent/signals/local_edge.py) | LocalEdgeClassifier — LLM pre-filter "does this market have non-English / insider info edge?" Cached forever. Routes capital to news_match + llm_forecaster on local-edge markets, downweights stat_lgbm. |
| [polyagent/signals/message_market_match.py](polyagent/signals/message_market_match.py) | Per-news-event LLM matcher emitting `(market_id, confidence, direction, reason_short)` rows. Adapted from `takakhoo/Polymarket_Agent`. |
| [polyagent/strategies/passive_poster.py](polyagent/strategies/passive_poster.py) | Maker-side paper-mode simulation. Posts inside the spread; queue-model-based per-cycle fill probability with cancel-latency penalty when realized vol is high. |
| [polyagent/data/telegram.py](polyagent/data/telegram.py) | TDLib listener with hard read-only allowlist (asserted by guard test). Opt-in via `TELEGRAM_API_ID/HASH/PHONE/CHANNELS` env vars. |
| [polyagent/data/telegram_planner.py](polyagent/data/telegram_planner.py) | Multilingual handle planner — LLM enumerates `(actors, countries, bridge languages, official/journalist/community handles)` per market. Static seed list (IDF, Pikud HaOref, Iran Intl, etc.) when LLM disabled. |
| [polyagent/models/article_retriever.py](polyagent/models/article_retriever.py) | Hybrid BM25+dense retrieval with Reciprocal Rank Fusion for the LLM forecaster. Replaces pure-recency LIMIT 8. |
| [polyagent/models/calibrator.py](polyagent/models/calibrator.py) | Three-tier calibrator: isotonic ≥80, **Venn-Abers ≥30** (Vovk/Petej; native conformal interval), Beta ≥15 (Kull et al. 2017), global isotonic fallback. `transform_with_interval` returns `(p_point, p_low, p_high)` triple consumed by conformal-Kelly. |
| [polyagent/models/llm_forecaster.py](polyagent/models/llm_forecaster.py) | Phi-4-mini retrieval-augmented forecaster. Adds `_aia_debias()` (acquiescence + round-number unsticking from Karger et al. 2025). N-sample geometric-odds aggregation. `consistency_score()` for NegRisk groups. |
| [polyagent/models/lgbm.py](polyagent/models/lgbm.py) | Default objective switched from `binary` to **`lambdarank`** with day-of-resolution as group key (Poh et al. 2021 ~3× Sharpe). CV switched from StratifiedKFold to **CPCV with market-id purging** (de Prado). |
| [tests/test_data_clients_readonly.py](tests/test_data_clients_readonly.py) | Static guard tests: greps `polyagent/data/{telegram,alchemy,clob_history}.py` for forbidden mutating method names. 4/4 pass. |

### Settings added
~30 new env-var-overridable knobs. Highlights:
- `combined_min_ask` (default $0.10) — longshot floor
- `combined_max_ask` (default $0.85) — fade favorite floor
- `combined_fee_buffer` (default 2pp) — half-spread buffer
- `combined_kelly_mult` (default 0.075) — half-Kelly
- `combined_max_per_trade_kelly` (default 0.025)
- `passive_poster_per_post_notional` (default $12) — halved after diagnostic
- `passive_poster_max_total_notional` (default $150) — halved
- `enable_message_market_matcher`, `enable_passive_poster`, `enable_wash_filter`, `enable_combinatorial_arb`
- `TELEGRAM_API_ID/HASH/PHONE/CHANNELS` (opt-in)
- `ENABLE_LLM_FORECASTER` (default 0; flip to 1 to activate planner+classifier+matcher)

### Ideas implemented from the literature

From a 12-idea ranked research report:

| # | Idea | Source | Status |
|---|---|---|---|
| 1 | DPO self-play fine-tune of Phi-4 | Turtel et al. 2025 (arXiv 2502.05253) | Deferred (6-12h GPU run) |
| 2 | AIA acquiescence + round-number debiasers | Karger et al. 2025 (arXiv 2511.07678) | **Shipped** in `_aia_debias()` |
| 3 | Venn-Abers replacing isotonic on thin cells | Manokhin 2025 / Vovk-Petej | **Shipped** as new tier in `CellCalibrator` |
| 4 | ColBERTv2 late-interaction reranker | Stanford-FutureData | Deferred (storage/eng cost) |
| 5 | Hybrid BM25+dense+RRF retrieval | 2025 RAG benchmarks | **Shipped** in `article_retriever.py` |
| 6 | Multi-level OFI (deep-tick) | Kolm/Turiel/Westray 2023 | Already had basic OFI in `orderbook.py` |
| 7 | TabPFN v2.5 as 5th expert | Hollmann et al. Nature 2025 | Deferred (training-pipeline change) |
| 8 | Mantic-style RL fine-tune | Thinking Machines Nov 2025 | Deferred (GPU cost) |
| 9 | Conformal lower-bound Kelly sizing | Vovk 2025 / ICLR | **Shipped** — Predictor exposes `calibrated_low/high`; trader sizes off lower bound |
| 10 | NegRisk LLM-extracted dependencies | Saguillo et al. AFT 2025 | Already had basic NegRisk arb scanner |
| 11 | Chronos-2 / Moirai-MoE for price trajectory | Various 2025 | Deferred (likely OOD on binary CLOB) |
| 12 | FinCast token-level sparse-MoE gating | Liang et al. 2025 | Deferred (architectural lift) |

### Ideas implemented from `takakhoo/Polymarket_Agent` survey

| # | Idea | Status |
|---|---|---|
| 1 | Multilingual Telegram-handle planner | **Shipped** (`telegram_planner.py`) |
| 2 | Local-edge classifier as LLM pre-filter | **Shipped** (`local_edge.py`); tilts log-pool weights |
| 3 | Per-message → market matcher with `(confidence, direction, reason_short)` | **Shipped** (`message_market_match.py`) |
| 4 | Read-only safety guard test for data clients | **Shipped** (`tests/test_data_clients_readonly.py`) |
| 5 | TDLib similar-channel graph walk | Skipped (operational complexity) |
| 6 | Handpicked-markets workflow | Skipped (manual, doesn't fit automated bot) |

### Loss diagnostics & fixes

Three losing sessions, each revealed a different layer of the same
fundamental issue: **a question-only ML model cannot beat an efficient
prediction market on direction** (Della Vedova SSRN 6191618, Akey et al.
SSRN 6443103, Yang Augusta 2026). The market's Brier on resolved markets
is ~0.05–0.10; ours is ~0.13–0.18 — **2–3× worse**. The gap masquerades
as "edge" until execution costs eat it.

#### Session 1 — −3.7% (longshot disaster)

Empirical:
- 31% of `combined_signal` events on markets with `p_market < 0.05`
- Model claimed **+24.8pp average edge** on every single one (impossible)
- 28% of all BUYs were on tokens under $0.10
- 22 stop-losses fired on cheap longshots dropping 40% (which is just 1-tick noise on $0.05 books)
- Slippage burn: 22% of all notional deployed

Diagnosis: question-only LightGBM regresses every prediction toward the training base rate (23.4%), so anywhere `p_market` is far from 23.4% the model "disagrees". That gap isn't edge — it's the model's residual error. Whelan (2024) Economica + GW Kalshi 2026: sub-10¢ contracts return −60% on average.

Fixes shipped:
- **Longshot floor** `combined_min_ask = $0.10` — refuse trades below it
- **Edge-sanity cap** `|edge| ≤ min(p_mkt, 1−p_mkt)` — bigger gap than the market's prior is statistically impossible from an AUC=0.77 model
- **Market-prior shrinkage in log-pool** — runtime enforces `p_market` weight ≥ 60% regardless of trained weights (Akey 2026's "shrink toward market" recipe)
- **Half-Kelly + 2pp fee buffer + tighter daily/per-trade caps**
- **Longshot-aware stop-loss**: 70% drop threshold for tokens < $0.10 (avoids tripping on 1¢ noise)
- **Pessimistic-NAV deployment gate** at 100 fills with 5% slippage-burn block

#### Session 2 — −2% (mid-priced averaging-down)

After Session 1's fixes the longshot pathology vanished, but the bot still lost 2% with 49 fills + 7 stops. Audit:

- `trade_hunter.py` was bypassing the new market-prior shrinkage (used raw trained weights)
- Same token bought 4× at the same price as it fell, then stop-lossed, then **bought back** within 60s ("falling knife")
- 5pp/15% averaging-down threshold too loose

Fixes shipped:
- **Market-prior shrinkage applied in `trade_hunter.py` too** (mirror of `combined.py`)
- **Stop-loss re-entry blacklist (24h)** — `broker.recently_stopped[token_id]` checked before every BUY
- **Averaging-down guard tightened** to 3% relative drop / 2pp absolute
- SQLite `busy_timeout` 10s → 30s, retry budget 1.4s → 12s (was producing supervisor-restart noise)

#### Session 3 — −5% (passive_poster cycling)

After Session 2's fixes, the longshot AND averaging-down patterns were both gone. New vector emerged: `passive_poster` cycling fills on the same token because the price didn't actually drop between fills.

- One token: **24 passive_poster fills totaling $170**
- Another: 28 fills cycling between combined_trader and passive_poster at $0.30
- The averaging-down guard checks `mid < avg_cost × 0.97` — when price stays flat between fills, the guard doesn't engage; we just keep piling on
- `passive_poster` had no per-token notional cap — only a global $300 cap

Fixes shipped:
- **Hard broker-level fill cap**: `max_buys_per_token_window = 2` per 24h, applies to every strategy
- `passive_poster` sizing halved: `per_post_notional $25→$12`, `max_total_notional $300→$150`, `max_concurrent 8→5`
- Fill counts tracked per-token in `broker._buys_per_token` with rolling-window expiry

### Risk gate stack (current, ordered)

Every BUY passes through this gate sequence. Order matters: cheaper checks first.

1. Kill switch (`data/.STOP`)
2. Pessimistic-NAV deployment gate (≥5% slippage burn / starting NAV over last 100 fills)
3. **Per-token fill cap** (broker-level, ≤2 BUYs / 24h / token)
4. Per-strategy daily loss kill ($250)
5. Volume24h floor ($1,000)
6. Edge-sanity cap (`|edge| ≤ min(p_mkt, 1−p_mkt)`)
7. Fee-adjusted edge ≥ θ_min × (1 + 10 × drawdown)
8. Per-market cooldown (600s)
9. Per-token cooldown (120s)
10. Adverse-selection blacklist
11. **Stop-loss re-entry blacklist (24h)**
12. **Averaging-down guard (3% drop / 2pp absolute)**
13. Smart-money tighten on geopolitical categories (×1.5 θ if ≥50 known wallets)
14. Wash-trade hygiene blacklist
15. Stale-quote skip (5 min)
16. **Latency p99 gate** (refuse if book age > 5× p99 of WSS source)
17. Best-ask in `[combined_min_ask=0.10, combined_max_ask=0.85]`
18. Spread ≤ 5pp
19. Min ask depth ≥ $25
20. Full Kelly > 0
21. Auto-throttler multiplier > 0
22. Drawdown-conditioned Kelly scale
23. Bandit-Thompson per-category multiplier
24. Per-trade Kelly cap (2.5% NAV) + notional cap ($30)
25. Per-market notional cap ($100)
26. Per-category concentration (≤30% NAV)
27. Daily notional cap ($500)
28. Final notional ≥ $1
29. Final size ≥ ε

Plus passive_poster has its own ~12 gates (eligibility, post-price sanity, smart-money tilt, recently-stopped block, averaging-down, fill-count cap, max_concurrent, max_total_notional, etc.).

### Honest performance ceiling

Per the research, a question-only model on Polymarket has a structural
ceiling near zero ROI even with all gates applied. Realistic targets:

- **Brier**: ~0.09–0.11 with B+C+D+9 fixes from the literature report
- **ROI**: roughly flat, ±2% per session is normal noise
- **Where positive ROI lives**: maker-side spread capture (Yang 2026: $121/market making vs $63 taking) and NegRisk arbitrage (IMDEA: $29M/$40M of all Polymarket arb 2024–2025) — neither of which is a directional model bet

The bot is now structurally honest about this — it sizes off conformal
lower bounds, blocks every documented loss vector, and **doesn't pretend
to have edge it can't have**. Improvements from here come from richer
data (the Telegram listener for non-English geopolitical wires, the
DPO-fine-tuned LLM forecaster) or from execution edge that requires
real-money infrastructure.

---

## Limitations / honest caveats

- **Polymarket is reasonably efficient**: arb opportunities last ~2.7 s in 2026 per the literature. Our 30 s scanner captures *structural* edges (combinatorial arb, base-rate priors), not latency-arb.
- **AUC = 1.0 per-category is partly leakage**: the held-out market price was 6 h pre-resolution, which on sports markets is post-game. Real trade-time AUC on long-horizon markets will be substantially lower.
- **~120 fills is a tiny sample**: realized P&L of +$75 is closer to noise than signal. Need 500+ resolutions over weeks before claiming the bot has edge.
- **Calibration is on backfilled markets only**: live distribution may shift. Forward-time gate in retrain_loop catches the worst regressions but isn't perfect.
- **Paper VWAP fills overstate real fills by ~half-spread**: real CLOB fills face queue-position uncertainty, partial fills, latency. Our paper P&L is the upper bound.
- **No on-chain redemption**: in real trading, settlement requires a `redeemPositions()` tx via the CTF Adapter (~$0.001 in gas). Not modeled here.
- **State persistence is paper-DB-local**: a hard disk failure would lose history. No off-site backup.
- **Single-process bot**: no horizontal scaling, no failover. Restart wipes any in-flight signals not yet committed.

---

## Where to look next

If you wanted to spend another week:

1. **Stop adding features.** Run the bot for 2–4 weeks and accumulate ≥500 resolved trades. The data dump from that is more valuable than another tuning pass.
2. **Build a clean weekly performance report**: per-strategy realized Sharpe over time, hit rate by category, the calibration plot of combined probability vs realized YES rate. Currently we have the raw data, not the report.
3. **Decide real-money or done.** If real: someone other than me has to handle wallet/keys/EIP-712. If done: this is a pretty good educational project as-is.

If you wanted to spend another month, the LP market maker is the single highest-EV item — but blocked on real-money setup.
