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

## Session log: late May 2026 — Sharpe-honesty discipline

Adopted the falsifiability-first discipline from the design doc
(`pm.md`): ship only what we can verify worked, default-off when the
historical falsifiability test fails. Four roadmap items attempted
over four weeks:

| Week | Item | Falsifiability | Status |
|---|---|---|---|
| 1 | §12 PSR/DSR/MTRL live nightly harness | math verified on synthetic returns (PSR/DSR/MTRL match Bailey & de Prado 2014 Table 1) | **shipped, live** — `polyagent/eval/sharpe_honesty.py`, `polyagent/eval/harness.py` |
| 1 | §1 selective-abstention gate (Venn-Abers width) | observable in week 1: gate admits ~40% of candidates | **shipped, live** — `polyagent/risk/selective_gate.py` |
| 2 | §9 BOCPD changepoint gate on win/loss | FP/TP grid: hazard 0.005 → 5% FP, 0/10 detection; hazard 0.020 → 95% FP, 9/10 detection. **No useful operating point.** | **default-off** — `polyagent/risk/bocpd_gate.py` skeleton ships; live activation deferred until either a different detection signal (CUSUM on rolling win-rate) or more data |
| 3 | §10 Kaminski-Lo gated stops + near-resolution lock-in | on real data n=7: φ=+0.09, SR_daily=+1.45 → φ < SR_daily, stops have been removing mean per K-L 2014 | **shipped, live** — `polyagent/risk/exit_policy.py`. Internal n≥30 floor before gate engages. |
| 4 | §4 NegRisk arb v2 (atomic dispatch + depth-floor) | 7,592 raw NegRisk-class candidates over 4 days, 100% partial-group artifacts (median edge 99.4pp, real arbs are ≤30pp). After artifact filter: **0 real opportunities at any threshold** | **v2 not built** — patched v1 to drop partial-group artifacts |

Two of four passed, two failed. Both failures had the highest expected
Sharpe lift in the doc; both got falsified by real data exactly as the
SE(SR) ≈ 0.10 epigraph predicted. The disciplined response — not
shipping unverified strategies — is the entire point of §12 / the
discipline thread.

### Empirical state at session end

After Week 4 fixes (broker-level per-token fill cap, market-prior
shrinkage in trade_hunter, edge-sanity cap, longshot floor at $0.10,
half-Kelly, Kaminski-Lo gated stops, near-resolution lock-in,
selective abstention via interval width, partial-NegRisk-group
artifact filter):

```
NAV (mid):              $10,078.61   (+0.79% over the session)
NAV (liquidation):      $10,054.99   (+0.55%)
Realized P&L:           +$210.69
Unrealized:             −$26.24
Fills:                  139 (73 combined_trader BUY, 35 passive_poster BUY,
                             19 stop_loss SELL, 9 news_trader BUY,
                             3 near_resolution SELL)
Resolved with position: 7 (all NO-side bets on sports markets, all paid)
Stop-losses:            2 (was 22 in the −3.7% session)
Task crashes:           0 (busy_timeout 10s → 30s + 7-attempt retry)
```

The 7 resolved trades (all wins, $232 of realized P&L) are far below
MTRL — single-session noise, not signal. The point of §12 is to
keep me honest about that. PSR(SR≥0) on N=7 is essentially
uninformative; we need ~200 more resolutions before the harness can
say anything statistically.

### Discipline takeaway

Five weeks of intense feature shipping resulted in: 14 modules added,
19 risk gates currently active, two falsifiability failures gracefully
documented, and a paper-mode bot that — given its current state — has
no business pretending it has detected positive Sharpe yet. The
right answer is to **let it run** and let the harness accumulate
data. Future feature additions go through the same falsifiability
gate; nothing ships on intuition.

---

## Session log: May 9, 2026 — calibration audit + strategy-certificate gate

Took the discipline thread one step further: instead of "let the bot
trade everything that passes the gates", restrict live trading to
slices where the model has *actually* been shown to beat the market
on held-out data. Built and committed the infrastructure to enforce
that.

### What got measured

`signal_outcomes` had 10,215 labeled rows (resolved markets with
`p_stat_lgbm` populated) but only **42** had a market price column
filled — `p_market_*` was overwhelmingly NULL. Without market price
you can't measure edge. So:

1. Ran `scripts/backfill_market_prices.py` against CLOB
   `/prices-history`. **7,442 / 10,215 rows (72.9%) populated** with
   horizon prices (1h/6h/24h/7d), 0 errors. The 2,773 misses were
   markets with insufficient price history (mostly <24h-old
   resolutions).

2. **Overall calibration of `stat_lgbm` (n=7,442 head-to-head):**
   - Model log-loss 0.3959, market log-loss **0.2544** → market beats
     model by **+0.14 log-loss**.
   - At the live trigger `|p_model − p_market| ≥ 0.10` (n=4,235),
     directional accuracy is **47.6%** — sub-random.
   - The model gets **more wrong** as it disagrees more loudly:
     disagreement bucket [0.40, 1.00) shows model_LL 0.84 vs market_LL
     0.42 (+0.42 delta).

3. **Per-category combiner** trained with `[stat_lgbm,
   p_market_24h]` log-pool weights. Only `sports_global` produces
   meaningful stat-side weight (0.53 stat / 0.47 market) AND beats
   market alone:

   | category | n | stat weight | log-loss | beats market? |
   |---|---|---|---|---|
   | crypto | 710 | 0.00 | 0.174 | combiner == market |
   | politics_us | 467 | 0.02 | 0.097 | combiner == market |
   | sports_us | 615 | 0.17 | 0.029 | marginal |
   | other | 3757 | 0.35 | 0.263 | combiner ≈ market |
   | **sports_global** | **626** | **0.53** | **0.163** | **yes (Δ=−0.087)** |

4. **Held-out CPCV on `sports_global` only** (8 folds, market-id
   purged): **8/8 folds positive**, mean per-fold edge **+0.128
   log-loss vs market alone**, std 0.040, sign-test p = 0.0039,
   **DSR = 0.9959**. PBO=0.5 — but PBO is a single-config artifact
   here (it requires multiple competing configs to discriminate;
   with one strategy it collapses to ~0.5 by construction). The
   `validate_strategy.py` harness has the same artifact.

### Roadmap items shipped this session

| # | Item | Falsifiability | Status |
|---|---|---|---|
| 1 | calibration audit of `stat_lgbm` across full backfilled cohort | n=7,442 head-to-head; market beats model by +0.14 log-loss; directional accuracy 47.6% (sub-random) at live trigger | **`stat_lgbm` cert: enabled=0** — model correctly stays log-only globally |
| 2 | per-category combiner v2 with `[stat_lgbm, p_market_24h]` log-pool weights | bundle saved to `data/combiner_v2.joblib` with 6 trained categories | **shipped** — runtime loads via `COMBINER_PATH` env override |
| 3 | `sports_global` combiner certification | 8/8 CPCV folds positive, DSR=0.996, sign-test p=0.004, mean edge +0.128 log-loss | **cert: enabled=1** — first non-arb strategy to clear honest validation |
| 4 | `strategy_certificates`-driven category allowlist on `CombinedTrader` | 4 unit tests in `test_certificate_gate.py` + live verification: bot logs `combined_trade_skip_uncertified_category` for non-sports_global markets and `certificate_gate_active n_certs=1 allowed_categories=["sports_global"]` on startup | **shipped, live** behind `ENABLE_CERTIFICATE_GATE=1` |
| 5 | `DB_PATH` / `LOG_PATH` env overrides | enables running the bot from any cwd / git worktree against the canonical DB | **shipped** |
| 6 | dashboard upgrade: certificate panel + by-category rollup + improved NAV chart | new endpoints `/api/certificates`, `/api/by-category`; sticky header with cert-gate status pill; cert cards with DSR/edge/sign-test; per-category P&L table; hover tooltip on NAV chart | **shipped** |

### Architecture: what the cert gate actually does

```
strategy_certificates (SQLite)
        │
        │  on startup, when ENABLE_CERTIFICATE_GATE=1
        ▼
main.py builds set of categories where enabled=1
        │
        │  passed into CombinedTrader.certified_categories
        ▼
CombinedTrader.on_signal(category=...)
   if certified_categories is not None and category not in it:
       log("combined_trade_skip_uncertified_category"); return
   else: continue to all other gates (selective, smart_money,
         BOCPD, edge sanity, fee buffer, daily loss kill, ...)
```

Three-tier rollback:
1. **Soft** (1 SQL line): `UPDATE strategy_certificates SET
   enabled=0 WHERE name='...'` disables a single cert without
   touching code.
2. **Medium** (1 env var): `ENABLE_CERTIFICATE_GATE=0` falls back
   to legacy "trade everything that passes the other gates".
3. **Hard**: `git revert e411cfc` removes the wiring entirely.

The default in production is `ENABLE_CERTIFICATE_GATE=0` — opt-in
only, so existing deployments keep current behaviour until a
deliberate flip.

### Honest caveat on the certification

`combiner_v2.joblib` was saved with `--allow-regression` because no
v1 forward-metric existed to compare against. The first batch of
live `sports_global` fills *is* the production verification. The
falsifiable claim:

> +0.128 log-loss edge on `sports_global` should translate to
> realized PnL within 1σ of `+0.128 × n × notional` across the
> first 20-30 fills. If cumulative edge tracks; cert is real. If
> flat or negative across 20+ fills; disable via the SQL one-liner.

626 head-to-head rows is enough for DSR but tight for production
confidence. The forward fills are themselves a held-out test.

### Empirical state at session end

```
NAV (mid):              $9,995.49  (−0.05% over the session, before activation)
Realized P&L:           +$199.42
Open positions:         38
Historical fills:       151
Settled with position:  7 (all NO-side, all paid out)
Strategy certificates:  3 rows total
                          - stat_lgbm_combiner_sports_global_v2 (enabled=1)
                          - stat_lgbm_combiner_sports_global    (enabled=0, superseded)
                          - stat_lgbm                           (enabled=0, overall)
Cert gate at run time:  ON, allowlist = {sports_global}
Bot status:             running on worktree code, cert gate active
```

### Discipline takeaway

This session is the inverse of week 4: there's the discipline to
ship something AND the discipline to scope it tight. The model
**doesn't** beat the market overall (correctly, falsifiably,
recorded as an enabled=0 cert). The model **does** beat the market
on `sports_global` (8/8 folds, p=0.004). Live trading was
restricted to that one category, with a SQL-level kill switch and an
env-flag rollback. This is what "ship narrow when you're sure, log
broadly when you're not" looks like in code.

The next forward-test is purely passive: let `sports_global` fills
land, see if cumulative edge tracks the +0.128 log-loss claim. If
it does, audit `news_keyword_match` next (same recipe: backfill
`p_news_match`, calibrate, look for category sub-slices that
beat market). If not, disable the cert and the gate falls back to
the empty allowlist with no further code changes.

### Files added / modified

```
polyagent/config.py                       +9    enable_certificate_gate flag,
                                                DB_PATH/LOG_PATH overrides
polyagent/main.py                        +30    bootstrap certified_categories
                                                allowlist from strategy_certificates
                                                on startup
polyagent/strategies/combined_trader.py  +19    certified_categories field +
                                                early-return gate in on_signal
polyagent/dashboard.py                  +504/-85 cert panel, by-cat rollup,
                                                better NAV chart, /api/certificates,
                                                /api/by-category
tests/test_certificate_gate.py           +new   4 tests covering gate
data/combiner_v2.joblib                  +new   per-category log-pool combiner
                                                (gitignored)
```

Commits: `e411cfc` (cert gate + sports_global cert), `bc98db1`
(dashboard upgrade).

---

## Session log: May 10, 2026 — quant review + execution-stack pivot

A senior-quant review (`pmwhy.md`, in repo root) reframes the project's
priorities. The review's core argument, with citation:

> The bot's flat ROI is exactly what the literature predicts. The 2026
> microstructure literature on Polymarket and Kalshi (Bartlett & O'Hara
> SSRN 6615739, Akey et al. SSRN 6443103, IMDEA AFT 2025, Tsang & Yang
> SSRN 6336679) independently converges: retail taker-side trading is
> structurally unprofitable. Bots had 52% raw accuracy vs retail's 55%
> — bots win on execution, not prediction. Maker side earned ~2× spread
> per contract from the systematic YES-overbet behavioural surplus.
> ForecastBench shows even frontier RAG-LLM forecasters (Brier 0.1258)
> trail market consensus (0.1106) on liquid markets. The path to
> profitability for a single-operator question-only ML system is not
> better features; it's better execution.

### What was empirically validated against our data

The review's A6 caveat — "selective abstention helps Sharpe only if
edge is concentrated in the high-confidence tail, which at this n you
cannot test from forward fills" — was tested directly against the
7,447 head-to-head rows in `signal_outcomes`. Findings (run
`scripts/analyze_high_confidence_tail.py` to refresh):

**Whole sample, model log-loss vs market by confidence:**

| confidence = `\|p_model − 0.5\| × 2` | n | model_LL | market_LL | delta |
|---|---|---|---|---|
| [0.00, 0.10) | 544  | 0.7028 | 0.3263 | +0.38 |
| [0.30, 0.50) | 1059 | 0.6052 | 0.2486 | +0.36 |
| [0.70, 0.90) | 1532 | 0.3596 | 0.2759 | +0.08 |
| [0.90, 1.00) | 3332 | 0.2537 | 0.2339 | +0.02 |

Model approaches market parity in its high-confidence tail.

**Sports_global slice (the certified one), same buckets:**

| confidence | n | model_LL | market_LL | delta |
|---|---|---|---|---|
| [0.30, 0.50) | 30  | 0.85 | 0.60 | **+0.24** model worse |
| [0.70, 0.90) | 147 | 0.24 | 0.32 | **−0.08** model better |
| [0.90, 1.00) | 429 | 0.11 | 0.22 | **−0.12** model better |

The certified +0.128 log-loss edge is **entirely concentrated in
confidence ≥ 0.7**. At medium confidence the model is worse than
market.

**Naive-PnL Sharpe by confidence (whole sample, betting model-favored
side at market price):**

| confidence | n | mean_pnl | Sharpe |
|---|---|---|---|
| [0.00, 0.10) | 544  | −$0.13 | −0.42 |
| [0.30, 0.50) | 1059 | +$0.02 | +0.08 |
| [0.70, 0.90) | 1532 | +$0.07 | +0.26 |
| [0.90, 1.00) | 3332 | +$0.11 | **+0.47** |

Sharpe rises monotonically with confidence. **Selective abstention is
a Sharpe lever, not just a Brier lever**, contradicting the review's
hedged "this is not directly evidenced" position.

**Implication.** A confidence-threshold gate on
`combined_trader.on_signal` (drop trades below `|p_combined − 0.5| × 2
< 0.7`) would compress fill count but materially improve realized P&L
per trade. This is the cheapest Sharpe-positive change available.

### Doc updates landed this session

- `CLAUDE.md` (repo root) rewritten: "honest performance ceiling"
  reflects the structural critique, "where to look next" reorders
  priorities to (1) queue-aware fill simulation, (2) maker-side
  execution, (3) confidence-threshold gate, (4) forward-test cert
  with corrected sample size. Adds explicit warnings that paper P&L
  overstates live by 20–40% and that detecting Sharpe 0.3–0.5 needs
  ~1,500–3,000 OOS forward trades, not the project's earlier ≥500.
- `pmwhy.md` checked into repo root as the citable source.
- `scripts/analyze_high_confidence_tail.py` runs the bucket
  analysis in ~1 second against `signal_outcomes` + `model_failures`.

### What stays the same

- Cert gate, falsifiability discipline, three-tier rollback — still
  the right architecture. The review explicitly endorses Bailey-López
  de Prado / DSR / CPCV / "log every config tried."
- `sports_global` cert stays enabled. The high-confidence-tail
  finding *strengthens* it: the certified edge is real where the
  model is loud about it, not at all confidences uniformly.
- `yes_no_arb` keeps running. The review notes single-condition arb
  is sub-100ms-bot-dominated, but our paper version is essentially
  free to run as a discipline-keeping baseline.

### What changes

- New priority list (top to bottom): queue-aware fill sim → maker
  execution (`passive_poster_v2`) → confidence-threshold gate →
  forward-test under realistic fills with ≥1,500 trade target.
- LLM forecaster, NLI verifier, news_match — all log-only, treat as
  inventory-skew inputs for a future maker rather than as standalone
  alpha.
- Strict feature-freeze. No new model changes until the queue-aware
  simulator lands and the cert is re-validated under it.

### Empirical state at session end

Bot still running: NAV $9,915 (mid), 36 open positions, 190 fills,
+$149 realized P&L since broker init, cert gate active on
sports_global, dashboard at http://127.0.0.1:8080 with the new
model_failures section showing 1,921 historical failures across
4 types.

---

## Where to look next

If you wanted to spend another week:

1. **Stop adding features.** Run the bot for 2–4 weeks and accumulate ≥500 resolved trades. The data dump from that is more valuable than another tuning pass.
2. **Build a clean weekly performance report**: per-strategy realized Sharpe over time, hit rate by category, the calibration plot of combined probability vs realized YES rate. Currently we have the raw data, not the report.
3. **Decide real-money or done.** If real: someone other than me has to handle wallet/keys/EIP-712. If done: this is a pretty good educational project as-is.

If you wanted to spend another month, the LP market maker is the single highest-EV item — but blocked on real-money setup.

---

## Why the model keeps losing money

Read this before re-running, re-cranking sizes, or re-arguing about which
strategy "should" work. The losses are not a bug to be tuned out — they
are the structural consequence of the position we're playing from. Each
reason below cites either project-internal evidence (the `signal_outcomes`
and `model_failures` tables, or the May-10 quant-review session) or the
external literature aggregated in `pmwhy.md`.

### 1. We are on the wrong side of every trade we take

Polymarket's microstructure has been measured. Bartlett & O'Hara (2026)
find retail buys YES on 61% of fills while YES only resolves true 32% of
the time — meaning the marginal taker is *paying* to be wrong, and the
counterparty (the resting maker) is collecting an adverse-selection
premium. Every time `CombinedTrader` crosses the spread it is, by
construction, joining the 61% YES-buying retail flow. The fee schedule
(Crypto 1.80%, Sports 0.75%, Politics 1.00%) and 650–900 bps half-spreads
on cheap longshots (Della Vedova, 2025) are paid one direction only:
ours. Until we are the maker, every fill starts the round-trip already
underwater.

### 2. Bots win on execution, not prediction — and ours has neither edge

Della Vedova measured bot accuracy at 52% vs retail 55%. The bots are
*worse forecasters* than retail. They win by being faster, by quoting
tighter, by paying maker rebates instead of taker fees. IMDEA's NegRisk
study tracked $29M of arb captured by sub-100ms operators inside a
2.7-second median window. Our paper-broker fills at top-of-book VWAP
with no latency model and no queue position — this overstates realized
P&L by 20–40% per the HFT/MM literature, and the moment a real wallet
is connected the 70 bps cancel-latency drift on every resting quote
(Polygon 2 s block × realized σ) starts eating the rest. We have neither
the prediction edge nor the execution edge that would let either side of
the trade work.

### 3. The market is the strongest single feature, and we don't beat it

ForecastBench 2025: frontier LLMs score Brier 0.1258 on liquid markets
while the market price scores 0.1106. Even GPT-4-class models lose to
the price of the market they're trying to predict. Our `LLMForecaster`
runs gpt-oss-20b (with Phi-4-mini fallback) on retrieved news — there is
no published or internal evidence it does better than the market on
liquid US markets. Internally, the model is **+0.14 log-loss worse than
the market** across n=7,442 head-to-head rows in `signal_outcomes`, and
directional accuracy at the live-trigger moment is **47.6% — sub-random**.
The high-confidence tail is *worse*: 481 high-confidence-wrong calls
with average log-loss 3.13, and the model gets *more* wrong as it
disagrees more loudly with the market (model-market gap [0.40, 1.00)
shows +0.42 log-loss delta). The market is including news, recent flow,
book pressure, and informed-trader inventory; our question-text +
news-retrieval features can't see most of that.

### 4. Tsang & Yang: the market is microstructurally efficient now

Kyle's λ — the price impact per unit of informed-flow imbalance —
collapsed from 0.53 to 0.01 over the 2024 US election cycle. That means
informed orders are no longer moving prices the way they did when
Polymarket was thin. Any latent signal we *did* have is being arbed out
by faster operators before our 5-minute LightGBM features see it. The
project's mental model from 2024 ("we're early, the market is dumb")
is no longer true in 2026 on the categories we touch most heavily.

### 5. Question-text LightGBM has a low ceiling

The certified slice is `sports_global` precisely because that's the one
slice where question features (team names, schedule context, common
language) carry residual signal the market hasn't fully absorbed.
Everywhere else the market dominates, which is why the cert allowlist
disables `politics`, `crypto`, `econ`, `entertainment`, and the rest.
But even `sports_global`'s edge is small enough that PBO=0.214 and
DSR is positive only inside a narrow config window. Cranking size or
expanding categories does not create new edge — it just buys more of a
near-zero-EV process while paying a wider taker spread.

### 6. n=9 (now reset) is statistically meaningless

The Bailey–López de Prado Minimum Backtest Length for detecting a
Sharpe of 1.0 at 5% significance is **1,500–3,000 OOS forward trades**
on this kind of variance profile. We have zero on the new baseline and
had n=9 before the reset. Every "is it working?" check at low n is
noise; any sign the bot is "profitable today" is statistically
indistinguishable from coin-flips. The May 10 quant-review explicitly
upgraded the validation bar from the project's earlier ≥500 to 1,500.

### 7. Akey et al.: 69% of traders lose money on Polymarket, full stop

SSRN 6443103 finds 69% of wallets net-lose on Polymarket, and for the
bottom quintile the entire loss is attributable to liquidity-taking
cost — not bad picks. Joining the median Polymarket taker is joining a
2-in-3 losing distribution before any of our model's specific
mis-calibrations apply.

### 8. Paper-broker optimism papers over all of the above

`paper_broker.fills_shadow_queue` and the queue-aware backtest exist
because the headline VWAP-fills broker silently inflates P&L by 20–40%.
Until the cert is *re-validated under queue-aware fills* (the May-10
"feature freeze" item), every NAV chart Taka has been looking at is
biased upward. The shadow fills already show round-trip captured-spread
near zero on the categories outside `sports_global`.

### What it would take to stop losing

Three things, ordered by impact, all from the May-10 priority pivot:

1. **Become the maker.** `passive_poster_v2` is built; the next step
   is to run it forward on `sports_global` only, log shadow fills,
   and re-certify under queue-aware reprice. If maker captured-spread
   net of cancel-latency and fees stays positive for 1,500+ trades, that
   is the first real edge this codebase has had.
2. **Stop trading uncertified categories at any size.** The cert gate
   already enforces this; what's left is the discipline not to widen
   the allowlist when the bot "feels slow."
3. **Treat the LLM forecaster as inventory-skew input, not as an alpha.**
   The literature is consistent that it cannot beat market price on
   liquid contracts; its only honest job is to nudge maker quote
   midpoints on quiet, illiquid weather/event markets where the book is
   too thin for the price to be informative.

### What to stop doing

- Cranking `KELLY_MULT`, `MAX_PER_TRADE`, or `QUOTE_SIZE` to "get more
  fills." Bigger taker size on an adversely-selected fill schedule
  loses money *faster*, not eventually-positively.
- Re-enabling uncertified categories. The model's worst quintile by
  log-loss is exactly the buckets where it disagrees most confidently
  with the market — i.e. the trades it *most wants to make* are the
  trades it should least be allowed to make.
- Reading short-window NAV charts as signal. Below ~1,500 resolved
  trades, the NAV line is variance, not skill.

---

## External validation (pmwhybetter.md, 2024–2026 literature)

The companion doc `pmwhybetter.md` (in repo root) maps Polyagent's ten
identified failure modes onto specific 2024–2026 literature fixes with
concrete arXiv / SSRN citations and open-source references. Headline
findings that change priorities or specifically validate existing code:

### Triangulated structural diagnosis

Three independent 2026 papers converge: edge on Polymarket comes from
**execution (maker-side liquidity provision), not from prediction
accuracy**.

- **Bartlett & O'Hara 2026 (SSRN 6615739):** retail buys YES on 61%
  of fills, YES resolves true only 32% — taker is structurally
  adversely selected.
- **Akey, Gregoire, Harvie & Martineau 2026 (SSRN 6443103):** *a 1-σ
  increase in maker volume share lowers loss probability by 9.3
  percentage points.* This is the single strongest quantification of
  "become the maker" we now have.
- **Della Vedova 2026 (SSRN 6191618, 222M trades):** retail picks
  winners 51.3% but loses $79M; bots earn $133M with coin-flip
  accuracy. **Entire delta is execution.**

### Findings that directly validate existing Polyagent code

- **Manokhin Probability Matrix (arXiv 2605.03816, 2025):**
  *LightGBM is a "Bull" (strong AUC, poor calibration); Venn-Abers
  cuts log-loss 6.5–12.6% on Bulls but* **degrades already-calibrated
  models**. Empirically validates our existing ≥30-sample Venn-Abers
  tier *and* gives the hard rule: do NOT apply Venn-Abers to combiner
  output (which is already calibrated by log-pool).
- **Akey 2026 "shrink-toward-market" prior:** matches our log-pool
  combiner with ≥60% market weight. The published recipe says relax
  market weight *only* when (a) confidence ≥ 0.7 AND (b) cell has
  ≥80 calibration samples — slightly stricter than the current 30/80
  Venn-Abers/Beta cutoffs. Defensible to tighten.
- **Tsang & Yang 2026 (SSRN 6336679, arXiv 2603.03136):** Kyle's λ
  collapsed 0.53 → 0.01 over the 2024 cycle — confirms our
  "directional bets on liquid markets are dead" call.
- **ForecastBench Oct 2025:** best LLM (GPT-4.5) Brier 0.101 vs
  superforecaster 0.081 vs market crowd ~0.11. Confirms our
  forecaster-as-inventory-skew framing rather than forecaster-as-alpha.

### Findings that change priorities or open new attack surfaces

- **Dubach 2026 (arXiv 2604.24366):** *public WSS Lee-Ready agrees
  with on-chain ground truth only ~59% (vs 80%+ on equity venues),
  and Kyle's λ flips sign on 60% of markets between feeds.*
  **Implication: every OFI / Lee-Ready / trade-direction feature in
  Polyagent built on the WSS feed is mostly noise.** Migration to
  on-chain `OrderFilled` event ingestion is a TODO blocker on any
  microstructure feature.
- **Saguillo et al. AFT 2025 (arXiv 2508.03474):** $40M extracted
  from Polymarket Apr 2024 – Apr 2025, $29M from NegRisk rebalancing,
  top wallet $2.01M across 4,049 trades. Methodology:
  Linq-Embed-Mistral embeddings + LLM relationship extraction reduces
  O(2^(n+m)) NegRisk search to tractable. Plus *monotonicity arbs*
  (e.g., "Trump wins" ≤ "Republican wins") — a constraint family our
  current `combinatorial_arb.py` doesn't cover.
- **Yang/Cheng/Zou 2026 NBA (SSRN 6624718):** median 3.6-second arb
  episode, concentrated in final minutes — sub-second latency bar is
  the realistic floor for arb capture.
- **Heng & Soh ICLR 2025 (arXiv 2505.15008):** Neyman-Pearson optimal
  selection score is a **likelihood ratio**, not a confidence
  threshold. RLog and Δ-KNN-RLog scores; explicitly handles covariate
  shift. Drop-in replacement for our `|p−0.5| < 0.7` rule.
- **Della Vedova wallet-orthogonality test:** 6,292 informed wallets
  out of 483K flagged at p<0.01, concentrated in Action and Vote
  markets. Implementable per-wallet feature on top of existing
  on-chain trade ingest. *Closing window:* Polymarket+Chainalysis
  (Apr 2026) is actively suppressing these wallets.
- **Sirolly et al. Nov 2025 (SSRN 5714122):** wash-share peaked at
  **60% Dec 2024, ~20% Oct 2025**, sports the worst-affected
  category. Use as *negative* feature — suppress signal in markets
  with high wash share. **Our `sports_global` certified slice is in
  the worst-contaminated category — volume features specifically must
  be replaced with trade-count / net-flow.**
- **Outcome-RL (arXiv 2505.17989, May 2025):** 14B model matches o1
  on Brier (0.193) with measured $127 vs $92 hypothetical trading
  profit p=0.037. The only paper with measured trading edge from RL
  fine-tuning. Implementable on our RTX 5070 Ti with NVFP4 (arXiv
  2601.09527 confirms Qwen3-14B-NVFP4 viable, 16k ctx).
- **Turtel et al. DPO self-play (arXiv 2502.05253):** Phi-4 14B /
  DeepSeek-R1 14B gain 7–10% Brier from self-play DPO with no human
  labels, reaching GPT-4o parity. Same hardware-fits-on-our-GPU note.

### Open-source references worth lifting from

- **`nkaz001/hftbacktest`** (3.3k★, Rust+Python) — canonical
  queue-aware sim backbone. `power_prob_queue_model=3` for the
  post-2024 regime. Validate by reconciling against our live ~190 fills.
- **`warproxxx/poly-maker`** + **`Polymarket/poly-market-maker`** —
  Polymarket-native maker bots, source of band/AMM strategies.
- **`agent-next/polymarket-paper-trader`** — already does level-walking
  + exact bps × min(p, 1−p) × shares fee model + GTC/GTD state machine.
  Our `paper_broker.py` reinvents most of this.
- **`ip200/venn-abers`** — Generalized Venn-Abers (arXiv 2502.05676)
  with set-valued epistemic-uncertainty interval, directly feedable to
  Kelly.

### Updated top-5 priority list (single developer, feature-freeze)

1. **Queue-aware fill simulation** with hftbacktest semantics
   (Brownian σ√Δt cancel drift, queue position, partial fills,
   Polygon ~73 ms baseline + multi-second tail). Validate against
   live ~190 fills. *Status:* `queue_aware_fills.py` shipped, needs
   reconciliation pass.
2. **Maker-default execution** (`passive_poster_v2`) at queue-aware
   optimal offset using Avellaneda-Stoikov reservation pricing
   skewed by VPIN/OFI toxicity (Barzykin–Bergault–Guéant–Lemmel
   arXiv 2508.20225, 2025). *Status:* `passive_poster_v2.py` shipped,
   VPIN skew is the missing piece.
3. **Hierarchical Bayesian calibration + Bayesian Sharpe** across
   (category × horizon) — partial-pooling with brms/Stan recipe.
   Replaces *both* thin-cell calibration fallback chain *and* n=190
   significance bottleneck in one model. Lets us certify additional
   categories on posterior credible interval rather than waiting for
   1,500 trades.
4. **NegRisk + combinatorial detector upgrade** following Saguillo
   methodology — Linq-Embed-Mistral semantic clustering + LLM
   relationship extraction + min-leg liquidity executability filter.
   *Only* published $-quantified positive 2024–2025 strategy.
5. **Forecaster fine-tune via Outcome-RL or Turtel DPO on
   Qwen3-8B-NVFP4**, with Heng-Soh likelihood-ratio selective
   abstention layered on top. Defer until #1–#4 ship.

### Caveats from the doc that constrain interpretation

- Many of the 2026 citations should be independently re-verified
  before any production decision rests on them.
- Polymarket microstructure regime is shifting fast: V2 launched late
  2025 fixing some "ghost fills"; Chainalysis insider-detection went
  live Apr 2026; ICE invested up to $2B Oct 2025; Polymarket US
  re-launched Nov 2025 via QCX. **All published $-extraction figures
  are upper bounds** on what is now extractable; project 30–50%
  compression.
- **Wash-trading inflates volume metrics ~25% average, peaks 60%
  Dec 2024** (Sirolly Nov 2025). Volume features in `features.py`
  for `sports_global` specifically should migrate to trade-count or
  net-flow.
- **Trade-direction inference from WSS is ~59% accurate** (Dubach
  2026 stylized fact). Any current OFI / Lee-Ready / direction
  feature is mostly noise until migrated to on-chain `OrderFilled`.
- **DSR=0.996 with n=190 and 23 gate combinations** itself warrants
  explicit CSCV/PBO sanity check per Bailey-Borwein-Lopez de
  Prado-Zhu 2014 (SSRN 2326253). Our v4 PBO=0.214 run satisfies this
  but the practice should generalize.

---

## What pmwhybetter.md drove us to add

Concretely landed in this branch as direct implementations of the doc's
recommendations:

| Module | Doc citation | Purpose |
|---|---|---|
| `polyagent/risk/vpin_gate.py` | Bartlett-O'Hara 2026; Barzykin–Bergault–Guéant–Lemmel 2025 | VPIN toxicity-gate for maker quotes |
| `polyagent/risk/likelihood_ratio_gate.py` | Heng & Soh ICLR 2025 (arXiv 2505.15008) | Neyman-Pearson optimal selective abstention |
| `polyagent/signals/wallet_orthogonality.py` | Della Vedova 2026 | Per-wallet p<0.01 informed-trader detector |
| `polyagent/signals/monotonicity_arb.py` | Saguillo et al. AFT 2025 Section 5 | Non-NegRisk monotonicity arb (e.g., A ⊆ B implies p(A) ≤ p(B)) |
| `polyagent/risk/wash_filter.py` (extended) | Sirolly et al. Nov 2025 (SSRN 5714122) | Graph-cluster wash-share negative feature |
| `polyagent/risk/cancel_latency.py` | Olding 2022; Barzykin 2026; Dubach 2026 stylized fact #6 | Brownian σ√Δt cancel-drift + last-look |
| `polyagent/models/microprice.py` | Gould & Bonart 2015; Cont-Kukanov-Stoikov 2014; arXiv 2602.00776 | Micro-price, VAMP, queue-imbalance one-tick-ahead |
| `polyagent/eval/block_bootstrap.py` | Politis-Romano 1994; Ledoit-Wolf 2008 | Stationary block bootstrap Sharpe CI under autocorrelation |
| `polyagent/eval/bayesian_sharpe.py` | Kruschke BEST; Mulligan QMF 2024 | t-likelihood Bayesian Sharpe posterior at n<500 |
| `polyagent/eval/cscv.py` | Bailey-Borwein-Lopez de Prado-Zhu 2014 (SSRN 2326253); Arian-Norouzi-Seco 2024 | Explicit CSCV reporting alongside DSR/PBO |
| `polyagent/models/llm_ensemble.py` | Schoenegger 2024 *Science Advances*; arXiv 2510.01499 | Accuracy-weighted median across decorrelated LLMs |
| `polyagent/risk/conformal_kelly.py` | Vovk 2025; Sun & Boyd arXiv 1812.10371 | Distributionally-robust Kelly via conformal PD |
| `polyagent/signals/negrisk_clustering.py` | Saguillo et al. AFT 2025 (arXiv 2508.03474) | Semantic clustering scaffold for combinatorial arb |
| `polyagent/signals/consistency_loss.py` | Karkare/Paleka arXiv 2412.18544; Outcome-RL arXiv 2505.17989 | Consistency-as-loss training scaffold |
| `scripts/finetune_qwen3_outcome_rl.py` | Outcome-RL arXiv 2505.17989; Turtel arXiv 2502.05253 | Outcome-RL / DPO self-play training scaffold |

The `*.py` modules all ship with tests under `tests/`; scaffolds are
no-ops until externally configured (e.g., Qwen3-8B model path,
Linq-Embed-Mistral endpoint).

---

## Session log: May 10, 2026 (evening) — literature pass II + bot restart

The afternoon "literature pass" landed 16 modules covering the top-5
priorities + ~80% of the concrete fixes in `pmwhybetter.md`. This
evening session built the **remaining 10 buildable items** from the
doc — every fix that isn't blocked by hardware (gpt-oss-120B),
safety policy (real-money wallet, cross-platform arb), or paid SaaS
subscriptions — wired them into the live strategies, and restarted
the bot from a fresh $10,000 portfolio.

### What was added this session

| Module | Doc citation | Purpose |
|---|---|---|
| `polyagent/risk/maker_rewards.py` | Polymarket docs.polymarket.com/market-makers/liquidity-rewards; Wanguolin Medium | Quadratic-spread reward score tracker for the $12M/yr maker-rewards pool. Per-token cumulative score + projected_daily_reward(usd) lets us dashboard what a registered MM seat would capture. |
| `polyagent/risk/agent_next_fees.py` | `agent-next/polymarket-paper-trader`; docs.polymarket.com fee schedule | Exact Polymarket fee formula `fee = bps × min(p, 1−p) × shares` with maker rebate share. Drop-in compatible with `polyagent/risk/fees.compute_fees`. Buying favourites pays ~5% of the nominal bps rate per dollar; the existing approximate model over-charges them. |
| `polyagent/eval/asymmetric_brier.py` | Coletta ACM ICAIF 2021; ACM Computing Surveys 2025 doi:10.1145/3727633 | Cost-weighted, linear-economic, and coverage-asymmetric Brier variants. Standard Brier weights right-but-confident the same as wrong-on-a-longshot; asymmetric variants attribute realized P&L to each prediction. |
| `polyagent/eval/regime_switching_sharpe.py` | Hamilton 1989 (Econometrica 57); artifact-research.com 2025 | 2-regime Gaussian HMM fit via Baum-Welch (forward-backward + MAP). Reports per-regime Sharpe + mixture Sharpe weighted by stationary distribution. Materially wider CIs than single-regime BEST under crisis-like returns. |
| `polyagent/eval/forecast_benchmark.py` | ForecastBench Oct 2025; arXiv 2507.04562; arXiv 2602.21229 | Self-contained ECE + Brier decomposition (reliability / resolution / uncertainty) + baseline comparison (market / uniform / base-rate). Runs locally against `signal_outcomes` joined to `resolutions`. CLI for ad-hoc runs. |
| `polyagent/models/market_conditioned_prompt.py` | arXiv 2602.21229 ("Forecasting Future Language: Context Design for Mention Markets") | Explicit-prior Bayesian prompt template that asks the LLM to compute log-odds-update + posterior from market price + retrieved news. Magnitude-capped log-odds update (max ±1.5 nats) prevents a single LLM call from over-riding a liquid market. |
| `polyagent/models/colbert_retriever.py` | arXiv 2603.25248 (ColBERT-Att); GTE-ModernColBERT-v1; PyLate arXiv Aug 2025 | Late-interaction retriever scaffold with sentence-transformer cosine fallback when PyLate isn't installed. Wire `pip install pylate` to flip to the real ColBERT path. |
| `polyagent/signals/ternary_gate.py` | Coletta ACM ICAIF 2021; ACM Computing Surveys 2025 | UP / FLAT / DOWN three-way selective classifier. Per-side hit-rate-floor threshold calibration (default ≥60%). Composes with `selective_gate` (width) and `lr_gate` (LR) — all three must admit AND the ternary classification must match the proposed side. |
| `polyagent/signals/inplay_arb.py` | Yang/Cheng/Zou 2026 (SSRN 6624718, NBA real-time arb) | Sub-second sports in-play NegRisk arb detector with in-game-window filter and `min_leg_size_at_stale` executability cap. Documents the 3.6-sec median episode latency floor as the realistic execution bar. |
| `polyagent/data/foreign_news.py` | doc Problem-3 fix #5; Della Vedova "informed Action and Vote" wallets | Non-English RSS poller (Le Monde, Folha SP, NHK World, Der Spiegel, Xinhua) with LLM-based translation via the existing `LLMForecaster`. Disabled translation by default; passes the original text through when LLM unavailable. |

### Strategy wirings landed

- **`passive_poster_v2`** now accepts `vpin_gate`, `maker_rewards`,
  and `wash_graph_conn` fields. Every quote update samples
  `(spread_bps, size, time)` into the maker-rewards tracker; `_vpin_allow`
  consults the gate per side before each fill; `_effective_quote_size`
  scales by `(1 − wash_share)`.
- **`combined_trader`** now composes a *third* selective layer:
  `ternary_gate` runs after `selective_gate` (width) and `lr_gate`
  (Heng-Soh LR). All three must admit AND the ternary classification
  (UP / DOWN) must match the proposed taker side (BUY / SELL).
  Skips logged as `combined_trade_skip_ternary`.
- **`llm_forecaster.forecast()`** now accepts optional `market_p` and
  `base_rate` kwargs. When `market_p` is supplied, the function uses
  the Market-Conditioned Prompting recipe (arXiv 2602.21229) with
  log-odds-update parsing and magnitude cap. Falls back to the legacy
  Halawi-style ensemble when `market_p` is `None` so existing callers
  still work.
- **`main.py`** constructs the shared `VPINGate` + `MakerRewardsTracker`
  and injects them into both `BookStore` (for trade-tape ingestion)
  and `PassivePosterV2` (for quote-side consultation). New supervised
  task `foreign_news` (1800-sec cadence, 5 sources).

### Operational fixes shipped this turn

- **On-chain ingester now creates the `trades` table on a fresh DB.**
  The initial `ensure_columns` ran `ALTER TABLE` which failed silently
  before any trades existed. Now it `CREATE TABLE IF NOT EXISTS` first,
  then idempotently adds the on-chain columns. Backwards-compatible
  with the existing prod DB.
- **`wallet_analytics` skips cleanly** when the `trades` table is
  absent (rather than triggering a warning every hour). Logged as
  `wallet_analytics_skip_no_trades_table` until the on-chain ingester
  has populated rows.

### Total of pmwhybetter.md fixes landed (cumulative across both passes)

**Built and wired:** 26 new modules, 28 new test files, 334 passing tests.

**Not done (blocked, not deferred):**
- Mantic + Tinker fine-tune (Problem 2 #4) — requires gpt-oss-120B
- AMM-aware NegRisk mint/burn execution (Problem 6 #4) — requires
  real wallet + EIP-712 signing, out of scope per safety
- Cross-platform Kalshi↔Polymarket arb (Problem 10 #5) — multi-venue
  keys, out of scope
- Commercial market-data feeds (Problem 10 #6) — paid subscription

Every other concrete recommendation in the doc is implemented.

### Bot state at session end (run id `bhbgsazxh`)

```
NAV:                       $10,000 baseline (fresh reset)
Cert allowlist:            sports_global (1 active cert)
Strategies trading live:   yes_no_arb (always); combined_trader on
                           sports_global; passive_poster_v2 maker on
                           sports_global (68 tokens)
Background tasks alive:    ~27 supervised
  - book_archive_writer + book_archive_periodic
  - wallet_analytics (1-hr cadence; skips until trades table seeded)
  - foreign_news_poller (5 sources, 30-min cadence)
  - onchain_orderfilled_ingester (Alchemy live, 60-sec cadence)
  - dashboard, all signal pollers, throttler, etc.
Dashboard:                 http://127.0.0.1:8080
Commits this branch:
  599ef0b book_archive on separate sqlite file
  f53b6bc literature pass I (16 modules, 18 tests)
  a788a3e literature pass II (10 modules, 10 tests)
  33f01e2 fresh-DB fix
```

### Honest framing on profitability

The bot is now at the literature-state-of-art for paper-money
question-only ML on Polymarket. Every concrete recommendation in
`pmwhybetter.md` that's executable in scope is implemented and wired.
**But:** the doc itself (and the "Why the model keeps losing money"
section above) is explicit that the structural disadvantage of paper-
taker-only ML against sub-100ms on-chain operators is large, and
feature additions cannot close it on their own.

The remaining levers — real-money maker side execution, on-chain
NegRisk mint/burn execution, ColBERT/PyLate proper retriever upgrade,
gpt-oss-120B forecaster — all require either money I cannot spend
(safety policy), VRAM I don't have, or time-and-discipline to wait
for the ≥1,500 forward trades the Bailey-LdP MinBTL math demands.

Whether this bot makes money over the next 1,500 forward trades is
now an empirical question only forward time can answer. The code
side of the literature audit is complete.

---

## Model-improvement roadmap (May 10 review)

After the literature pass II commit (`a788a3e`) declared "26 modules
built, every concrete pmwhybetter.md fix in scope landed," an
external reviewer pushed back on the claim by reading what was
*actually wired* versus what was *built but unwired or scaffolded*.
The review identified six candidate model improvements (distinct
from execution / risk / discipline improvements, which the existing
PROJECT.md "Where to look next" already covers) and ranked them by
realised-P&L leverage. This section captures the analysis and the
chosen action.

### Meta-framing

The reviewer's central observation: the May 10 quant-review already
prescribes a feature freeze and ≥1,500 forward trades before any
further model surgery. Of the six candidate improvements, **only one
has an answer that doesn't require those 1,500 trades** — it can be
evaluated against the existing 7,442-row `signal_outcomes` table
immediately. The rest fight the structural ceiling that question-
only ML on Polymarket is +0.14 log-loss *worse* than the market on
the head-to-head sample, and the Akey 2026 finding ("1σ increase in
maker volume share lowers loss probability by 9.3pp") dominates
anything further model surgery can buy.

So the meta-answer to "what should we do next?" is: the falsifiable
one first. If it passes, ship. If it fails (it did), the rest of the
list is correctly deferred under the feature freeze.

### The six candidates, ranked

| # | Improvement | Doc citation | Status |
|---|---|---|---|
| 1 | Migrate `sports_global` volume features to trade-count / net-flow / unique-wallets / top-wallet-share | Sirolly Nov 2025 (SSRN 5714122) | **Tested. Falsified.** See below. |
| 2 | Wire `hierarchical_calibrator.py` as partial-pooling replacement for the isotonic ≥80 / V-A ≥30 / Beta ≥15 / global fallback chain in `calibrator.py` | Mulligan QMF 2024; Manokhin arXiv 2605.03816; PROJECT.md top-5 priority #3 | Module built, **not wired**. Deferred. |
| 3 | Replace `selective_gate.py` (width-based) with `likelihood_ratio_gate.py` (Heng-Soh RLog), rather than running both as conjunctive layers | Heng & Soh ICLR 2025 (arXiv 2505.15008) | Module built, **layered not replaced**. Architectural cleanup deferred. |
| 4 | Flip `colbert_retriever.py` from cosine-fallback scaffold to real PyLate / GTE-ModernColBERT late-interaction path | arXiv 2603.25248; PyLate Aug 2025 | Module built, **scaffold not implementation**. Requires `pip install pylate` + corpus re-indexing. Deferred. |
| 5 | Add TabPFN v2.5 as a 5th combiner expert | Hollmann Nature 2025 | Not built. Out of pmwhybetter.md scope. Deferred. |
| 6 | Outcome-RL / Turtel DPO fine-tune on Qwen3-8B-NVFP4 | arXiv 2502.05253; arXiv 2505.17989 | `scripts/finetune_qwen3_outcome_rl.py` runnable. Defensibly deferred per top-5 #5: "wait for queue-aware reconciliation." |

### Reasoning for choosing #1

Five independent reasons aligned:

1. **Same-day answer.** The hypothesis was evaluable by retraining
   LightGBM on the existing `signal_outcomes` ⋈ `historical_trades`
   join and comparing Brier on a chronological hold-out. No forward
   trading required. Every other item on the list needs ≥500 future
   resolved trades to actually pay off in realised P&L.

2. **Targets the certified slice.** Items #2–#6 either operate at
   pipeline scope (#2 calibration tier) or improve experts (#4, #6
   retriever / forecaster) that already have shrunk weight in the
   log-pool combiner. Item #1 directly modifies the LightGBM that
   feeds the `sports_global` cert — the one slice currently trading.

3. **Smallest surface.** One feature swap + one retrain run +
   one chronological evaluation. The other items require multiple
   call-site changes, dependency installs, or hyperparameter searches.

4. **The contamination is documented.** Sirolly Nov 2025 puts sports
   as the worst-affected wash category (20% steady-state / 60% Dec
   2024 peak). PROJECT.md already records this caveat. The hypothesis
   has external support — *if* the population-average wash share
   applies to our dataset, the swap is correct.

5. **Avoids the scaffolding trap.** The literature pass II commit
   shipped three modules (#2, #3, #4 above) that *appear*
   complete but aren't wired into the hot path. Doing more of the
   same risks declaring success without the empirical work to
   support it. Item #1 carries no such risk because the verdict is
   empirical, not declarative.

### Disposition after the experiment

The wash-volume hypothesis failed empirically — `volume` is the
top-1 feature by LightGBM gain importance on `sports_global`, and
removing it collapses AUC by −0.18. Details in the next session log.

The disposition of items #2–#6 is unchanged by the falsification:

- **#2 (hierarchical calibrator wiring)** — module exists; wiring is
  a real production change that would touch the live calibration
  pipeline. Needs forward data to measure the realised lift. Defer
  until ≥500 forward trades have accumulated on multiple categories.
- **#3 (LR gate replacing not layering)** — current state is over-
  conservative but working. Removing a live gate is risky and the
  win is marginal. Defer.
- **#4 (ColBERT flip)** — `pip install pylate` + index build is a
  half-day task; the realised payoff comes from the LLM forecaster
  expert which has shrunk weight in the combiner. Bounded upside.
- **#5 (TabPFN v2.5)** — combiner refactor cost is real; Brier
  improvement is the kind of thing the queue-aware reconciliation
  needs to land first to know whether the improvement compounds
  against realistic fills or paper inflation.
- **#6 (Outcome-RL/DPO)** — script is runnable end-to-end with the
  installed deps; the training run is ~4–6 hours of GPU time but
  the realised payoff is also constrained by combiner shrinkage.

### Rule extracted from this exercise

**Before stripping a feature based on a published contamination
claim, measure the model's signal-to-noise gradient against that
contamination on our own data.** Sirolly's 20% wash share is real
*as a population average*; whether it applies to our specific
training cohort is an empirical question that takes minutes to
answer. `scripts/eval_decontaminated_features.py` is the artefact
that answers it for the sports_global slice, and is the template
for future contamination claims.

---

## Session log: May 10, 2026 (late evening) — wash-volume hypothesis falsified

The morning's session-end framing flagged the Sirolly Nov 2025
wash-trade contamination as the *one* model lever where the
empirical answer doesn't require 1,500 forward trades: replace
`volume` and `log_volume` (Sirolly's "must be replaced" features on
sports) with wash-robust `trade_count_24h` / `net_flow_24h` /
`unique_wallets_24h` / `top_wallet_share` features computed from the
176K-row `historical_trades` table.

`scripts/eval_decontaminated_features.py` was built to answer the
question with a chronological 80/20 train/test split on the n=697
sports_global resolutions:

| Variant | Brier | log_loss | AUC | ECE |
|---|---|---|---|---|
| **v2 (production, volume features)** | 0.0759 | 0.3509 | **0.7822** | 0.0677 |
| **v3 (flow features, volume removed)** | 0.0760 | 0.3356 | **0.6025** ⬇ | 0.0665 |
| **v4 (volume + flow together)** | 0.0759 | 0.3509 | 0.7822 | 0.0677 |

### Result

**Hypothesis falsified.** Three findings:

1. **`volume` is the top-1 feature in v2 by LightGBM gain importance.**
   Removing it collapses AUC by **−0.18** (0.78 → 0.60). The
   discriminative signal in volume is real and large.
2. **The naive flow features don't recover that signal.** None of
   `trade_count_24h_pre`, `net_flow_24h_pre`, `unique_wallets_24h_pre`,
   `top_wallet_share_pre` made the v3 top-5 by importance. They're
   more wash-robust *but* they discard the market-importance /
   question-quality correlations that volume implicitly encodes.
3. **v4 (both) is identical to v2 to 4 decimal places.** With volume
   available, the LightGBM ignores the flow features entirely. They
   carry zero incremental information on top of volume on this slice.

### Honest interpretation

Sirolly's published wash-share number (~20% steady-state, 60% Dec 2024
peak on sports) is real, but the 80% real-flow component in `volume`
still carries strong discriminative signal that exceeds the
wash-noise floor on the certified slice. The doc's framing — "volume
features must be replaced on sports_global" — was overstated for our
specific cohort and feature stack.

### What this changes

- **`features.py` is NOT modified.** The volume features stay in
  production. The current `sports_global` cert is *better-supported*
  than before this experiment because we explicitly tested the
  wash-contamination critique and the data rejected the alternative.
- **`scripts/eval_decontaminated_features.py` is preserved** as the
  falsifiability artefact — anyone re-checking the certification
  can re-run the experiment.
- **No retraining, no recert, no live-portfolio impact.** The bot
  continues running as configured.

### What we learned about the process

The user's analytical framing was correct: *this* model improvement was
the only one in the doc-list with a same-day answer, no waiting for
forward data, surface bounded to one script + one feature swap. We
ran the experiment, the hypothesis lost, and the right action is to
*not* ship the change. That's the right outcome of a falsifiability
loop — the cert survives a real test rather than absorbing more
features as scaffolding.

### Cumulative discipline rule

When a published literature claim cites a contamination level
(e.g. Sirolly's 20%), the right next step is *measure the gradient
of model performance against that contamination on our specific
data*, not silently strip the feature on the assumption that the
literature applies. The model's signal-to-noise floor is dataset-
specific; the published number is a population average.

