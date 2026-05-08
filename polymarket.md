# Polymarket Information-Arbitrage Trading Agent: Full Technical Blueprint

## TL;DR
- **Single-GPU build (RTX 5070 Ti, 16 GB)**: An NVFP4-quantized Qwen3-8B (≈5.5–6 GB) running on vLLM 0.10+ handles event extraction and news-to-market mapping; a 304M-param ModernBERT/FinBERT classifier head (~0.6 GB) handles fast sentiment; a GLiNER-small bi-encoder (~0.6 GB) does zero-shot NER; LightGBM + N-HiTS + Chronos-Bolt-Small (~200M, ~0.5 GB CPU/GPU) cover the statistical edge layer. Total steady-state VRAM ≈ 9–11 GB, leaving ~5 GB headroom for KV cache and embeddings.
- **Three layered edges**: (1) sub-minute news/X-Twitter latency arb on Polymarket CLOB v2 (post-Dec-2025 US-permitted via the intermediated DCM, with the offshore CLOB still the primary venue for international addresses); (2) base-rate / behavioral edge exploiting favorite-longshot bias (longshots <10¢ overpriced, favorites 90¢+ slightly underpriced) with Bayesian/Elo/MRP fundamentals models; (3) cross-venue arb against Kalshi, sportsbook implied probabilities, and Polygon on-chain Chainlink price feeds. Edges are combined via **logarithmic opinion pool** with empirically tuned per-source weights, then sized with **0.25× fractional Kelly** capped per-market.
- **Execution**: Async Python (uvloop + websockets) connected to `wss://ws-subscriptions-clob.polymarket.com/ws/market` and `/ws/user`, posting EIP-712-signed orders via `py_clob_client_v2` (GTC for passive maker, FOK for taker on info edge); colocated on a t3-medium-class VPS in **AWS us-east-1** (Polymarket's CLOB operator region). Risk layer enforces per-market exposure caps, daily VaR-style loss limit, automatic kill switch on UMA dispute or oracle pause, and walk-forward backtesting on tick-level Polymarket history (Telonex / MarketLens / Goldsky subgraph).

---

## 1. System Architecture

```
                                 ┌─────────────────────────────────────────────────────────┐
                                 │                    INGESTION LAYER (asyncio)            │
                                 │                                                          │
 Polymarket WSS /ws/market ─────►│ ws_polymarket.py    ─┐                                  │
 Polymarket WSS /ws/user   ─────►│ ws_polymarket_user.py─┤                                  │
 Polymarket RTDS (Chainlink)────►│ ws_rtds.py            │                                  │
 Kalshi REST + FIX/WSS     ─────►│ ws_kalshi.py          ├──► Redis Streams (XADD)         │
 X/Twitter (twitterapi.io) ─────►│ ws_x.py               │     ─ stream:trades             │
 GDELT 2.0 DOC API (15-min)─────►│ ingest_gdelt.py       │     ─ stream:news               │
 NewsAPI / Reuters RSS / AP─────►│ ingest_rss.py         │     ─ stream:books              │
 Truth Social / Telegram   ─────►│ ingest_social.py     ─┘     ─ stream:cross_quotes       │
 BLS / FRED / Fed APIs     ─────►│ ingest_macro.py                                          │
 Sportsbooks (TheOddsAPI)  ─────►│ ingest_odds.py                                           │
                                 └────────────────────────────────────┬─────────────────────┘
                                                                      │
                                 ┌────────────────────────────────────▼─────────────────────┐
                                 │                STORAGE  (process-local + persistent)      │
                                 │   TimescaleDB (ticks, books, trades, NAVs)                │
                                 │   Postgres   (markets, conditions, positions, audit)      │
                                 │   Qdrant/LanceDB (news + market description embeddings)   │
                                 │   DuckDB     (offline analytics + backtests)              │
                                 │   Redis      (hot state, pub/sub, rate-limit tokens)      │
                                 └────────────────────────────────────┬─────────────────────┘
                                                                      │
                                 ┌────────────────────────────────────▼─────────────────────┐
                                 │       FEATURE / SIGNAL LAYER (GPU-resident on 5070 Ti)    │
                                 │                                                            │
                                 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
                                 │  │  ML SERVING  │  │  STAT MODELS │  │  ARB ENGINE  │    │
                                 │  │ (vLLM / TRT) │  │ (CPU+GPU)    │  │              │    │
                                 │  │              │  │              │  │              │    │
                                 │  │ Qwen3-8B-NVFP4│ │ LightGBM     │  │ YES+NO=1    │    │
                                 │  │ ModernBERT-FT │ │ N-HiTS / TFT │  │ Poly↔Kalshi │    │
                                 │  │ GLiNER-small │  │ Chronos-Bolt │  │ Poly↔books  │    │
                                 │  │ bge-small EMB│  │ Elo/Glicko   │  │ Poly↔Chainlnk│    │
                                 │  │              │  │ MRP (PyMC)   │  │              │    │
                                 │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
                                 │         └──────► alpha_combiner.py ◄───────┘             │
                                 │                  (log pool + calibration)                  │
                                 └────────────────────────────────────┬─────────────────────┘
                                                                      │
                                 ┌────────────────────────────────────▼─────────────────────┐
                                 │          RISK GATE  (synchronous, in-process)             │
                                 │   • per-market exposure cap (% NAV)                       │
                                 │   • portfolio gross/net cap                                │
                                 │   • daily P&L floor + circuit breaker                     │
                                 │   • UMA-resolution-window block list                      │
                                 │   • adverse-selection blacklist (whale-on-other-side)     │
                                 └────────────────────────────────────┬─────────────────────┘
                                                                      │
                                 ┌────────────────────────────────────▼─────────────────────┐
                                 │            EXECUTION  (py_clob_client_v2 + ccxt)          │
                                 │   • GTC passive (capture spread + LP rewards)             │
                                 │   • FOK taker (when latency-edge timer < 30 s)            │
                                 │   • FAK partial-fill on cross-market arb                  │
                                 │   • Order manager: cancellation, replacement, EIP-712 sign│
                                 │   • Polygon RPC (Chainstack premium) for allowances/redeem│
                                 └────────────────────────────────────┬─────────────────────┘
                                                                      │
                                 ┌────────────────────────────────────▼─────────────────────┐
                                 │     MONITORING (Prometheus + Grafana + Loki, alertmanager)│
                                 │     PnL attribution by edge, latency histograms,          │
                                 │     model drift (Brier/log-loss), oracle dispute tracker  │
                                 └───────────────────────────────────────────────────────────┘
```

Process boundary: one `supervisord` (or systemd) parent; each ingest websocket and each model is a child. Inter-process bus is **Redis Streams** (consumer groups + XACK so a crashed worker doesn't drop fills). The decision loop is **single-threaded asyncio inside one process** so position state is consistent without locks; it consumes from Redis, calls model RPCs over Unix sockets, and emits orders.

---

## 2. Polymarket Mechanics & API (state of the platform, May 2026)

### 2.1 Endpoints

| Service | Base URL | Auth | Use |
|---|---|---|---|
| Gamma API (discovery) | `https://gamma-api.polymarket.com` | none | `/markets`, `/events`, `/series` for slugs, condition IDs, end dates, token IDs |
| CLOB REST | `https://clob.polymarket.com` | L1 (EIP-712) + L2 (HMAC API key) | order placement, cancels, balances, books |
| Data API | `https://data-api.polymarket.com` | L1 wallet | positions, trade history, P&L |
| Market WSS | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | none | book deltas, last trade, tick changes; subscribe with `{"assets_ids":[...], "type":"market", "custom_feature_enabled":true}` to receive `best_bid_ask`, `new_market`, `market_resolved` events |
| User WSS | `wss://ws-subscriptions-clob.polymarket.com/ws/user` | API key/secret/passphrase | order lifecycle: `PLACEMENT`, `MATCHED`, `MINED`, `CONFIRMED`, `RETRYING`, `FAILED` |
| Sports WSS | `wss://sports-api.polymarket.com/ws` | none | live scores |
| RTDS | `wss://ws-live-data.polymarket.com` | none | Binance + Chainlink price feeds the platform itself uses for short-duration crypto markets |
| Polygon RPC | use Chainstack/Alchemy paid plan | — | allowances, OrderFilled events, `redeemPositions()` |

### 2.2 Rate limits (Cloudflare-throttled, returns 429)

- General CLOB: 9,000 req / 10s
- `POST /order` (single): 3,500 / 10s burst, 36,000 / 10min sustained (~60/s sustained)
- `POST /orders` (batch up to 15): 1,000 / 10s burst, 15,000 / 10min
- `DELETE /order`: 3,000 / 10s burst, 30,000 / 10min
- `/book`, `/price`, `/midprice`: 1,500 / 10s
- Gamma `/markets`: 300/10s, `/events`: 500/10s
- WebSocket: 5 concurrent connections per IP

**Implication**: at retail-prosumer scale you will not hit these except on cancels-during-news-storms. Build a token-bucket throttle in Python (`aiolimiter`) per endpoint and exponential backoff on 429.

### 2.3 Order types (via `py_clob_client_v2`)

- `OrderType.GTC` — Good-Till-Cancelled, the default for passive maker quotes (also earns LP rewards / maker rebates).
- `OrderType.FOK` — Fill-Or-Kill; used as your "market order" for latency-edge takes.
- `OrderType.FAK` — Fill-And-Kill (partial OK); used for cross-market arbitrage where you want as much as you can get and don't want a stale resting order if conditions move.
- `OrderType.GTD` — Good-Till-Date, useful for time-decaying news edges (e.g., expire 5 minutes after a tweet).
- Post-only flag on `postOrder` to guarantee maker status.

### 2.4 Tick sizes & precision (`ROUNDING_CONFIG` in py-clob-client)

| Tick | Price decimals | Size decimals | Amount decimals |
|---|---|---|---|
| 0.1 | 1 | 2 | 3 |
| 0.01 | 2 | 2 | 4 |
| 0.001 | 3 | 2 | 5 |
| 0.0001 | 4 | 2 | 6 |

Tick sizes can change dynamically when a market becomes one-sided (e.g., 0.01 → 0.001 near 99¢). Always pull `tick_size` and `minimum_order_size` from `get_clob_market_info(condition_id)` before signing. FOK sell-side market orders are stricter: maker amount ≤ 2 decimals, taker ≤ 4 decimals.

### 2.5 Fees & rewards

- **Most markets**: zero maker fee, zero taker fee on the international CLOB. Resting limit orders earn a share of the daily liquidity-rewards pool (~$1M/day post-CLOB-v2) when within the configured `max_incentive_spread` of the midpoint and above `min_incentive_size`. Score is a quadratic kernel of distance-to-mid times size, summed both sides.
- **15-minute crypto Up/Down markets**: dynamic taker fee that peaks near 50/50 (~3.15% at $0.50) explicitly to neutralize Chainlink-vs-CLOB latency arb. Treat these markets as maker-only or skip.
- **Maker Rebates Program**: portion of taker fees (where charged) redistributed to makers daily.
- **CLOB v2 collateral is pUSD** (Polymarket USD) since April 28 2026; `USDC.e` is auto-wrapped on settlement. You still hold a USDC.e/USDC balance for funding; pUSD is internal.

### 2.6 Resolution (UMA Optimistic Oracle v2)

1. Market end_date hits → CTF Adapter posts a Request to UMA OO.
2. **Whitelisted proposer** (≥20 proposals/3 months, ≥95% accuracy as of 2025) posts an outcome with a $750 USDC.e bond.
3. **2-hour challenge window**.
4. If undisputed → resolves; redeem with `redeemPositions()` on the CTF (gets back pUSD/USDC.e).
5. If disputed once → adapter ignores, new request issued.
6. If disputed twice → DVM (UMA tokenholder vote, 48–72h).
7. Short-duration crypto markets bypass UMA and use **Chainlink Data Streams + Automation** (no challenge window).

**Risk implication**: every market your bot holds going into resolution must be force-flagged: stop adding exposure ≥4 hours before `end_date`, and refuse to take new positions on any condition_id where `accepting_orders=false` or that has an open UMA dispute (poll `https://oracle.uma.xyz` API or watch the adapter contract events).

### 2.7 Wallet, gas, allowances

- Polygon mainnet, chain_id 137. Hold ~5 POL ($1–2) for gas; allowances must be pre-approved once for the CTF Exchange and USDC contracts (use `nautilus_trader/adapters/polymarket/scripts/set_allowances.py` from poly-rodr's gist or equivalent).
- `signature_type=2` (Gnosis Safe proxy wallet, the default for accounts created via the Polymarket UI / Magic-link) is what you'll typically use; specify `funder=<safe_address>`.

### 2.8 US legal status (May 2026)

- Dec 2 2025: CFTC issued an Amended Order of Designation letting Polymarket operate an **intermediated DCM** for US users (limited markets, KYC, registered intermediary).
- The international/offshore CLOB (the high-liquidity venue) is **still ToS-blocked for US persons**; Polymarket has filed (April 2026) to lift that ban but no decision yet.
- Practical advice for the operator: trade against the venue your wallet is legally entitled to. Keep VPN-attempts off the table — Polymarket bans wallets and freezes funds when it detects them, and the IRS treats winnings as ordinary income regardless. The blueprint below is venue-agnostic and works against either the international CLOB or the new US DCM stack (same `clob.polymarket.com` API surface; the difference is which markets are listed for your KYC tier).

---

## 3. Information Edge Taxonomy

### 3.1 News / sentiment latency edge

Empirically, the typical price-impact window on Polymarket after a major-news event is **30 s to 5 min** for political/macro markets and **2–15 s** for events resolved by Chainlink-priced underlyings. Examples that historically created the largest mispricings:

- **Wire flashes** that contradict market consensus by >10pp (Reuters/AP/Bloomberg).
- **Tweets from a small set of "primary-source" accounts**: heads of state, central bankers, league commissioners, official sports beat reporters (e.g., Adrian Wojnarowski-tier), the actual companies' verified handles for M&A/earnings.
- **Government data drops at exact release times** (BLS NFP at 8:30 ET, FOMC at 14:00 ET, Fed minutes at 14:00 ET). Pull from `api.bls.gov/publicAPI/v2/`, `api.stlouisfed.org/fred/`, and `home.treasury.gov` rather than scraping.
- **Court rulings, regulatory orders** (PACER RSS, SEC EDGAR, FederalRegister.gov API).
- **Sports in-game state** before broadcast catches up (ESPN, official league APIs are 10–40 s ahead of TV).

The edge has compressed since 2024 — bots reach Polymarket within 1–3 s of a wire — so design assumes you're competing with the second tier (manual + slow bots) on 30–300 s windows, not the first tier on 1–10 s windows.

### 3.2 Statistical / base-rate edge

- **Favorite-longshot bias**: contracts ≤10¢ resolve YES less than implied (often <60% of implied probability), contracts ≥90¢ resolve YES slightly more often than implied. Largest in low-information markets (pop culture, niche politics, exotic geopolitics). Build a "bias model" that is just a calibrated mapping from Polymarket price → empirically-observed-frequency, fit on resolved markets.
- **Recency / narrative bias**: prices over-react to single polls or single news events. Detect with a 1h vs 24h vs 7d implied-volatility ratio; fade extreme moves where volatility is in the top 5% of recent history but news velocity (GDELT mentions) hasn't accelerated.
- **Time-decay near resolution**: as `time_to_resolution → 0`, theta-like effects make 95¢+ contracts converge to 100¢ quickly; passive limit orders 1–2¢ inside the spread on near-certain favorites with >24h to go reliably get filled and earn LP rewards.
- **Election base rates**: state-by-state MRP using public polling (538 archive, RealClearPolitics scrape, Roper iPoll, ABC/WaPo tracking, YouGov public toplines) + ACS post-stratification frame.
- **Sports**: Elo/Glicko-2 + log5 head-to-head; for NFL/NBA/MLB, pull team-level features (rest, injuries, travel, weather) and feed a LightGBM on top of the rating-based prior.
- **Macro / crypto**: GARCH(1,1) on hourly log-returns for crypto vol → fair price for "BTC closes above X" markets; Bayesian structural time series (`tfp.sts`) for unemployment/CPI prints.

### 3.3 Cross-market edge

- **Polymarket internal**: YES + NO < $1 − fees ⇒ buy both. Given near-zero fees on most markets, threshold is 0.99 minus 1 tick × 2 sides.
- **Polymarket vs Kalshi**: same event, accounting for Kalshi taker fee `ceil(0.07 × N × p × (1-p))`. A 7.5% raw spread on a 50¢ contract loses ~1.75¢ to Kalshi taker fees, leaving real edge ~5.75%. Use Laika Labs / FinFeedAPI mappings or build your own with embedding similarity + LLM verification.
- **Polymarket vs sportsbooks**: convert American odds → implied probability, devig with multiplicative method (Shin or power), accounting for 4–7% sportsbook hold. TheOddsAPI ($59–$299/mo) gives consolidated US sportsbook odds via REST.
- **Polymarket vs Chainlink/Binance spot** for crypto-resolved markets: the 5-min/15-min/1h/4h/24h Up-Down series. As of Jan 2026 dynamic taker fees neutralize the pure latency play; you can still earn maker rebates by quoting the right side as the underlying drifts.
- **Polymarket negative-risk markets**: in multi-outcome neg-risk events (e.g., "How many Fed cuts?"), sum of all NO prices should equal `n − 1` where n is outcomes, and Polymarket allows one-way conversion of NOs to a basket of YESes + pUSD — small structural arbitrage when conversion gas is below the spread.

---

## 4. ML / Model Architecture (RTX 5070 Ti, 16 GB)

### 4.1 GPU specs

- 8,960 CUDA cores, 280 5th-gen Tensor Cores (FP4/FP8/FP16/BF16)
- 16 GB GDDR7, 256-bit bus, **896 GB/s memory bandwidth**
- ~1,406 AI TOPS @ FP4 sparse
- 300 W TGP, PCIe 5.0
- Native NVFP4 (4-bit float, NVIDIA-spec) gives ~1.6× throughput vs BF16 with 2–4 pp quality loss; W4A16 (AWQ-style) is the fallback when an NVFP4 checkpoint isn't on HF.

### 4.2 VRAM budget

| Component | Format | Weights | KV cache @ 8k ctx | Total |
|---|---|---|---|---|
| Qwen3-8B (event mapping, news verification) | NVFP4 | ~4.0 GB | ~1.5 GB | ~5.5 GB |
| ModernBERT-large fine-tuned (fast sentiment, 304M) | FP16 | 0.6 GB | tiny | 0.6 GB |
| GLiNER-small bi-encoder (zero-shot NER, ~150M) | FP16 | 0.3 GB | — | 0.3 GB |
| bge-small-en-v1.5 embeddings (33M) for Qdrant | FP16 | 0.07 GB | — | 0.1 GB |
| Chronos-Bolt-Small (~48M, time-series) | FP16 | 0.1 GB | small | 0.2 GB |
| N-HiTS / NHITS (custom, Nixtla) | FP16 | 0.05 GB | — | 0.05 GB |
| TimesFM-1.0-200m (optional zero-shot horizon) | FP16 | 0.4 GB | small | 0.5 GB |
| LightGBM, XGBoost | CPU | — | — | 0 |
| Activations + workspace + CUDA overhead | — | — | — | ~1.5 GB |
| **Steady-state GPU** | | | | **~9–11 GB** |
| **Headroom** | | | | **~5–7 GB** |

That headroom is real and lets you (a) bump KV cache to 16k for long-document news verification, (b) hot-swap a fine-tuned LoRA per category (politics / sports / crypto) without unloading the base, or (c) batch GLiNER + ModernBERT inferences during news storms.

### 4.3 Sentiment / NLP stack

**Tier 1 — fast filter (every news item, every tweet from a watched account):**
- **ModernBERT-large** (~395M, but the FT classifier head is what matters for VRAM; encoder runs FP16) fine-tuned on FinancialPhraseBank + Twitter Financial News + a hand-labeled Polymarket-question-relevance set you build (target: 5–10k labeled pairs of `<headline, market_question>` → relevant/not + sentiment(-1,0,+1)). Sub-5 ms per item.
- Why ModernBERT not FinBERT: ModernBERT supports 8k context (vs BERT's 512), is faster, and beats FinBERT on out-of-domain text. Keep FinBERT-tone (FinBERT2-base) as a corroborating model for earnings/macro language only.

**Tier 2 — entity & relation extraction (anything that passes Tier 1):**
- **GLiNER-small** (~150M) with custom entity types: `politician`, `sports_team`, `crypto_asset`, `country`, `policy_action`, `ruling_body`, `date_phrase`. Bi-encoder variant for million-label scaling. Runs on CPU fast enough but pin to GPU for batched throughput.

**Tier 3 — heavyweight verification (only for high-conviction triggers):**
- **Qwen3-8B-Instruct in NVFP4** served with vLLM 0.10+, using FlashInfer attention. Prompt: "Given this news headline + body and this Polymarket question + resolution rules, output JSON `{relevant: bool, direction: yes|no|neither, confidence: 0–1, mechanism: <1 sentence>}`."
- Alternative for stronger reasoning: Gemma3-12B-it at NVFP4 (~6 GB) or GPT-OSS-20B at MXFP4 (~10 GB at the limit of the card) — only if you find Qwen3-8B mis-mapping events. Benchmarks (arXiv 2601.09527) show RTX 5070 Ti gets ~25–60 tok/s on these models at batch=1 with vLLM, plenty for a hundreds-of-events-per-hour pipeline.

**Why not run a 70B over remote API**: latency. The whole point is sub-30s reaction. Local 8–12B with NVFP4 at 50–80 tok/s round-trips a 200-token JSON in <3 s.

### 4.4 Probability calibration

After every model that emits a probability, run a calibration layer fit on resolved markets:
- **Isotonic regression** (sklearn) for non-parametric calibration when you have ≥500 resolved samples per category.
- **Platt scaling / temperature scaling** when sample is small.
- Refit weekly with walk-forward: yesterday's resolutions → today's calibration. Monitor **Brier score** and **expected calibration error (ECE)**; alert when either drifts >20% week-over-week.

### 4.5 Forecasting models (statistical edge)

- **LightGBM** (CPU, ~50 ms per market): tabular features per market — price-history percentiles, order-book imbalance, news-velocity from GDELT, time-to-resolution, YES-side liquidity, category dummies, cross-market spread, hours-since-last-trade, neg-risk flag, longshot-bias-bucket.
- **N-HiTS** (Nixtla `neuralforecast`): minute-bar time series for the YES probability — captures multi-rate seasonality (e.g., daily news cycles) and outperforms transformer baselines by ~20% MAE at 50× lower cost. Train per-category, ~50k params, runs on GPU in ~2 ms per series.
- **Chronos-Bolt-Small / TimesFM-200M**: zero-shot foundation models for cold-start markets where you have <100 ticks of history. T5-style tokenized time series, ~30 ms inference.
- **Per-domain priors**:
  - **Elections**: hierarchical Bayesian MRP in PyMC — partition voters into ~6,500 demographic-state cells, fit multilevel logit on poll micro-data (or YouGov public toplines), poststratify with ACS. Outputs a state-by-state vote-share posterior; convert to win probabilities via Monte Carlo over correlated state errors.
  - **Sports**: Glicko-2 ratings (per-league, ratings deviation included) + log5 for head-to-head probability + LightGBM uplift on rest/injuries/weather features. For tennis specifically, surface-conditioned Elo (Glicko has empirically not beaten Elo on tennis in clean comparisons).
  - **Crypto vol**: GARCH(1,1) via `arch` for hourly log-returns; convert to BS-style fair value for "X above strike at time T" markets.
  - **Macro**: Bayesian structural time series (`tfp.sts.fit_with_hmc`) for CPI/NFP — captures trend + seasonality + AR(1) noise with credible intervals.

### 4.6 Embedding & retrieval (news → market mapping)

- **bge-small-en-v1.5** (33M, FP16) embeds every market question + resolution rules at market-creation time → Qdrant collection `markets_v1`, hnsw m=16, ef_construction=128.
- Every incoming news headline is embedded the same way → kNN search top-50 → re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, fast) → top-5 to Qwen3-8B for verification.
- **Deduplication**: maintain a recent-news Qdrant index keyed by 5-min windows; any incoming headline with cosine similarity >0.92 to the last 30 min is treated as the same event (avoids double-counting RT chains and wire-rebroadcasts).

---

## 5. Data Pipeline

### 5.1 Real-time sources

| Source | Endpoint | Tier | Cost |
|---|---|---|---|
| Polymarket Market WSS | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | free | $0 |
| Polymarket User WSS | same `/ws/user` | API key | $0 |
| Polymarket RTDS | `wss://ws-live-data.polymarket.com` | free | $0 |
| Polygon RPC | Chainstack Growth or Alchemy Growth | paid | ~$50/mo |
| Kalshi REST + WSS | `https://trading-api.kalshi.com/trade-api/v2` + WSS | free with account | $0 |
| X / Twitter | **twitterapi.io** ($0.15 / 1k tweets) — third-party that scrapes; sub-second latency, full-text search. Official Pro tier is $5,000/mo. | paid | ~$50–200/mo |
| GDELT 2.0 DOC API | `https://api.gdeltproject.org/api/v2/doc/doc` (15-min refresh, free) + GDELT Cloud Events API (`gdeltcloud.com/api/v2/events`) for paid CAMEO+ | free + optional | $0–$200/mo |
| NewsAPI | `https://newsapi.org/v2/everything` | Business tier needed for commercial | $449/mo |
| Reuters / AP | direct paid feeds are enterprise; substitute is **NewsCatcher API** (~$200/mo) or **Aylien** | paid | $200–500/mo |
| Truth Social | scrape via official RSS or `truthbrush` (community) | free | $0 |
| Telegram | MTProto via `telethon`, channel-subscribe to crypto/geo channels | free | $0 |
| FRED (Fed) | `https://api.stlouisfed.org/fred/` | free | $0 |
| BLS | `https://api.bls.gov/publicAPI/v2/` | free | $0 |
| Treasury | `home.treasury.gov` JSON | free | $0 |
| TheOddsAPI (sportsbooks) | `https://api.the-odds-api.com/v4/` | paid | $59–$299/mo |
| Polymarket historical | **Telonex** or **MarketLens** (tick + L2 books) or Polymarket `/prices-history` | paid for full | $50–500/mo |
| On-chain events | Goldsky subgraph or self-indexed Polymarket subgraph | free–$50/mo | $0–$50/mo |

Realistic monthly data budget: **$400–$900/mo**. Above $1k/mo only if you add NewsAPI Business + premium news wires.

### 5.2 Storage

- **TimescaleDB** (Postgres extension): hypertable `ticks` (time, condition_id, token_id, side, price, size), `books` (snapshot + delta), `trades`. Compress chunks older than 7 days. ~5–20 GB/month for the markets you care about.
- **Postgres** vanilla: `markets`, `events`, `series`, `positions`, `orders`, `fills`, `pnl_attribution`, `model_runs` (audit).
- **Qdrant** (or LanceDB if you want pure-Python / file-backed): collections `markets_v1`, `news_v1`, `tweets_v1`. ~1 GB.
- **DuckDB**: read-only mirror of Parquet exports for backtests and notebook research; vastly faster than Postgres for analytical scans.
- **Redis**: hot state — current best bid/ask per token, current position per condition_id, rate-limit tokens, model-output cache (5-second TTL on LLM verification calls to dedupe identical headlines hitting in parallel).

### 5.3 Stream processing

- `asyncio` + `uvloop` event loop in each ingest process.
- `websockets` library, with auto-reconnect and exponential backoff (start 1 s, cap 60 s).
- Heartbeats: send `PING` every 10 s on Polymarket WSS; the server closes idle connections.
- Redis Streams as the message bus, consumer groups so the signal layer can scale to multiple workers if a single Python process saturates.
- Polars (lazy) for any in-memory feature recomputation; pandas only at the API surface where ergonomics matter.

### 5.4 Feature engineering

Per-market features computed on every book update or every 1 s, whichever comes first:

- **Microstructure**: best bid/ask, mid, spread (cents and bps), book imbalance at depth-5 and depth-20, weighted mid (`Σ size_i × price_i / Σ size_i`), top-of-book size, queue position estimate.
- **Time series**: 1-min, 5-min, 1-h, 24-h returns of the YES price; realized vol; signed-volume momentum (Lee-Ready style trade direction is unreliable on Polymarket — paper finds only 59% accuracy vs on-chain truth, so prefer `last_trade_price` change-direction over inferred trade sign).
- **Cross-venue**: synced quote of mapped Kalshi market, mapped sportsbook implied probability, Chainlink/Binance underlying price for crypto markets; sign and magnitude of the spread.
- **News**: GDELT mention count for the market's entities in the last 5/15/60 min, normalized by 7-day baseline; sentiment z-score; novelty score (1 − max-cosine-sim to last 30 min news); count of "primary-source" account tweets in the last 5 min.
- **Resolution-distance**: hours-to-end_date, log of that, "is_within_24h_of_close" flag.
- **LP-rewards features**: `min_incentive_size`, `max_incentive_spread`, current rewards-pool size for the market (visible via Gamma API).

### 5.5 News-to-market mapping (the hard part)

```
news_item  ──► embed(bge-small) ──► Qdrant kNN top-50 markets
                                         │
                                         ▼
                                cross-encoder rerank → top-5
                                         │
                                         ▼
                       Qwen3-8B-NVFP4 verification:
                       prompt = (headline + body[:3000] + market.question + market.resolution_rules)
                       returns JSON {relevant, direction, confidence, mechanism}
                                         │
                                         ▼
                         if confidence > 0.7 and relevant ──► signal candidate
```

Two-pass design: kNN narrows to ~5 candidates per item (sub-10 ms total), then ~3 s of LLM verification. Critical that resolution rules are in the prompt — they're what determines `direction` correctly (e.g., "Will Trump tweet X" vs "Will Trump say X publicly" resolve differently).

---

## 6. Signal Generation & Alpha Combination

### 6.1 Per-edge fair-value probability with confidence

Each model emits `(p̂, σ̂)` where σ̂ is a model-specific uncertainty:

- **Sentiment/news edge**: p̂ = current_market_price + Δ from logistic regression on (sentiment, novelty, source_credibility, time_since_event). σ̂ from bootstrapped residuals.
- **Statistical edge (LightGBM/N-HiTS)**: p̂ from the model output; σ̂ from quantile regression heads (LightGBM with `objective=quantile`, predicting the 10/50/90 quantiles).
- **Cross-market edge**: p̂ from the mapped venue's mid (devigged for sportsbooks, fee-adjusted); σ̂ from the mapped venue's spread + a venue-specific noise floor.
- **Domain priors (MRP/Elo/GARCH)**: posterior mean and posterior std deviation directly.

### 6.2 Combination via logarithmic opinion pool

For binary outcomes, log-pool combines log-odds rather than probabilities. Given K experts with probabilities p_k and weights w_k (Σw_k = 1):

```
logit(p*) = Σ_k w_k × logit(p_k)
p* = σ(logit(p*))
```

Why log-pool over linear pool: it's externally Bayesian, geometric-mean-like (so a confident 99% from one expert combined with 50% from another lands closer to the confident estimate, matching Bayesian intuition), and empirically stronger on forecasting benchmarks.

**Weight learning**: fit weights once a week via constrained-optimization minimizing log-loss on resolved markets in the last 90 days, separately per market category (politics/sports/crypto/macro/culture). Use scipy `minimize` with `LinearConstraint(np.ones(K), 1, 1)` and `Bounds(0, 1)`.

For ensembles within a single edge type (e.g., Chronos-Bolt + N-HiTS + LightGBM as the "statistical" composite), use **stacking** with a logistic regression meta-learner on out-of-fold predictions.

### 6.3 Edge calculation and threshold

```
edge_yes = p_model − ask_yes − 0.5 × spread − fee_taker − slippage_estimate
edge_no  = (1 − p_model) − ask_no − 0.5 × spread − fee_taker − slippage_estimate
trade if max(edge_yes, edge_no) > θ_min(category)
```

`θ_min` defaults: politics 4¢, sports 5¢, crypto-resolved 6¢ (higher because of dynamic taker fees), pop culture 7¢ (longshot-bias adjustment), niche/low-liquidity 10¢.

For passive (maker) orders, `fee_taker` becomes 0 and you actually add LP-rewards EV; θ_min can drop to 1–2¢ but you accept queue-position risk.

### 6.4 Position sizing — fractional Kelly

For a binary contract bought at price q with model probability p:

```
b = (1 − q) / q                 # net odds
f* = (b·p − (1 − p)) / b         # full Kelly fraction
f_used = max(0, min(f_kelly_cap, 0.25 × f*))
```

- **0.25× Kelly** (quarter-Kelly) is the production setting — captures ~75% of the geometric-growth-rate of full Kelly with ~25% of the variance, and tolerates the inevitable miscalibration of p̂.
- **Per-market cap**: `f_kelly_cap = 0.05` (5% of NAV in any single condition_id).
- **Per-event cap** (multi-outcome): 8% across all tokens of a single event.
- **Per-category cap**: 25% (so a politics blowup can't take more than a quarter of NAV).
- **Gross / net cap**: gross deployed ≤ 70% of NAV; net (signed) ≤ 50% of NAV.
- For correlated bets (multiple markets all keying off the same news), use **simultaneous Kelly** (Whitrow / Smoczynski-Tomkins formulation) — solve a small QP minimizing −E[log(W)] across the correlated bets jointly. If that's too heavy, divide the per-bet f* by the number of correlated bets as a conservative approximation.

---

## 7. Execution Engine

### 7.1 Order strategy by edge type

| Edge | Order Style | Type | Why |
|---|---|---|---|
| News latency (clean signal, t<60s) | Aggressive take | FOK or FAK | Speed > price; fill the whole leg or none |
| News latency (decaying, t<5min) | Cross-the-spread by 1 tick | GTC | Want priority but fine with 30–60s wait |
| Statistical mispricing (no time pressure) | Passive maker at mid ± 1 tick | GTC | Earn LP rewards + capture spread |
| Cross-market arb | Both legs FAK | FAK | Partial fills OK, never want a stale leg |
| Pre-resolution favorite drift | Passive maker | GTC | Earn rewards; theta works for you |

### 7.2 Latency budget

Target end-to-end (news event → order on book) = 1.5–4 s for the "fast" path:

- News ingest (Twitter API webhook or GDELT 15-min poll → 0–900 s on GDELT, 0.5–2 s on twitterapi.io)
- Embedding + kNN: ~10 ms
- LLM verification (Qwen3-8B NVFP4, 200-token output): ~2.5 s
- Signal combination + risk gate: ~5 ms
- Order build + EIP-712 sign: ~30 ms (`eth_account` does this in pure Python; pre-cache the domain separator)
- POST `/order`: ~80–200 ms RTT from AWS us-east-1

Bypass the LLM for very-high-confidence triggers (top-tier source + direct entity match + sentiment magnitude > threshold) and accept rule-based mapping in <500 ms.

### 7.3 Colocation

- Polymarket's CLOB operator runs out of AWS us-east-1 (Cloudflare-fronted but the origin is there). Run the bot in **us-east-1** on a t3.large or c7i.large EC2 instance ($60–80/mo on-demand, $25/mo with 1-yr RI).
- The GPU lives **at home** (the RTX 5070 Ti) — exposed to the trading VPS via Tailscale or a Wireguard tunnel. Model RPCs go over the tunnel; only orders go from VPS → Polymarket. Tunnel adds 10–25 ms RTT, fine for the 2.5 s LLM step.
- Alternative if you want everything in one place: rent a single bare-metal box in NYC/Ashburn with a 5070 Ti or 5090 (Hostkey, GPUMart, OVH) — ~$200–400/mo and you get ~50 ms to Polymarket's origin.

### 7.4 Order management

- Maintain a local mirror of your open orders keyed by Polymarket order_id; reconcile on every `/ws/user` message (PLACEMENT, MATCH, CANCEL).
- Partial fills: track `size_matched` cumulatively; do not re-place until original is canceled (Polymarket order_ids are unique per signature/salt).
- Stale-order watchdog: any GTC order open >1 hour without a price recheck triggers an auto-cancel; replace with a fresh quote.
- Cancel-on-disconnect: on websocket reconnect, fetch `get_orders()` and `cancel_market_orders()` for any condition_id where the model-price has moved >2 ticks while we were blind.
- Use **batch endpoints** (`POST /orders` up to 15, `DELETE /orders` up to 15) when refreshing maker quotes across many markets — saves rate-limit budget.

### 7.5 Slippage model

For any size S at top-of-book size T, slippage in cents ≈ `(S/T − 1) × spread + γ × (S/T)²` with γ fit empirically per market category. At retail-prosumer sizes ($200–$5,000 per leg) on liquid Polymarket markets, slippage rarely exceeds 1–2 ticks, but on mid-tier markets (<$50k 24h volume) you can pay 5–10 ticks for a $5,000 take. Always preview with a `book.impact(side, size)` calculation (replicate locally from the order_book snapshot) before sending FOK.

### 7.6 Resolution & redemption

- Watch each held condition_id's `accepting_orders` flag and `end_date_iso`.
- On `market_resolved` event over WSS: call `redeemPositions(conditionId, indexSets)` on the CTF Adapter via Polygon RPC. ~$0.001 in gas. Funds arrive as pUSD; auto-swap to USDC.e via the platform's collateral adapter or hold pUSD.
- Run a daily sweeper that aggregates winning tokens across all resolved markets and redeems them in one tx where possible.

---

## 8. Risk Management

### 8.1 Pre-trade gates (must pass all)

1. **Per-market exposure** ≤ 5% of NAV.
2. **Per-event** (sum across all tokens of one event) ≤ 8% of NAV.
3. **Per-category** ≤ 25% of NAV.
4. **Gross deployment** ≤ 70% of NAV.
5. **Net signed** ≤ 50% of NAV.
6. **Time-to-resolution** ≥ 4 hours (no new positions inside the UMA window).
7. **Liquidity check**: top-of-book size on the side you're taking ≥ your order size (else split or pass).
8. **Adverse-selection check**: if a single non-MM wallet (per Polymarket's on-chain data, filterable via Goldsky) has taken >$25k on the OTHER side in the last 5 min, pause new entries on that condition_id for 10 min — they often know something.
9. **Dispute check**: query the UMA OO for any open dispute on this condition_id; block if present.

### 8.2 Continuous limits

- **Daily loss limit**: −5% of NAV at 00:00 UTC start triggers a 24-hour kill (cancel all GTCs, no new entries; keep open positions).
- **Drawdown circuit**: −15% peak-to-trough triggers full liquidation and 7-day pause.
- **Per-edge attribution**: track P&L tagged by which edge produced the signal. If any edge's 30-day Sharpe drops below 0.3 or its 30-day P&L is < −2% of NAV, auto-reduce its weight in the log-pool to 0 until manual review.
- **Model drift**: per category, compare 7-day-trailing Brier vs 90-day-baseline Brier. Drift > +30% → halve sizing on that category until next weekly recalibration.

### 8.3 Tail-risk specific to Polymarket

- **UMA dispute risk**: ambiguous markets (e.g., culture/celebrity, geopolitics with unclear "official source" rules) carry a ~1.3% dispute rate and 4–6 day resolution delay. Categorize markets by ambiguity (your LLM can score the resolution rules clarity 1–5 at market discovery time); don't size up on score-1 markets.
- **Counterparty/platform risk**: keep ≤ 20% of NAV on the platform at any one time; sweep excess to a cold wallet weekly.
- **Smart-contract risk**: the CTF Exchange and Adapter contracts have been audited (OpenZeppelin) but bugs have happened on every DeFi platform; your max-on-platform cap mitigates.
- **Censorship/regulatory risk**: the platform can pause markets (admin function on UMA Adapter). Watch for governance / admin-action events on the contracts and assume any held position in a paused market is illiquid for 1–7 days.

### 8.4 Backtesting framework

- **Data**: Telonex or MarketLens tick + L2 book history; Polymarket's own `/prices-history` for OHLCV at 1-min/1h/1d aggregation; on-chain trades via Goldsky/PolymarketDataLoader for ground-truth fills. Augment with archived GDELT (BigQuery — free, full history back to 2015 in 15-min granularity) and Kaggle Polymarket dumps for cross-checking.
- **Simulator**: replay tick-by-tick, queue-position-aware. Use MarketLens's Python harness as a starting point or write your own — strategy subscribes to `on_book`, `on_trade`, `on_fill`; simulator advances time, fills your GTC orders only when the book's last_trade_price crosses your level (proxy for queue priority is "fraction of size at your level that's behind you" — assume FIFO and queue position = your size / total size at level when you arrived).
- **Walk-forward**: rolling 6-month train, 1-month validation, 1-month test; refit weights weekly within the test window. Never train on data after the prediction time (no look-ahead in news embeddings either — use point-in-time embeddings).
- **Costs**: include Cloudflare jitter (50–250 ms), maker/taker fees per category, slippage model fitted on real fills, gas on resolution.
- **Metrics**: Sharpe per edge type; total Sharpe; max drawdown; Brier; ECE; capacity (does P&L survive at 5×, 10× size); robustness across regimes (election year vs not; bull vs bear crypto).

### 8.5 Monitoring stack

- **Prometheus**: scrape custom `/metrics` from each process — order success rate, websocket lag, model latency p50/p99, current NAV, gross/net exposure, per-edge P&L.
- **Grafana** dashboards: one per edge, one for execution, one for risk.
- **Loki** for structured logs (one JSON line per order, fill, model run).
- **Alertmanager**: pages on (a) websocket disconnect >30 s, (b) order rejection rate >5%, (c) NAV drop >2%/hour, (d) unhandled exception, (e) UMA dispute on held position, (f) Brier drift breach.

---

## 9. Infrastructure / Code Structure

### 9.1 Repo layout

```
polyagent/
├── pyproject.toml                # uv-managed
├── docker-compose.yml            # postgres+timescale, redis, qdrant, prometheus, grafana, loki
├── infra/
│   ├── terraform/                # AWS us-east-1 VPS, security groups, IAM
│   ├── systemd/                  # unit files per service
│   └── tailscale/                # mesh config to home GPU box
├── polyagent/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings, env-driven
│   ├── data/
│   │   ├── ingest_polymarket.py  # /ws/market, /ws/user
│   │   ├── ingest_kalshi.py
│   │   ├── ingest_x.py
│   │   ├── ingest_gdelt.py
│   │   ├── ingest_rss.py
│   │   ├── ingest_macro.py
│   │   ├── ingest_odds.py
│   │   ├── store_timescale.py
│   │   └── store_qdrant.py
│   ├── models/
│   │   ├── llm_server.py         # vLLM OpenAI-compatible endpoint
│   │   ├── modernbert.py
│   │   ├── gliner_ner.py
│   │   ├── lgbm_tabular.py
│   │   ├── nhits_ts.py
│   │   ├── chronos_zero_shot.py
│   │   ├── elections_mrp.py
│   │   ├── sports_glicko.py
│   │   ├── garch_crypto.py
│   │   └── calibrator.py         # isotonic / temperature
│   ├── signals/
│   │   ├── news_to_market.py     # embedding + Qwen3 verifier
│   │   ├── statistical.py
│   │   ├── cross_market.py
│   │   ├── alpha_combiner.py     # log-pool + weights
│   │   └── kelly.py
│   ├── execution/
│   │   ├── clob_client.py        # py_clob_client_v2 wrapper
│   │   ├── order_manager.py
│   │   ├── slippage.py
│   │   └── resolution.py         # redeemPositions, sweeper
│   ├── risk/
│   │   ├── gates.py
│   │   ├── limits.py
│   │   ├── kill_switch.py
│   │   └── attribution.py
│   ├── monitoring/
│   │   ├── metrics.py            # Prometheus
│   │   └── alerts.py
│   └── backtest/
│       ├── replay.py
│       ├── strategies.py
│       └── walkforward.py
├── notebooks/                    # research, calibration, ad-hoc
├── tests/
└── scripts/
    ├── set_allowances.py
    └── bootstrap_qdrant.py
```

### 9.2 Async architecture pattern

Each ingest is its own asyncio Task in its own process; results onto Redis Streams. The decision loop is a single asyncio process consuming from Redis with consumer groups. The execution layer is also async (the py_clob_client_v2 v2 SDK is async-friendly; if not, wrap in `asyncio.to_thread`). This matches the pattern QuantPyLib's `wrappers.polymarket` already implements.

```python
# polyagent/main.py
async def main():
    bus = await Redis.from_url(settings.redis_url)
    risk = RiskGate(...)
    executor = OrderManager(clob_client, risk)
    combiner = AlphaCombiner(weights_path=...)

    async with anyio.create_task_group() as tg:
        tg.start_soon(news_consumer, bus, combiner, executor)
        tg.start_soon(stats_consumer, bus, combiner, executor)
        tg.start_soon(arb_consumer, bus, combiner, executor)
        tg.start_soon(book_consumer, bus, executor)  # for stop-outs / re-quotes
        tg.start_soon(user_consumer, bus, executor)  # fill reconciliation
        tg.start_soon(resolution_watcher, bus, executor)
        tg.start_soon(risk.heartbeat)
```

### 9.3 Key libraries

- `py_clob_client_v2` — Polymarket SDK (async, EIP-712 signing built in).
- `kalshi-python` (community) or hand-rolled HTTPX client.
- `web3>=7` for Polygon RPC, `eth_account` for signing.
- `vllm>=0.10` for LLM serving, `transformers`, `accelerate`, `bitsandbytes` (NF4 fallback only — NVFP4 lives in vLLM/llm-compressor).
- `gliner`, `sentence-transformers`, `qdrant-client`.
- `lightgbm`, `xgboost`, `scikit-learn`.
- `neuralforecast` (Nixtla — N-HiTS, NHITS, TFT), `chronos-forecasting`, `timesfm`.
- `pymc>=5` for MRP, `arch` for GARCH, `statsmodels`.
- `polars`, `duckdb`, `psycopg[binary,pool]`, `redis>=5`, `aiohttp`, `websockets`, `uvloop`.
- `pydantic`, `pydantic-settings`, `structlog`, `prometheus-client`.

### 9.4 vLLM vs TensorRT-LLM vs llama.cpp on Blackwell consumer

- **vLLM 0.10+ with NVFP4 + FlashInfer** is the recommended choice on RTX 5070 Ti as of early 2026: native Blackwell kernel autotuning, `nvidia/Llama-3.1-8B-Instruct-NVFP4` and Qwen3-8B-NVFP4 checkpoints exist on HF, batched throughput is ~3.5–7× single-request, OpenAI-compatible serving.
- **TensorRT-LLM** has the highest single-stream throughput (~35–50% above vLLM at low concurrency) but the build pipeline (ModelOpt → engine compile per model) is heavy and engine builds aren't portable. Consider only if you're CPU-bound by Python overhead at the LLM call.
- **llama.cpp** is the easy fallback (great Q4_K_M / UD-Q4_K_XL Unsloth quants, fast startup, runs on anything) but lacks continuous batching at the level vLLM does and won't hit native NVFP4 throughput. Use for laptop development, not production.

Recommendation: **vLLM in production**, llama.cpp for dev. Run vLLM as a Docker container with `--quantization nvfp4 --kv-cache-dtype fp8 --max-model-len 8192 --gpu-memory-utilization 0.65` so the LLM caps at ~10 GB and the rest stays free for ModernBERT/GLiNER which run in the same Python process via `transformers` directly.

### 9.5 Deployment

- **Two-host setup**: AWS VPS (us-east-1, t3.large) for ingest + execution + Postgres + Redis + Qdrant; home/colo box for the GPU. Tailscale mesh between them.
- **Docker Compose** for the VPS services (Postgres+Timescale, Redis, Qdrant, Prometheus, Grafana, Loki, Promtail).
- **systemd** units for the Python processes (ingest_*, signal_loop, executor); auto-restart on failure with rate-limit (avoid restart-storms during upstream outages).
- **Secrets**: AWS SSM Parameter Store or 1Password CLI, never `.env` in plaintext.
- **CI/CD**: GitHub Actions runs lint + tests + a small backtest sanity check on every PR; deploys via SSH + `git pull` + `systemctl restart`.

### 9.6 Monthly cost estimate (serious build)

| Item | Range |
|---|---|
| AWS VPS (us-east-1, t3.large, 100 GB EBS) | $80–120 |
| Polygon RPC (Chainstack Growth) | $50 |
| twitterapi.io (5–10M tweets/mo) | $50–200 |
| TheOddsAPI (sports) | $59–119 |
| Polymarket historical data (Telonex/MarketLens) | $50–500 |
| Optional NewsAPI Business / NewsCatcher | $200–500 |
| Domain + monitoring (Grafana Cloud free tier OK) | $0–30 |
| Power for home GPU (300 W avg, 24/7) | $25–40 |
| **Total realistic** | **$500–1,500** |
| **Spartan version** (skip premium news, smaller historical set) | **$250–400** |

GPU is sunk capital, not OpEx. Net: a $250–500/mo data + infra spend is enough to operate; >$1k/mo only if you want premium news wires.

---

## 10. Build Order (so you get to first $ fast)

1. Stand up the CLOB client + websockets + TimescaleDB writer; confirm you can subscribe to a market and persist every book delta.
2. Implement the cross-market YES+NO=$1 internal arb — simplest edge, no ML, validates execution end-to-end.
3. Add Kalshi ingest + market-mapper (LLM verifier), run paper-trade arb against historical 2024–2026 data.
4. Bring up the GPU LLM stack (vLLM + Qwen3-8B-NVFP4) and the GLiNER/ModernBERT stack; build the news → market verifier and run it offline against 6 months of GDELT + matched Polymarket prices to estimate realistic latency-edge P&L at your sizes.
5. Add LightGBM + N-HiTS statistical layer with backtest.
6. Wire the alpha combiner (log-pool with default equal weights), risk gate, Kelly sizing, monitoring.
7. Go live with $5k, hard daily loss cap of $250, edge-by-edge gradual sizing-up over 4–6 weeks.

---

## Caveats

- **Edge compression is real**. Pure latency arb on 15-min crypto Up/Down has been killed by dynamic taker fees; bot competition on breaking-news politics has tightened reaction windows from minutes to seconds. The blueprint assumes you're competing in the 30 s–5 min reaction band, not the 1–10 s band where dedicated firms with ~5 ms infra dominate. Plan to make most of your money on (b) base-rate / behavioral edges and (c) cross-venue arb, with (a) news as a kicker.
- **Twitter API**: `twitterapi.io` and similar third-party providers operate in a gray area relative to X's ToS. They're widely used but not officially sanctioned. For more durable infra, budget for the official Pro tier ($5k/mo) or build your own scrapers with care.
- **CFTC / regulatory drift**: the US picture is fluid (Polymarket's intermediated DCM went live Dec 2025, the international CLOB ban is under review as of April 2026). The blueprint is venue-agnostic, but a US person trading the offshore CLOB via VPN is asking for an account freeze and possible IRS / CFTC issues — treat that as an unacceptable operational risk, not a design choice.
- **UMA disputes** are rare (~1.3%) but P&L-asymmetric: you can lose 100% of a position to a wrong resolution. The "score the resolution-rules clarity" gate is a real mitigation, not a hand-wave.
- **VRAM math is steady-state**. During news storms with high-concurrency LLM verification you can spike to 14 GB; vLLM's `--gpu-memory-utilization 0.65` plus FP8 KV cache keeps you safe, but if you swap to a 12B model you should drop ModernBERT to a smaller distillation (e.g., FinBERT-Tone 110M) to keep headroom.
- **Backtests overstate P&L**. Polymarket book inference from public WSS agrees with on-chain ground truth on only ~59% of buckets (recent academic work), so simulators built on the public feed alone misattribute trade direction. Bias your expectations down by 20–40% from clean backtest numbers.
- **Self-funded LP rewards** can mask negative alpha. Track maker-rebate income separately from spread-capture and information-edge P&L; if rebates are >50% of total, your "alpha" is just rented liquidity and won't survive a rewards-program change (Polymarket's pool was cut significantly post-2024 election and is currently rebuilding around CLOB v2).