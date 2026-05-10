# CLAUDE.md — Polyagent

This file is loaded automatically by Claude Code when working in this repo. **Read this first.**

Polyagent is a Polymarket paper-trading agent. **No private keys, no real money, no on-chain placement.** Real-money conversion would require wallet handling, EIP-712 signing, USDC funding — which Claude does not build per safety policy.

---

## Discipline (most important section)

This project follows **falsifiability-first** discipline. Ship only what passes a real validation gate. Default-off when the historical falsifiability test fails. Never default-on a strategy whose edge has not been measured against the market price on held-out data.

Concretely:

- A strategy is allowed to **trade live only** if there is a row in `strategy_certificates` with `enabled=1` AND `ENABLE_CERTIFICATE_GATE=1`. The runtime reads this on startup.
- A strategy that has been measured and **failed** validation gets a row with `enabled=0` and a `reason` field documenting why. This is an audit trail, not just a kill switch.
- The cert criterion is rigorous: 8-fold purged CPCV with market-id grouping, sign-test on per-fold edge, DSR ≥ 0.95 (Bailey/Lopez de Prado 2014). PBO is computed but treated carefully — with a single config it collapses to 0.5 and is uninformative.
- "I think this should work" is not a reason to ship. "8/8 folds positive, p=0.004, DSR=0.996" is.
- **Paper P&L is optimistic.** The current `PaperBroker` uses VWAP fills with no queue model. Per the HFT/MM literature (hftbacktest docs, Quantopian-era data) this overstates live P&L by 20–40% in low-liquidity venues. Any cert that uses paper-fill P&L is provisional until validated on a queue-aware simulator. Do not treat paper Sharpe as live Sharpe.
- **Sample-size targets are higher than they look.** Bailey & López de Prado's MinBTL math: detecting a Sharpe 0.3–0.5 edge (realistic for question-only ML on Polymarket) needs ~1,500–3,000 fully-resolved OOS forward trades, not the ≥500 the project's earlier docs assumed. For correlated baskets multiply by 2–3×.

If you (Claude) are tempted to enable a strategy by default to "let it accumulate data," **don't**. Log-only is the correct mode for any signal whose edge is unmeasured. The bot already runs `yes_no_arb` (risk-free, by definition certifiable) and the certified `sports_global` combiner; everything else is log-only or behind opt-in env flags.

---

## What's currently running

```
Markets streaming:           ~500 markets / ~1500 token books (YES + NO)
Background tasks:            ~25 supervised, exponential-backoff restart
Strategies trading live:     yes_no_arb (always); combined_trader on sports_global only
Strategies log-only:         stat_signal, news_keyword_match, news_nli_match (opt-in)
Risk gates active:           29+
GPU:                         RTX 5070 Ti (sm_120 Blackwell, cu128 nightly), 17 GB
LLM forecaster default:      openai/gpt-oss-20b (Phi-4-mini-instruct fallback)
Dashboard:                   http://127.0.0.1:8080
```

---

## Active strategy certificates

| name | enabled | category | DSR | edge (log-loss) | n |
|---|---|---|---|---|---|
| `stat_lgbm_combiner_sports_global_v3_sport_features` | **1** | sports_global | 1.0000 | +0.123 | 626 |
| `stat_lgbm_combiner_sports_global_v2` | **1** | sports_global | 0.9959 | +0.128 | 626 |
| `stat_lgbm_combiner_sports_global` | 0 | sports_global | 0.996 | +0.128 | 626 (PBO false-fail; superseded by v2) |
| `stat_lgbm` | 0 | (overall) | — | — | 6,621 (`model_does_not_beat_market_logloss_overall`) |

Inspect with:
```sql
SELECT name, enabled, dsr_holdout, n_holdout, detail
FROM strategy_certificates ORDER BY issued_ts DESC;
```

To **disable** a cert (one-line rollback):
```sql
UPDATE strategy_certificates SET enabled=0 WHERE name='<cert_name>';
```

---

## Architecture (top to bottom)

```
                        ┌────────────────────────────────────────┐
                        │   Data ingestion                       │
                        │   Polymarket WSS, Gamma REST, RSS,    │
                        │   FRED, BLS, Congress, CourtListener, │
                        │   SEC EDGAR, Bluesky, USGS, NASA EONET │
                        └────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────────────┐
                        │   BookStore (in-memory order books)    │
                        └────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────────────┐
                        │   Signal layer (GPU)                   │
                        │   stat_signal       LightGBM + bge     │
                        │   news_match        Jaccard + FinBERT  │
                        │   news_nli_match    NLI (opt-in)       │
                        │   combined_signal   per-cat log-pool   │
                        │   natural_event_match  USGS+EONET      │
                        │   weather_llm_forecast LLM + base rate │
                        └────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────────────┐
                        │   Risk layer (29+ gates)               │
                        │   • CERTIFICATE GATE (this is new)     │
                        │   • selective abstention (Venn-Abers)  │
                        │   • smart-money tighten                │
                        │   • Kaminski-Lo gated stops            │
                        │   • drawdown-conditioned θ_min         │
                        │   • daily loss kill, fee buffer, etc.  │
                        └────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────────────┐
                        │   PaperBroker (asyncio-locked)         │
                        │   VWAP fills, WAL SQLite, kill switch  │
                        └────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────────────┐
                        │   Dashboard (aiohttp, port 8080)       │
                        └────────────────────────────────────────┘
```

Every long-running coroutine is wrapped in `supervised(name, factory)` — exponential-backoff restart, no silent task deaths.

---

## Module map

```
polyagent/
├── config.py                      env + defaults; all feature flags
├── main.py                        wiring; spawns ~25 supervised tasks
├── gamma.py                       Polymarket Gamma REST + categorize fallback
├── ws_polymarket.py               /ws/market subscription + reconnect
├── orderbook.py                   in-memory book reconstruction
├── paper_broker.py                cash + positions + fill simulation + settle
├── news_store.py                  unified news + signals SQLite layer
├── dashboard.py                   live web UI + JSON API
├── data/
│   ├── rss.py, fred.py, bls.py, congress.py, courtlistener.py
│   ├── sec_edgar.py, bluesky.py, alchemy.py, telegram.py
│   ├── clob_history.py            CLOB /prices-history backfill
│   ├── resolution_watcher.py      atomic settle on close
│   └── natural_events.py          USGS + NASA EONET poller (NEW)
├── models/
│   ├── features.py                question + book features (incl. sport-specific)
│   ├── lgbm.py                    LightGBM trainer + Predictor
│   ├── calibrator.py              isotonic / Venn-Abers / Beta cell calibrator
│   ├── embedder.py                bge-large via sentence-transformers
│   ├── llm_forecaster.py          gpt-oss-20b (or Phi-4-mini fallback) + AIA
│   ├── chronos.py                 Chronos-Bolt zero-shot (opt-in)
│   ├── retrain_loop.py            scheduled LGBM retrain
│   └── outcomes.py                signal_outcomes table
├── signals/
│   ├── news_match.py              keyword Jaccard + FinBERT direction
│   ├── news_verifier.py           NLI verifier (opt-in, scaffold)
│   ├── direction.py               VADER + question-polarity
│   ├── stat.py                    log-only LGBM signaler
│   ├── combined.py                CombinedSignaler (log-pool over experts)
│   ├── combiner.py                fit_weights + log_pool
│   ├── consistency.py             NegRisk consistency check
│   ├── natural_event_match.py     deterministic event→market matcher (NEW)
│   └── weather_forecaster.py      LLM-augmented weather forecaster (NEW)
├── strategies/
│   ├── yes_no_arb.py              risk-free YES+NO arb (always live)
│   ├── combined_trader.py         certified-categories-only after cert gate
│   ├── news_trader.py             log-only until news_match certified
│   ├── passive_poster.py          paper maker
│   └── stop_loss.py               trailing stops + Kaminski-Lo
├── risk/
│   ├── selective_gate.py          Venn-Abers width abstention
│   ├── smart_money.py             top-PnL wallet registry
│   ├── adverse_selection.py       per-token blacklist
│   ├── throttle.py                per-strategy daily kill
│   ├── exit_policy.py             Kaminski-Lo gated stops
│   ├── bocpd_gate.py              changepoint gate (default-off, falsified)
│   └── kill_switch.py             data/.STOP file watcher
├── eval/
│   ├── sharpe_honesty.py          PSR/DSR/MTRL nightly harness
│   └── harness.py                 CPCV/PBO test runner
├── supervisor.py                  exponential-backoff task wrapper
└── logging_setup.py               structlog config
```

---

## Storage

```
data/paper.db                      WAL-mode SQLite
├── markets, fills, fills_shadow   trade tape
├── nav_history, sharpe_history    portfolio snapshots
├── resolutions                    settled markets (10,220+)
├── signal_outcomes                LGBM/news/market predictions on resolved markets
├── signals                        log-only signal stream (>300k rows)
├── news                           ingested news with dedup
├── natural_events                 USGS + EONET event catalog (NEW)
└── strategy_certificates          cert audit trail (NEW)

data/lgbm_model.joblib             v1 LightGBM (active in production)
data/lgbm_model_v2.joblib          v2 with sport features (cert wash)
data/combiner.joblib               v1 per-category combiner
data/combiner_v2.joblib            v2 with v2 LGBM (cert wash; for sports_global use v3)
data/polyagent.log                 stdout dump
```

---

## How to run

```bash
# Activate venv (Python 3.10+)
.venv/Scripts/python.exe -m polyagent.main
```

Watch:
```bash
.venv/Scripts/python.exe -c "import sqlite3; c = sqlite3.connect('data/paper.db'); print('FILLS'); [print(r) for r in c.execute('SELECT ts,strategy,side,price,size,reason FROM fills ORDER BY ts DESC LIMIT 20')]"
```

Halt new orders without killing the bot: `touch data/.STOP`. Resume: delete the file.

Dashboard: http://127.0.0.1:8080.

### Current activation env (as the bot is running today)

```
DB_PATH=C:/Users/benja/Downloads/Polymarket/data/paper.db
LOG_PATH=C:/Users/benja/Downloads/Polymarket/data/polyagent.log
COMBINER_PATH=C:/Users/benja/Downloads/Polymarket/data/combiner_v2.joblib

ENABLE_CERTIFICATE_GATE=1          # gate combined_trader on certs
ENABLE_NATURAL_EVENTS=1            # USGS + NASA EONET ingest + matcher
ENABLE_LLM_FORECASTER=1            # load gpt-oss-20b (or Phi-4-mini fallback)
ENABLE_WEATHER_LLM_FORECAST=1      # spin up the LLM weather forecaster

# while gpt-oss-20b is downloading or unavailable, fall back:
LLM_FORECASTER_MODEL=microsoft/Phi-4-mini-instruct
LLM_FORECASTER_N_SAMPLES=2
LLM_FORECASTER_TEMPS=0.7
WEATHER_LLM_POLL_SEC=1800
```

To pre-download gpt-oss-20b:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/Scripts/huggingface-cli download openai/gpt-oss-20b
```
Once cached, drop the `LLM_FORECASTER_MODEL` override and gpt-oss-20b is picked up automatically.

---

## Key decisions Claude must respect

1. **Don't enable a strategy by default** unless its certificate has `enabled=1`. The cert gate is the source of truth.
2. **Don't ship a model-improvement claim without a CPCV test.** New features → retrain → re-materialize `p_stat_lgbm` → re-train combiner → re-cert. The pipeline lives in `scripts/`.
3. **Don't replace structurally validated code with "more elegant" versions.** Each gate / filter has a falsifiability story attached (see `PROJECT.md` § session logs). The risk-gate stack ordering matters.
4. **Don't commit data files to git.** `combiner_v2.joblib`, `lgbm_model_v2.joblib`, the SQLite DB, etc. are gitignored. The cert table state lives in `paper.db`, not in code.
5. **Don't push directly to `main`.** Push to a same-named feature branch and open a PR. The default upstream of new branches is `origin/main`, which is wrong; explicit `git push -u origin HEAD:claude/<branch-name>` is the pattern.
6. **For weather markets**: the `natural_event_match` rules are deterministic (confidence=1.0). The `weather_llm_forecast` is probabilistic. They are **complementary**, not competing — if both fire and agree, that's high confidence; if they disagree, the deterministic one wins.

---

## What's broken / blocked

- **gpt-oss-20b auto-download** — HF was throttling safetensors shards from this connection (stalled at 4.5/13 GB across three attempts). Fallback to Phi-4-mini works fine. Will resolve overnight or on a different connection.
- **`news_keyword_match`** — keyword Jaccard, hits 43% (worse than coin-flip). Replacement (`news_nli_match`) is built but the cross-encoder NLI model is timid on real entailments. Production-quality news classification needs a stronger NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) or LLM-tailored prompts.
- **No active earthquake markets** in the top 3,000 by volume right now. The natural-event matcher is ready and will fire when the next "Another 7.0+ EQ by [date]?" market appears (these cycle monthly historically).
- **NegRisk arb v2 falsified** — 100% partial-group artifacts on raw signal; v1 was patched, v2 was never built. See `PROJECT.md` for the full falsifiability log.

---

## Recent sessions

See `PROJECT.md` for the full session log. Most recent:

- **2026-05-10 — cert gate + natural events + gpt-oss-20b**. Calibration audit (`stat_lgbm` is +0.14 log-loss worse than market overall), `sports_global` certified at DSR=0.996 / edge +0.128, cert-gate architecture with 3-tier rollback, dashboard upgrade, USGS+EONET ingest + deterministic matcher, LLM-augmented weather forecaster, default LLM switched to gpt-oss-20b. 8 commits on `claude/nervous-haslett-3bf443`. Wiki note: `wiki/meta/Polyagent 2026-05-10 - Cert Gate, Natural Events, GPT-OSS-20B.md`.
- **2026-05-09 — Sharpe-honesty discipline pass.** Adopted falsifiability-first; 4 weeks of work documented (PSR/DSR/MTRL harness shipped, selective-abstention shipped, BOCPD default-off after falsification, Kaminski-Lo stops shipped with n≥30 floor, NegRisk arb v2 not built after falsification). See `PROJECT.md`.

---

## Honest performance ceiling

A question-only, taker-only, paper-broker ML system on Polymarket has a structural ceiling near zero ROI even with all gates. The 2026 microstructure literature (Bartlett & O'Hara, Akey et al., IMDEA, Bloomberg/Della Vedova) converges: retail taker-side trading is **structurally unprofitable** on prediction markets — bots had 52% raw accuracy vs retail's 55%, **bots win on execution, not prediction**. Maker side earned ~2× spread per contract from the systematic-YES-overbet behavioural surplus. The IMDEA arbitrage profits ($29M NegRisk + $10.58M single-condition over 12 months) are captured by sub-100ms operators on atomic on-chain `convertPositions` calls — a paper VWAP broker cannot compete.

Realistic targets given that ceiling:
- Brier ~0.09–0.11 with the full literature stack
- ROI roughly flat, ±2% per session is normal noise
- **Positive ROI lives** in (a) certified narrow slices like `sports_global` *with selective abstention on the high-confidence tail* (see "High-confidence-tail finding" below — this is the only literature-backed taker edge for question-only ML), (b) **maker-side spread capture** (Bartlett-O'Hara 2026, Polymarket Maker Rebates 20–25%, Akey et al. SSRN 6443103: "for 1-in-5 losers, the lower-bound cost of taking liquidity alone is enough to flip PnL from negative to positive"), (c) on-chain NegRisk arbitrage with co-located low-latency execution (out of scope per safety policy).

Directional model bets on question text alone are not the path. ForecastBench shows even frontier RAG-LLMs (Brier 0.1258) trail market consensus (0.1106) on liquid markets — meaning a state-of-the-art LLM forecaster has near-zero post-fee taker edge. Our `weather_llm_forecast` and `news_nli_match` are research projects, not profit levers.

---

## High-confidence-tail finding (2026-05-10 analysis)

The senior-quant review observed that **selective abstention via Venn-Abers / conformal prediction improves Sharpe only if the model's edge is concentrated in its high-confidence tail**. We tested this empirically on the 7,447 head-to-head rows (`signal_outcomes` joined to `p_market_24h`):

| confidence | sports_global model_LL | sports_global market_LL | delta |
|---|---|---|---|
| [0.30, 0.50) | 0.85 | 0.60 | **+0.24** model worse |
| [0.70, 0.90) | 0.24 | 0.32 | **−0.08** model better |
| [0.90, 1.00) | 0.11 | 0.22 | **−0.12** model better |

The certified +0.128 log-loss edge **is entirely in confidence ≥ 0.7**. At medium confidence the model is worse than market.

Whole-sample naive PnL Sharpe rises monotonically with confidence: −0.42 at confidence < 0.10, +0.47 at confidence ≥ 0.90. This means a **confidence-threshold gate** on `combined_trader.on_signal` (drop trades with `|p_combined - 0.5| < 0.7`, say) would compress fill count but materially improve realized P&L per trade. This is a Sharpe-positive change with empirical support.

Run `scripts/analyze_high_confidence_tail.py` to refresh the bucket table.

---

## Where to look next

If the user says "what now":

The senior-quant review (`pmwhy.md`) reorders our priorities. The new ranking, with literature support, is:

1. **Build queue-aware fill simulation.** Our `PaperBroker` uses VWAP fills with no queue model — the single largest source of paper-to-live Sharpe degradation in the HFT/MM literature. hftbacktest is the canonical reference; NautilusTrader's Polymarket adapter is the pragmatic alternative. Re-run all certified strategies through the new fill model and expect Sharpe to compress 30–60%. The certs that still pass under realistic fills are the real edges.
2. **Build maker-side execution (`passive_poster_v2`).** The 2026 fee/rebate regime explicitly subsidises this: makers pay zero, receive 20–25% of taker fees as USDC rebates. Bartlett & O'Hara 2026 documents the ~2× spread per contract maker advantage from the YES-overbet behavioural surplus. Akey et al. (SSRN 6443103) shows for 1-in-5 retail losers, switching from taker to maker would flip their PnL positive. This is where 70% of engineering effort should go.
3. **Add a confidence-threshold gate to `combined_trader`** before the cert gate. The high-confidence-tail finding above says this is Sharpe-positive on existing data. Cheap one-line change in `combined_trader.on_signal`.
4. **Forward-test `sports_global` cert under the new fill model** — but with the corrected sample-size target: **≥1,500–3,000 OOS forward trades**, not 500. Bailey-López de Prado's MinBTL math says detecting a Sharpe 0.3–0.5 edge (realistic for question-only ML) needs ~3–6× more trades than the original "≥500" estimate.

Things to **not** do:
- Stop adding features. Strict feature-freeze until ≥1,500 forward trades accumulate. Every new config inflates the multiple-testing burden in DSR (Bailey-López de Prado 2014).
- Stop hoping the LLM forecaster will rescue this. Frontier RAG-LLMs trail market consensus on liquid markets; gpt-oss-20b and Phi-4-mini are an order of magnitude below frontier.
- Anything that requires real-money keys (LP making, on-chain NegRisk arb) — out of scope per safety policy.

The big-picture answer: **the bot is not making money because it's competing in the most-bot-saturated taker niches with paper-quality infrastructure**, and the gap to profitability is structural (execution stack), not algorithmic. Source: `pmwhy.md` (in repo root) — read in full before any architecture decision.

---

## Repository conventions

- Branch: `claude/nervous-haslett-3bf443` is the active worktree branch. Default branch is `main`. Push pattern: `git push -u origin HEAD:<same-name>` to land on a same-named remote branch and open a PR rather than pushing directly to `main`.
- Commits follow a `topic: short summary` style with body text explaining falsifiability where applicable. See git log on this branch for the recent rhythm.
- Tests: `pytest tests/`. Currently 39 tests, all passing. Each new gate / matcher / forecaster ships with a tests file.
- Data files (`*.joblib`, `*.db`, `*.log`) are gitignored. The cert table state lives in `paper.db`.
