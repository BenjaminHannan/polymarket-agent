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

A question-only ML model on Polymarket has a structural ceiling near zero ROI even with all gates. Realistic targets:
- Brier ~0.09–0.11 with the full literature stack
- ROI roughly flat, ±2% per session is normal noise
- **Positive ROI lives** in (a) certified narrow slices like `sports_global` (the +0.128 log-loss edge translates to small but real PnL), (b) maker-side spread capture (Yang 2026: $121/market making vs $63 taking — paper-mode only), (c) NegRisk arbitrage if the partial-group artifact problem can be solved (IMDEA: $29M of $40M of all Polymarket arb 2024-2025).

Directional model bets on question text alone are not the path. Specialization, structured retrieval, and execution-side capture are.

---

## Where to look next

If the user says "what now":

1. **Forward-test the `sports_global` cert.** Watch fills land over 24-48h. If cumulative edge tracks `+0.128 × n × notional` within 1σ → cert is real. If flat or negative across 20+ fills → disable via the SQL one-liner.
2. **Audit `news_keyword_match` the same way `stat_lgbm` was audited.** Backfill `p_news_match` on resolved markets with news signals; check calibration; identify category sub-slices where it beats market.
3. **External sport data for sports_global.** Team Elo + schedule + recent form + home/away from FBref / transfermarkt would actually expand the +0.128 edge. Regex features alone are a wash (proven this session).
4. **Better news verifier.** Try `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` or wire a Claude/GPT API call for top-100 markets only.

Things to **not** do:
- Stop adding features. Run for weeks, accumulate ≥500 resolved trades, then re-evaluate.
- Anything that requires real-money keys (LP making, on-chain arb, etc.) — out of scope per safety policy.

---

## Repository conventions

- Branch: `claude/nervous-haslett-3bf443` is the active worktree branch. Default branch is `main`. Push pattern: `git push -u origin HEAD:<same-name>` to land on a same-named remote branch and open a PR rather than pushing directly to `main`.
- Commits follow a `topic: short summary` style with body text explaining falsifiability where applicable. See git log on this branch for the recent rhythm.
- Tests: `pytest tests/`. Currently 39 tests, all passing. Each new gate / matcher / forecaster ships with a tests file.
- Data files (`*.joblib`, `*.db`, `*.log`) are gitignored. The cert table state lives in `paper.db`.
