# polyagent — Polymarket Paper Trading Bot

End-to-end paper-trading agent: streams the live Polymarket public WSS feed,
reconstructs order books, ingests news/macro/legal data from 30+ free sources,
runs the YES+NO ask-side arb detector, and matches news to markets via keyword
overlap. Persists everything to SQLite. **No real money, no private keys.**

## What runs today

### Market data (no auth)
- `gamma.py` — pull top-volume binary markets + token IDs
- `ws_polymarket.py` — `/ws/market` subscription, auto-reconnect
- `orderbook.py` — in-memory book reconstruction per token

### Paper execution
- `paper_broker.py` — simulates fills against live top-of-book, tracks cash,
  positions, NAV, persists fills to SQLite

### Strategies
- `strategies/yes_no_arb.py` — buys both legs when YES_ask + NO_ask < $0.99
- `signals/news_match.py` — keyword Jaccard overlap, logs candidates only
  (no trades; need calibration first)

### News / macro / legal ingest
| Module | Source | Auth |
|---|---|---|
| `data/rss.py` | Reuters / AP / BBC / NPR / CNBC / Politico / The Hill / Guardian / Al Jazeera / Fed / Treasury / FederalRegister / SEC press / SCOTUSblog / ECB / IMF / WhiteHouse — 23 feeds | none |
| `data/bluesky.py` | Bluesky Jetstream public firehose | none (watchlist DIDs) |
| `data/fred.py` | FRED — UNRATE, CPI, NFP, GDP, FedFunds, 10Y, VIX, WTI, etc. | FRED key |
| `data/bls.py` | BLS — labor/inflation series | BLS key |
| `data/congress.py` | Congress.gov — 119th Congress bills | Congress key |
| `data/courtlistener.py` | CourtListener — recent federal opinions | Token |
| `data/sec_edgar.py` | SEC EDGAR — 8-K filings RSS | UA header |
| `data/alchemy.py` | Polygon RPC heartbeat | Alchemy key (currently 403) |

## Statistical layer

**Backfill** populated **4,565 resolved markets** from Gamma in ~10s.
Base rate across all binary outcomes: **17.4% YES, 82.6% NO** (the
favorite-longshot bias is large — a model that learns this prior already
beats the market on extreme questions).

After running `scripts.train_lgbm`, the resulting `data/lgbm_model.joblib`
is a question-only LightGBM classifier. The `StatSignaler` task scores
every active market every 2 min and logs a `stat_signal` row to the
`signals` SQLite table when |p_model − p_market| ≥ 0.10. Currently
log-only — calibration on a held-out cohort is required before sizing.

## Smoke-test result (130s run)

```
Books:   147 streaming, 84 with two-sided quotes
News:    ~700 items ingested across all sources in 130s
Signals: 97 candidate news→market matches stored
Fills:   3 paper YES+NO arbs executed
NAV:     $10,000 → $10,003 (+$3, +0.03%)
```

Top news matches were good: "Yellen calls Trump's Powell investigation" →
"Jerome Powell out as Fed Chair" (overlap=4); Russia/Ukraine wire → Ukraine
ceasefire markets. False positives exist (e.g., World Cup markets matched on
generic "win/world") — these get filtered when we add a real verifier.

## Run

```bash
.venv/Scripts/python.exe -m polyagent.main
```

Ctrl-C to stop. Logs to stdout + `data/polyagent.log`.

## Inspect

```bash
.venv/Scripts/python.exe -c "
import sqlite3
c = sqlite3.connect('data/paper.db')
print('FILLS:'); [print(r) for r in c.execute('SELECT ts,strategy,side,price,size,notional,reason FROM fills ORDER BY ts DESC LIMIT 20')]
print('\\nNEWS BY SOURCE:'); [print(r) for r in c.execute('SELECT source,COUNT(*) FROM news GROUP BY source ORDER BY 2 DESC')]
print('\\nTOP SIGNALS:'); [print(r) for r in c.execute('SELECT score,condition_id,detail FROM signals ORDER BY score DESC LIMIT 10')]
"
```

## Config (.env)

| Var | Default | Meaning |
|---|---|---|
| `STARTING_NAV` | 10000 | paper cash |
| `MAX_MARKETS` | 50 | top-volume markets to watch |
| `MIN_LIQUIDITY` | 5000 | skip markets below this AMM TVL |
| `ARB_THRESHOLD` | 0.99 | YES+NO arb fires below this |
| `PER_TRADE_SIZE` | 100 | max shares per leg |
| `MAX_PER_MARKET` | 500 | max paper notional per market |
| `RSS_POLL_SEC` | 60 | RSS feed poll cadence |
| `FRED_POLL_SEC` | 1800 | FRED poll cadence |
| `NEWS_MATCH_MIN_OVERLAP` | 2 | min keyword overlap to emit signal |

## Layout

```
polyagent/
├── config.py
├── logging_setup.py
├── gamma.py
├── ws_polymarket.py
├── orderbook.py
├── paper_broker.py        # incl. settle_market for resolution payouts
├── news_store.py
├── data/
│   ├── rss.py
│   ├── bluesky.py
│   ├── fred.py
│   ├── bls.py
│   ├── congress.py
│   ├── courtlistener.py
│   ├── sec_edgar.py
│   ├── alchemy.py
│   ├── clob_history.py    # CLOB /prices-history
│   └── resolution_watcher.py
├── models/
│   ├── features.py        # question + book features
│   ├── lgbm.py            # LightGBM trainer + Predictor
│   └── chronos.py         # Chronos-Bolt zero-shot (opt-in via ENABLE_CHRONOS=1)
├── signals/
│   ├── news_match.py
│   ├── direction.py       # VADER + question-polarity classifier
│   └── stat.py            # P(YES) vs market price gap, log-only
├── strategies/
│   ├── yes_no_arb.py
│   └── news_trader.py
└── main.py

scripts/
├── backfill_resolutions.py   # Gamma /markets?closed=true bulk backfill
└── train_lgbm.py             # LightGBM training pipeline
```

## Pipelines (one-shot)

```bash
# Backfill thousands of historical resolutions in seconds
python -m scripts.backfill_resolutions --max 5000

# Train LightGBM on the resolutions table
python -m scripts.train_lgbm
```

## Known gaps

- **Alchemy key 403** — key is valid, but Polygon Mainnet is not enabled on the Alchemy app. Toggle it on at https://dashboard.alchemy.com/apps/8xf83y91az36bnyx/networks .
- **Reuters RSS feed URL stale** — DNS resolves NXDOMAIN. Other 22 feeds work.
- **Bluesky watchlist empty** — add DIDs to `data/bluesky_watchlist.txt`.
- **News matcher trades nothing** — by design. Builds labeled corpus first.
- **No resolution handling** — held positions stay at last mark when market resolves.
- **Top-of-book fill simulation is generous** — overstates live P&L by ~20–40%.
