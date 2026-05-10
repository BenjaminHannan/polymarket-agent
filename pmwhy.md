# Why Polyagent Isn't Making Money: A Senior-Quant Diagnosis

**Scope.** Treat this as a senior quant's review of a sophisticated junior project. The system is well-built, the methodology is mostly correct, and the author's self-diagnosis is largely on the right track. The bot is *not* making money for structural and methodological reasons that the literature predicts almost exactly. ~120 paper fills is statistical noise, the question-only ML ceiling is real, and the two strategies actually trading (`yes_no_arb`, sports-only `combined_trader`) sit in the most competitive niches on Polymarket. Where the literature is mixed, I say so. Where a "fix" is folklore, I flag it.

---

## TL;DR
- **The bot's flat ROI is exactly what the literature predicts** for a single-operator, taker-side, question-only ML system on Polymarket: documented retail-loss base rates are 69–87% on Polymarket/Kalshi, profits concentrate in <1% of wallets that are almost all bots with execution edges, and the IMDEA $40M arbitrage is overwhelmingly captured by sub-100ms operators — your taker-only, no-queue paper broker cannot compete in any of these niches.
- **The three highest-evidence fixes** are (1) **switch from taker to maker** (Polymarket's 2026 fee/rebate regime makes this the only niche with positive-expectation paper-to-live transfer), (2) **build queue-aware fill simulation** (paper-broker optimism is the single largest source of paper-to-live Sharpe degradation in the HFT/MM literature; hftbacktest is the canonical reference), and (3) **stop training new features** — at n=120 fills and Brier ~0.10, you are statistically blind to any edge smaller than ~3–5 percentage points; the author's "run for weeks, ≥500 trades" plan is directionally correct but actually understated (you likely need ≥1,500–3,000 resolved trades to deflate-Sharpe a real edge).
- **The other two priority levers (LambdaRank, RAG-LLM forecaster) are speculative and lack empirical support in prediction-market settings**; ForecastBench/Halawi-style RAG forecasters merely *approach* superforecaster Brier (~0.075–0.13), which is roughly the level Polymarket prices already imply for liquid markets — meaning even a state-of-the-art RAG forecaster gives you near-zero post-fee taker edge on the markets you're already trading.

---

## A) Diagnosis — Ranked by Likely Impact

### A1. (Highest impact) Execution structure: you are a paper-only, taker-only, no-queue operator in a venue where profit is concentrated in maker-side and latency-arbitrage flow

Bloomberg's 2026 analysis of every Polymarket wallet active since January 2025 found that >100,000 accounts lost ≥$1,000, retail traders lost a *combined $131M*, and the top 1% of accounts — almost all bots averaging 89 trades/day — captured most of the profits. Critically, Della Vedova's analysis is unambiguous: **bots had a 52% raw accuracy vs retail's 55% — they win not by predicting better but by executing better** (early entry, better prices). This is your structural problem in one sentence: your ML can be *more accurate than the bots* and still lose money to them on execution.

The IMDEA paper (Saguillo, Ghafouri, Kiffer, Suarez-Tangil, AFT 2025, arXiv:2508.03474) quantifies this. Of $39.59M in realized arbitrage profits over the 12 months ending April 2025, **$29M (73%) came from NegRisk multi-condition rebalancing and $10.58M (27%) from single-condition arb**, executed by a small handful of wallets — the top 3 wallets placed 10,200+ bets for $4.2M, ≈$400/trade. The InsiderSignal write-up corroborates the median arbitrage *opportunity window* collapsed from 12.3 seconds in 2024 to 2.7 seconds in early 2026, with 73% of profits captured by sub-100ms bots. Your `yes_no_arb` strategy is the simplest variety (single-condition rebalancing) and competes against these bots; on a paper broker with VWAP fills and no queue model, you will essentially never win the race.

The Bartlett & O'Hara (2026) paper *Adverse Selection in Prediction Markets: Evidence from Kalshi* (SSRN 6615739, 41.6M trades, 478,167 markets) is even more pointed: in single-name markets, traders buy YES 61% of the time and YES wins only 32%. Adverse selection (Kyle's λ, Glosten-Harris) is high, but **market makers earned roughly 2× per contract** what the spread alone would imply because of a "behavioral surplus" — the systematic YES overbet by retail cross-subsidizes makers. **This is direct evidence that the maker side is structurally profitable and the taker side structurally unprofitable**, even before fees. The Anatomy of Polymarket paper (Tsang & Yang, arXiv:2603.03136) shows this maturity is real and accelerating: arbitrage half-lives fell from hours to under a minute and Kyle's λ dropped from 0.53 to 0.01 over the 2024 election cycle. The market is becoming microstructurally efficient at the same time you're scaling up taker-side ML.

> *Note on the "Yang 2026 — $121 maker / $63 taker" figure cited in your audit log:* I can verify Tsang & Yang (2026) "The Anatomy of Polymarket" (SSRN 6336679 / arXiv:2603.03136) and that it provides per-contract maker/taker decomposition consistent with the directional finding (makers > takers per contract). I could not independently verify the exact $121/$63 numbers in public excerpts; the surrounding evidence (Bartlett-O'Hara, IMDEA, Bloomberg/Della Vedova, the InsiderSignal "0.55% of profitable maker wallets capture 50% of gains") all points the same direction, so the *qualitative* conclusion is robust even if the specific dollar figures need re-verification from the SSRN PDF directly.

**Impact ranking: this is by far the largest single explanation for flat ROI.**

### A2. (High impact) The ~120 paper-fill sample is statistical noise. You cannot detect a real edge at this n.

This is the cleanest, most defensible diagnosis. Bailey & López de Prado's Probabilistic/Deflated Sharpe Ratio papers (SSRN 2460551, 2326253, 2308659) and "Pseudo-Mathematics and Financial Charlatanism" (Notices of the AMS, 2014) show that **with only 7 strategy configurations tried, you'd expect to find a 2-year backtest with annualized Sharpe >1 even when the true OOS Sharpe is 0**. The Minimum Backtest Length (MinBTL) for a single strategy hovering around Sharpe 1 is on the order of years of daily data; for *paper trades on resolved Polymarket markets* with high outcome variance and binary settlement, the effective sample size requirement is even higher.

Two concrete numbers: (1) The sports_global certification at DSR=0.996, edge=+0.128 log-loss, n=626 sounds strong, but n=626 is the *training* fold count, not the OOS forward-trade count. The post-cert paper-trade sample is ~120; per Bailey/López de Prado, the *minimum* track-record length to distinguish Sharpe 1 from 0 at p<0.05, given typical Polymarket return non-normality (heavy tails, settlement-event clustering), is ~250–400 *independent* trades for a Sharpe 1 edge and substantially more for the Sharpe 0.3–0.5 edges that are realistic for question-only ML. (2) On Brier-score detectability: at typical Polymarket Brier ~0.10 with a hypothetical model improvement of 0.005 (i.e., your model is meaningfully better), the standard error of the mean Brier at n=120 is ~0.015 — **you cannot statistically distinguish a real Brier improvement of 0.005 from zero at this sample size**. The "On misconceptions about the Brier score" paper (Sci. Direct, 2025; PMC12818272) emphasizes this: Brier-score variance under realistic conditions does not shrink usefully without large n.

The author's stated "stop adding features, run for weeks, accumulate ≥500 resolved trades" is directionally right but **understated**. Realistic minimum: **1,500–3,000 fully-resolved, OOS-forward paper trades** before you can confidently deflate-Sharpe a 1–3 percentage-point Brier edge. PBO collapsing to 0.5 with a single config is itself a finding (Bailey et al. 2014): without genuine variation across hyperparameters that you actually trade, PBO is uninformative, and your effective sample size for overfitting tests is essentially zero.

### A3. (High impact) The "question-only model" ceiling is real and well-characterized

Wolfers & Zitzewitz's foundational survey (JEP 2004; "Prediction Markets in Theory and Practice", NBER 12083 2006) established the now-canonical empirical pattern: prediction markets **outperform most moderately-sophisticated benchmarks** as forecasters, and Tetlock's Tradesports work cited there finds that even mispricing in sports prediction markets is *not large enough to allow profitable trading strategies* after costs. Manski (2006) and the Wolfers & Zitzewitz "Interpreting Prediction Market Prices as Probabilities" paper (NBER 12200) further show market prices are typically within a couple percentage points of mean beliefs.

For LLM/RAG forecasters specifically: Halawi et al. (NeurIPS 2024, *Approaching Human-Level Forecasting with Language Models*, arXiv:2402.18563) — the gold standard — *approaches* human-crowd Brier and only beats the crowd in selective settings. ForecastBench (Karger et al., arXiv:2409.19839) shows GPT-4o/Claude-3.5 with retrieval roughly tie the median of *non-experienced* humans and lose to superforecasters. The 2025 follow-up (arXiv:2507.04562) puts frontier-model Brier at ≈0.075–0.13 on Metaculus-style questions vs the ForecastBench AIA Forecaster at 0.0753 (matching superforecaster 0.0740) — but on *market-liquid* questions the AIA Forecaster (0.1258) *trails the market consensus* (0.1106). **Translation**: the very best public LLM forecaster is *worse than the market price* on liquid markets — exactly the markets your bot is trading. Your `weather_llm_forecast`, `news_nli_match`, and even a hypothetical RAG-LLM expert face a hard ceiling: you can match the market, you cannot reliably beat it on the liquid slice.

Your `stat_lgbm` overall failure (+0.14 log-loss vs market on n=6,621 markets) is the in-sample manifestation of this ceiling. The sports_global slice survival (+0.128 log-loss edge) is consistent with the literature: sports markets are where *non-LLM* features (line-movement, base rates, Pinnacle-style closing-line value) historically win, and where the IMDEA paper notes "Sports are largely absent from arbitrage plots — maybe a less-explored venue."

### A4. (Medium-high) Selective signal layer issues: NLI verifier conservatism, keyword Jaccard ceiling, FinBERT direction-only signal

The keyword Jaccard hit rate of 43% (worse than coin-flip) is consistent with the broader benchmark: Aarab's BTZSC zero-shot benchmark (arXiv:2603.11991) finds NLI cross-encoders, rerankers, and LLMs all cluster around F1=0.6–0.72 on text classification — and sentiment benchmarks (FinancialPhraseBank) are *near-saturated*. MoritzLaurer's DeBERTa-v3-large-mnli-fever-anli-ling-wanli was state-of-the-art on ANLI as of 2022 and remains a sensible choice; the well-known "timid on real entailments" pathology of cross-encoder NLI on prediction-market questions is documented in the Hugging Face *NLI Cross Encoders: 6 Ways to Use Them* practitioner write-up (Lee Miller, HF blog) — the recommended fix is *prompt restructuring* (prefix the hypothesis with the temporal/question framing) rather than swapping the model. Krzemiński & Laurer (arXiv:2501.10860) show that **few-shot LLMs with PD/NLI hybrid prompts beat fine-tuned XLM-RoBERTa** (F1 97.2% vs 96.2%), suggesting your "LLM-tailored prompts" plan has empirical support and is preferable to model swaps.

The harder problem the literature flags is **claim–evidence temporal alignment**: the Kazinnik/Halawi 2025 audit of date-filtered web retrieval (cited in arXiv:2402.18563 follow-ups) found 71% of Google `before:` filter queries leak post-cutoff information into retrieval. Any RAG-augmented news layer needs strict temporal fencing or you're optimizing on labels.

### A5. (Medium) Methodological: NegRisk arb v2 partial-group artifact is correctly diagnosed; the LambdaRank pivot is plausible but unproven

You correctly falsified `NegRisk arb v2` at design stage — partial-group artifacts are exactly the kind of look-ahead bias that Polymarket's NegRiskAdapter `convertPositions` semantics introduce when you score on partially-resolved event groups. The IMDEA paper's $29M NegRisk-rebalancing figure is real but is captured by a handful of wallets executing atomic on-chain `convertPositions` calls — a paper broker streaming WSS cannot replicate this without simulating on-chain conversion cost, gas, and atomicity. Several practitioner sources (the `find-negrisk-opportunities` Go tool, Polymarket docs) are explicit that **NegRisk capital efficiency ≠ arbitrage** — using NegRisk for a sum<1 setup actually destroys the arb. Your falsification was correct.

LightGBM LambdaRank with NegRisk event grouping is *plausible* (LambdaRank with `group=event_id` is the canonical setup for ranking related instruments), but I could find **no published empirical evidence that LambdaRank beats binary-classification log-loss for prediction-market log-pool aggregation**. This is folklore-adjacent: the LightGBM docs (lightgbm.readthedocs.io) and Akash Dubey's practitioner guide describe LambdaRank for IR-style relevance ranking, not probability calibration. LambdaMART/LambdaRank optimizes NDCG-like metrics, **not log-loss or Brier**, so unless you write a custom Brier-aware ranking objective, switching loses calibration. *Recommendation: this is worth a one-week prototype but should not be prioritized over execution-side fixes.*

### A6. (Medium) Risk-layer fixes that probably don't help Sharpe

- **Selective abstention via Venn-Abers**: Vovk & Petej's Venn-Abers predictors (arXiv:1211.0025) and the SDM 2019 empirical paper (Lambrou et al.) show Venn-Abers is *the best calibrator* on probability estimation trees. The 2025 generalized Venn-Abers framework (van der Laan & Alaa, arXiv:2502.05676) extends to general loss functions. **However**, the trading-PnL evidence is thin: most published applications are medical/NLU calibration. The most relevant theoretical result for trading is the conformal-abstention / selective-classification literature (Geifman & El-Yaniv 2017; Tayebati et al. arXiv:2502.06884) showing that abstention improves *conditional accuracy* by 22% on hallucination detection — but this does not directly translate to Sharpe improvement, because in trading the abstained trades represent foregone PnL, and the surviving trades have reduced volatility but also reduced expectation. The honest answer: **Venn-Abers will help your Brier and your calibration plots; whether it materially improves Sharpe depends entirely on whether your edge is concentrated in the high-confidence tail, which at n=120 you cannot test.**
- **Kaminsky-Lo gated stops and BOCPD**: The author already falsified BOCPD. Trailing stops in binary-settlement markets are widely regarded in the prediction-market practitioner literature as folklore — the prediction-markets-reading repo (spfunctions/prediction-markets-reading) and Prediction Hunt's position-sizing guide both note that stops in event contracts often realize losses that mean-revert before resolution.
- **Drawdown-conditioned θ_min**: This is a sensible meta-control but reduces sample size further (the gates-thinning-the-trades problem).

### A7. (Lower-but-cumulative) Operational/infra tax on signal quality

The Eurico Paes (Medium 2024) and MMquant write-ups on async-WSS+SQLite trading bots, and the Jonathan Petersson dev.to *Real-Time Oracle Latency Bot for Polymarket* (which mirrors your setup uncannily) document the standard pathologies you're at risk of:

- **WSS "silent stall"**: connection alive, ping/pong fine, but upstream stops sending. Petersson's recommended fix (a `STALL_TIMEOUT` watchdog on `_last_price_ts`) is likely necessary for you. Without it, Polyagent can run for tens of minutes against stale books.
- **SQLite WAL contention** at ~25 background tasks is real but manageable with `aiosqlite` + a single writer queue; multi-writer contention will cause `SQLITE_BUSY` retries that delay decision logging and can push your VWAP fills to stale prices.
- **asyncio task starvation** when CPU-heavy tasks (LightGBM inference, FinBERT, NLI cross-encoder, gpt-oss-20b) block the event loop without `run_in_executor`. The websockets docs explicitly warn about this. On a single RTX 5070 Ti with 17GB, running gpt-oss-20b *and* Phi-4-mini *and* DeBERTa-NLI *and* bge-large concurrently will cause memory pressure that triggers GPU thrashing under load.
- **WSS reconnect storms** under upstream maintenance can cascade to dropped fills. The Polymarket changelog notes the V2 cutover (April 28, 2026) and continuing rolling fee changes — your simulator's fee model needs to mirror per-category curves (Crypto 1.80% peak, Sports 0.75%, Politics 1.00%) or you'll mis-estimate edge by 30–100 bp per trade.

These don't *individually* explain flat ROI but they degrade signal quality silently and amplify all the other problems.

---

## B) Fixes with Citable Evidence

### B1. Switch from taker to maker. Highest evidence, highest expected impact.

- **Polymarket Maker Rebates Program docs** (help.polymarket.com/articles/13364471) — makers pay zero, receive 20–25% of taker fees as daily USDC rebates, calculated per-market. This *is* a structural edge for any operator that can keep limit orders alive at competitive levels.
- **Bartlett & O'Hara (2026)** SSRN 6615739 — makers earn ~2× spread per contract due to behavioral surplus from systematic YES overbets.
- **The "defiance" Polymarket-news write-up (*Automated Market Making on Polymarket*)** is the most useful practitioner reference: documents that there were "only 3–4 serious liquidity providers" on Polymarket pre-2026, and gives concrete strategy parameters (low-volatility markets, both-sided quotes for ~3× rebate density). Caveat: post-2024 election rewards decreased; the warproxxx/poly-maker repo's README is brutally honest — *"In today's market, this bot is not profitable and will lose money. Use it as a reference implementation."* This is a reality check: maker-side is structurally favored but not a free lunch in 2026's compressed-spread environment.
- **Avellaneda & Stoikov (2008)** is the canonical reference for inventory-aware MM; for prediction markets specifically, the discrete tick size and binary settlement violate the AS Brownian assumption, so use AS as a sizing/skew framework, not a pricing model.

**Recommended action**: build a `passive_poster_v2` that actually trades (currently log-only), targeting low-volatility, high-rebate slices. Use hftbacktest's queue-position models (next item) to size correctly.

### B2. Build queue-aware fill simulation. Highest-evidence methodological fix.

- **hftbacktest** (github.com/nkaz001/hftbacktest, ReadTheDocs) is *the* open-source reference — full L2/L3 reconstruction, configurable queue models (RiskAdverse, Power, Log), tick-by-tick simulation. Its docs are explicit: *"Without queue modeling, backtests optimistically assume instant fills when your price is reached. This drastically overstates HFT profitability."* This is a direct quote-of-record on your `PaperBroker`'s VWAP-fills assumption being optimistic.
- **ABIDES / ABIDES-Gym / ABIDES-Markets** (arXiv:1904.12066, 2110.14771; github.com/jpmorganchase/abides-jpmc-public) — JPMorgan's open agent-based discrete-event simulator; supports configurable per-agent network latency. More work to integrate but better for prediction-market dynamics with heterogeneous agents.
- **NautilusTrader Polymarket adapter** (nautilustrader.io/docs/integrations/polymarket) is a more pragmatic option: production-grade, has real Polymarket signing/order semantics, and the evan-kolberg/prediction-market-backtesting repo extends it specifically for Polymarket backtests.
- **Almgren-Chriss (2000/2001)** and the Guéant-Lehalle-Fernandez-Tapia closed-form market-making formulas (arXiv:1105.3115, 1605.01862) for the theory.
- **Quantopian's 2014 disclosure** (and the Paybis paper-vs-live guide, traderspost.io blog) document the empirical Sharpe-degradation pattern: high-turnover paper-trade Sharpes of 3.0+ collapsing to negative live; in low-liquidity venues 40–60% Sharpe loss is typical. Your VWAP-fill `PaperBroker` is *exactly* the kind of optimistic simulator these warn about.

### B3. Statistical rigor: the author's "stop adding features, run for weeks" plan is correct but needs to be tightened.

- **Bailey & López de Prado (2014)** SSRN 2460551 (DSR) and 2326253 (PBO) — *required reading*. The DSR formula automatically incorporates skewness, kurtosis, and number-of-trials; your sports_global cert at DSR=0.996 is meaningful only if you have correctly counted *all* configs you tried before that one.
- **Bailey/Borwein/López de Prado/Zhu (2014)** "Pseudo-Mathematics" Notices AMS — the Minimum Backtest Length result. Use it to set a *target trade count* before any new feature is allowed in.
- **Lopez de Prado, Advances in Financial ML (2018)**, Ch. 7 (CPCV) — your 8-fold purged CPCV is correct; the additional discipline is to **log every config tried** (PBO denominator) and use that to deflate Sharpe.
- **Practical recommendation, with citations**: target ≥1,500 OOS forward paper trades on `combined_trader` before re-evaluating, with strict feature-freeze. ≥500 (the author's number) is the *floor* for detecting Sharpe ≥1; you should expect Sharpe ≤0.5 on a question-only ML system in this venue, which approximately quadruples the n required.

### B4. News verifier: prefer prompted LLM over cross-encoder swap

Empirical evidence:
- **Krzemiński & Laurer (2025)** arXiv:2501.10860 — Mistral with PD prompt few-shot reaches F1 95% with 10 examples; Gemini few-shot with NLI+PD prompt template reaches F1 97.2%, beating fine-tuned XLM-RoBERTa.
- **Aarab BTZSC (2026)** arXiv:2603.11991 — Qwen3-Reranker-8B reaches macro F1=0.72 across 22 datasets; cross-encoder NLI lags rerankers and instruction-tuned LLMs in zero-shot.
- **Pillai 2023** arXiv:2305.16633 — for *financial* tasks specifically, ChatGPT zero-shot does not beat fine-tuned RoBERTa, but the gap narrows on tasks with little public training data.

**Concrete recommendation**: keep DeBERTa-v3-base-mnli-fever-anli as a *fast* first-stage filter, and add an LLM second-stage with a hybrid PD+NLI prompt for the cases the cross-encoder marks as "uncertain." This preserves your latency budget on the GPU while improving precision on the entailments DeBERTa is timid on.

### B5. Kelly sizing — you're probably already doing this approximately right

- **Kelly's original 1956 formulation**, MacLean/Ziemba/Blazenko (Mgmt Sci 1992) — *full* Kelly creates a 33% probability of halving the bankroll before doubling. Use **half- or quarter-Kelly**.
- arXiv:2412.14144 "Application of the Kelly Criterion to Prediction Markets" — formal analysis specific to prediction markets, including the price-vs-probability gap (prices ≠ probabilities under risk aversion or wealth effects).
- Practitioner consensus from prediction-hunt, marketmath.io, agentbets.ai, rekko.ai is uniform: **fractional Kelly (0.25–0.5×) with a portfolio cap of 20–25% across correlated bets**.
- *Folklore alert*: Kelly sized on a *Brier-margin* edge ("p̂ − market") is highly sensitive to calibration error; 1% calibration drift can flip Kelly sign. Pair Kelly with Venn-Abers calibration explicitly.

### B6. NegRisk arbitrage — only if you add atomic on-chain execution

The IMDEA $29M NegRisk-rebalancing figure is real but **it is captured atomically via `convertPositions`** (Polymarket/neg-risk-ctf-adapter Solidity contract). A paper broker that simulates this with "buy then sell across legs" will systematically over-estimate edge because the IMDEA arbitrage *requires* the atomicity guarantee. Yang's "Decoding the Digital Tea Leaves" (yzc.me) and the Polymarket docs both spell this out. Your falsification of NegRisk arb v2 is correct *for your current execution stack*; revisiting requires building real on-chain execution, which is out of scope for paper-trading mode.

---

## C) What to Stop Doing — Validated Against the Literature

The author's instruction is: *"stop adding features, run for weeks, accumulate ≥500 resolved trades, then re-evaluate."*

**Verdict: directionally correct, numerically optimistic.** Specific corrections:

1. **≥500 trades is the floor for detecting Sharpe ≥1**, which is implausible for a single-operator question-only ML system in a venue where the *best* known LLM forecasters trail market consensus on liquid markets. For realistic edges (Sharpe 0.3–0.5, Brier improvement 0.005–0.02), you need ≈1,500–3,000 *fully-resolved* OOS forward trades.
2. **Feature-freeze must be strict**. Bailey/López de Prado are explicit: every config you try before "the one you trade" inflates the multiple-testing burden in DSR. If you've tried ≥7 configurations on sports_global, your reported DSR=0.996 is already optimistic.
3. **Stop running PBO with one config**. PBO needs a *grid* of plausible-but-not-overfitted configs (n_splits ≥100, per the QuantBeckman CPCV write-up) to be informative. PBO=0.5 with one config is uninformative, but adding configs *for the sake of PBO* is itself overfitting. The honest fix is: declare a small ex-ante config grid (e.g., 5 reasonable hyperparam choices), run them, report PBO, and freeze.
4. **Log-only signals (`stat_signal`, `news_keyword_match`, `news_nli_match`, `passive_poster`, `stop_loss`) should remain log-only until they accumulate ≥1,500 paper trades each.** Otherwise you're just generating more multiple-testing burden.
5. **Stop hoping the LLM forecaster will rescue this.** Halawi et al. (NeurIPS 2024) and ForecastBench show retrieval-augmented LLMs ≈ market on liquid prediction markets. Your gpt-oss-20b/Phi-4-mini stack is an *order of magnitude* below the frontier (GPT-5/o3/Claude-4.5/Gemini-3 are the relevant comparison group). It is a research project, not a profit lever.

---

## D) Execution Margin — Operational Pitfalls on a Single Consumer GPU

From the Polymarket-bot practitioner literature:

- **Petersson dev.to** — concrete recipe: stall-timeout watchdog on `_last_price_ts`; force-reconnect on staleness; per-WSS keepalive PING (Polymarket RTDS requires PING every 5 seconds, *not* a WebSocket-level ping but a literal "PING" string).
- **MMquant orderbook-replication write-up** — separate WSS connections per token (not multiplexed) for resilience.
- **websockets 16.0 docs** — `async for websocket in connect(...)` for automatic exponential-backoff reconnect; `process_exception` to distinguish transient vs fatal.
- **SQLAlchemy asyncio docs / aiosqlite** — single writer task, multi-reader; never share a connection across tasks.
- **GPU memory at 17GB**: gpt-oss-20b (~14GB Q4) + Phi-4-mini (~3GB Q4) + bge-large (~1.4GB) + DeBERTa-v3-base-NLI (~0.5GB) + LightGBM (CPU) + FinBERT (~0.4GB) is right at the ragged edge. Under attention spikes you will OOM. Practical fixes: serialize LLM calls behind a queue, quantize aggressively (Q4_K_M), and never run two transformer forwards concurrently.
- **The `stop_loss` log-only is correct**: trailing stops on binary-settlement contracts are widely panned in the prediction-market practitioner literature (see prediction-markets-reading repo, spfunctions/prediction-markets-reading) as a way to lock in unrealized losses that would mean-revert to resolution.

None of these *individually* explain flat ROI, but cumulatively they introduce 50–200 bp/trade of silent edge erosion that, on a system whose true edge is plausibly <100 bp/trade, easily flips the sign.

---

## E) The Big-Picture Question: Can a Single Retail Operator Profitably Trade Polymarket in 2026 with Question-Only ML + News + Paper Execution?

**Answer: almost certainly not via the path you're on, plausibly yes via the three paths the author already identified — and the literature is clear on which.**

The empirical base rates:
- Bloomberg/Della Vedova (2026): 100,000+ Polymarket accounts lost ≥$1,000 since Jan 2025; retail aggregate loss = $131M; top 1% (almost all bots) take most of the profits. Bot accuracy 52% < retail accuracy 55% — bots win on *execution*, not prediction.
- Akey, Grégoire, Harvie, Martineau (SSRN 6443103, *Who Wins and Who Loses on Polymarket*, March 2026): ~69% of traders lose money since 2022; top 1% capture ~75% of profits. Losses are "associated primarily with liquidity-taking; longshot betting plays a smaller role once we control for activity scale." For 1-in-5 losers, **the lower-bound cost of taking liquidity alone is enough to flip their PnL from negative to positive** — i.e., switching to maker would make them profitable. This is the single most important paper for your situation; read it twice.
- KuCoin/Becker analysis (72.1M Kalshi trades): takers had a +2.0% advantage in 2022, now -1.12%. Maker-taker has flipped over the last three years.
- Solidus HALO / CoinDesk (April 2026): on Polymarket politics Dec 2025–Feb 2026, **0.55% of profitable maker wallets captured 50% of maker gains**, ~$8M of $16M. Concentration is extreme even within the winning niche.

**Of the three viable paths the author identified:**

1. **Certified narrow slices** (current sports_global). Path of least resistance, but the literature predicts this slice is small and shrinking as market efficiency improves (Tsang & Yang's Kyle-λ collapse). Expect Sharpe 0.3–0.6 *if* DSR holds out-of-sample, with substantial regime risk. **Realistic — but do not expect to scale.**

2. **Maker-side spread capture**. Strongest theoretical and empirical support (Bartlett-O'Hara, Akey et al., Polymarket Maker Rebate docs, Bloomberg/Della Vedova). The 2026 fee/rebate regime explicitly subsidizes this. **This is where I would put 70% of the engineering effort.** Risks: Polymarket V2 cutover (April 2026), spread compression as more bots enter, and "warproxxx/poly-maker" caveat that returns have already compressed.

3. **NegRisk arbitrage with on-chain execution**. The IMDEA $29M figure is real, but the 2.7-second median window and sub-100ms-bot dominance mean retail-grade infrastructure cannot capture this. The right framing: NegRisk *capital efficiency* (1.0 collateral instead of N×0.5) is a real benefit, but treating it as a profit center requires colocated low-latency on-chain execution that is incompatible with paper trading mode. **Do not pursue this in 2026 unless you fundamentally reposition the project.**

**The strict honest answer**: the bot is not making money because (a) you are competing in the most-bot-saturated niches with paper-quality infrastructure, (b) your sample size cannot statistically detect any edge plausible for a question-only system, and (c) the market microstructure literature published in 2026 — Bartlett-O'Hara, Akey et al., the Bloomberg analysis — independently converges on the conclusion that retail taker-side trading on Polymarket is *structurally* unprofitable, while maker-side has a small but real edge concentrated in <1% of operators.

The single highest-leverage move is not LambdaRank or RAG-LLM. It is to **rebuild the execution stack with hftbacktest-style queue-aware fill modeling, switch the trading layer to maker-side, and treat all of stat_signal/news_*/lgbm/llm-forecaster as auxiliary features for inventory skew rather than as standalone alpha sources**. The author's "honest performance ceiling" of Brier 0.09–0.11 and ROI ~flat ±2% per session is *exactly what the literature predicts* for the current architecture; the gap to profitability is structural, not algorithmic.

---

## Caveats

- The "Yang 2026 — $121 maker / $63 taker" specific dollar figures could not be independently verified in public excerpts; the qualitative finding (maker > taker per contract) is robust across at least three independent 2026 papers (Tsang-Yang, Bartlett-O'Hara Kalshi, Akey et al. Polymarket).
- ForecastBench/AIA-Forecaster numbers (Brier 0.0753 vs 0.0740 superforecaster, vs 0.1258 on market-liquid) come from an "Emergent Mind" topic page citing Alur et al. (Nov 2025) and Yang et al. (Oct 2025); I did not verify the underlying papers directly. Treat as directional evidence.
- The InsiderSignal "12.3s → 2.7s arbitrage window" figure is sourced from a third-party trading-infrastructure marketing page citing IMDEA; the IMDEA paper itself reports the 75% of bids fall within 950 blocks (≈1 hour) window, not the sub-second figures. The 2.7s figure may be from a separate 2026 study but I couldn't trace primary source.
- LambdaRank for prediction-market log-pool aggregation has no peer-reviewed empirical evidence either way. The recommendation to deprioritize it is based on the *absence* of evidence, not evidence of failure.
- Venn-Abers improving trading Sharpe (vs Brier/calibration) is similarly not directly evidenced in the literature; the recommendation is hedged.
- Polymarket's fee/rebate regime is in active flux (multiple changes in 2026 per docs.polymarket.com/changelog and the V2 cutover April 28, 2026); any maker-side strategy must re-verify fee parameters before deployment.
- The Bloomberg/Della Vedova analysis covers 2025 onward; pre-2025 base rates may differ. The 87% loss rate predates Della Vedova's analysis and comes from Layerhub/Cryptonews 2024 — different methodologies; the *direction* of the finding (vast majority lose) is stable across all sources.
- All sample-size calculations for "minimum trades to detect edge" are order-of-magnitude estimates assuming roughly i.i.d. binary outcomes with Brier ~0.10; clustered settlement events (e.g., correlated NFL Sunday) will require even larger n. The 1,500–3,000 figure is conservative; for correlated baskets, multiply by 2–3×.