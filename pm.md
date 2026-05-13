# Polyagent — Sharpe-Ratio Improvement Design Document

**Status:** Draft v1 — Paper-trading scope only.
**Audience:** Polyagent author (sophisticated; has shipped CPCV, conformal-Kelly, Karkare NegRisk, Cont/Kukanov/Stoikov, AIA debiasers).
**Goal:** A prioritized, implementable set of changes that raise Sharpe = mean(r) / std(r) of paper-session returns, given the existing system's documented diagnostics (three losing sessions: −3.7%, −2%, −5%; bot Brier 0.13–0.18 vs market 0.05–0.10; slippage burn 22% in worst session).
**Author's note on epistemics:** every Sharpe-lift figure below is an *order-of-magnitude estimate* derived from analogous published results, not a guarantee. With ~120 fills the *Probabilistic Sharpe Ratio* (Bailey & López de Prado 2012) of any single session is essentially uninformative; honest evaluation requires Minimum Track Record Length analysis (see §12).

---

## TL;DR — Top 10 Sharpe-Impact Changes, Ranked

| # | Change | Mechanism | Est. Sharpe Lift | Effort | Risk |
|---|--------|-----------|------------------|--------|------|
| 1 | **Selective abstention layer** with conformal-coverage gate (§1) | Cuts trade frequency 40-70% on lowest-confidence trades → kills variance, raises mean | +0.4 to +0.8 | 2-3 days, ~400 LOC | Low |
| 2 | **Cross-market NegRisk simultaneous-fill arb scanner** (§4 / §11) | Replaces directional Brier-loss bets with sum-to-1 mispricing; near-zero variance edge | +0.5 to +1.0 | 4-6 days, ~700 LOC | Medium (paper-fill assumption) |
| 3 | **Volatility-targeting layer** on top of conformal-Kelly (§2) | Equalizes per-trade dollar-vol; cuts tail variance per Moreira-Muir 2017 / Harvey 2018 | +0.3 to +0.6 | 1-2 days, ~150 LOC | Low |
| 4 | **Deflated-Sharpe / PSR live evaluation harness** (§12) | Stops the system from chasing noise; reroutes effort honestly | Indirect (prevents −Sharpe overfits) | 2 days, ~250 LOC | Low |
| 5 | **BOCPD regime gate with auto-deleverage** (§9) | Detects edge decay; cuts size before drawdown completes | +0.2 to +0.5 (mainly variance cut) | 3 days, ~350 LOC | Low |
| 6 | **TabPFN-2.5 as 5th expert in log-pool** (§6) | Decorrelated tabular forecast; ensemble variance ↓ | +0.15 to +0.35 | 2 days, ~200 LOC | Low (paper sim) |
| 7 | **Smart-money replication signal as feature** (§7) | Wallet-level orthogonality flag (Della Vedova 2026) → real edge | +0.2 to +0.4 | 3 days, ~300 LOC | Medium (look-ahead risk) |
| 8 | **Online conformal (Gibbs-Candès 2024) replacing static three-tier** (§8) | Coverage holds under regime shift → sizing stays calibrated | +0.1 to +0.3 | 2 days, ~200 LOC | Low |
| 9 | **Event-id correlation-aware sizing (risk parity on NegRisk groups)** (§3) | Cuts portfolio variance from cluster-correlated bets | +0.15 to +0.35 | 2 days, ~200 LOC | Low |
| 10 | **Replace fixed stop-loss with event-driven exit + AR(1)-aware hold** (§10) | Per Kaminski-Lo 2014, stops only help when φ ≥ SR; else they bleed mean | +0.05 to +0.25 (mostly removes mean drag) | 1 day, ~100 LOC | Low |

**Net expected Sharpe lift if all ten ship and stack independently (they don't fully):** roughly +1.0 to +2.5 in annualized Sharpe over the current losing baseline. The bulk comes from (1), (2), (3) — selectivity, arb, vol-targeting. Items (4) and (12) are not Sharpe-additive but Sharpe-*honesty* additive: they prevent you from believing in lift that isn't there.

---

## Theoretical Frame — Where Sharpe Comes From in Paper Mode

Sharpe = E[r] / σ(r). With ~120 fills, the *standard error* of an estimated SR is approximately √((1+SR²/2)/T) ≈ 0.10 even at the true SR. So *any* observed Sharpe below ~0.4 is statistically indistinguishable from zero. The sessions you've run (−3.7%, −2%, −5%) are *consistent* with a true Sharpe in [-2, +0.5]; you cannot reject either skill or anti-skill from this sample. **This is the single most important fact in the document.**

The bot's documented "honest ceiling" (question-only ML → ~0 ROI; real edge in maker-side and NegRisk arb) is consistent with three converging 2026 papers:

- **Yang (Augusta, March 2026)** — top-5% skilled traders earn $121/market as makers vs $63 as takers; *trader skill, not the maker-taker distinction, determines who profits*. The maker premium is conditional on skill.
- **Della Vedova (SSRN 6191618, Feb 2026)** — bots have captured +$136M while retail lost $82M; profits come from *execution*, not directional information; 1.30% of wallets show predictive accuracy too high to be luck.
- **Akey, Grégoire, Harvie, Martineau (SSRN 6443103, March 2026)** — top 1% capture 76.5% of gains; "a 1-σ increase in maker-volume share lowers loss probability by 9.3 ppt"; ~20% of losers would flip to profitable if liquidity-taking cost were eliminated.

In paper mode you cannot collect maker rebates, and your "passive_poster" simulator already approximates queue-aware fills via Cont/Kukanov/Stoikov. So in paper mode the *only* unbiased Sharpe sources remaining are:

1. **Selectivity** — fewer, better trades.
2. **Arbitrage** — sum-to-1 / NegRisk consistency (paper sim approximates this if simultaneous-fill assumption is honored).
3. **Variance reduction** — vol-targeting, correlation-aware sizing, ensemble diversification.
4. **Calibration honesty** — sizing matched to true uncertainty.

Everything below is structured to attack one of these four levers.

---

## 1. Selective Abstention with Conformal Coverage Gate

**One-liner:** Add a selective-classification layer that abstains when the conformal interval around the combined probability is wider than a coverage-controlled threshold; only ~30-40% of currently-considered trades pass.

### Theoretical rationale
Selective classification (Chow 1970; El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017, NIPS) trades coverage for selective risk: at coverage c, the selective error on the top (1−c)·N most-confident predictions is markedly lower than the unconditional error. Bai & Jin (2026) "Conformal Selective Prediction with General Risk Control" (arXiv 2603.24704) extend this with e-value-based bounds suitable for online deployment. Chalkidis & Savani (2021) "Trading via Selective Classification" (arXiv 2110.14914) show the *Sharpe* improvement is monotone in coverage cuts for sparse-edge regimes — exactly Polyagent's setting (Brier gap 0.13 vs 0.10 means edge-per-trade ≈ 0.005-0.02).

In a sparse-edge regime, abstaining on the noisiest 50% of signals roughly doubles mean-edge per remaining trade while only halving trade count, giving a Sharpe boost ≈ √2 ≈ 1.4× before considering variance reduction from skipping high-uncertainty trades (which tend to be the highest-variance ones).

### Why it raises Sharpe
- **Mean lift:** the bot's combined-probability calibration shows the largest Brier improvements concentrated in the top quartile of confidence (typical of all calibrated predictors). Abstaining on bottom quartile preserves nearly all of mean edge.
- **Variance cut:** the highest-conformal-interval trades are also the highest-σ trades (price moves a lot when the market is also uncertain).

### Implementation sketch

```python
# polyagent/risk/selective_gate.py  (NEW)
from polyagent.calibration.conformal import conformal_interval

class SelectiveGate:
    def __init__(self, target_coverage=0.40, alpha=0.10):
        self.target_coverage = target_coverage   # take only 40% of candidates
        self.alpha = alpha                        # 90% conformal coverage
        self.recent_widths = collections.deque(maxlen=2000)

    def width(self, p_hat, x_features):
        lo, hi = conformal_interval(p_hat, x_features, alpha=self.alpha)
        return hi - lo

    def threshold(self):
        # quantile that admits target_coverage fraction
        if len(self.recent_widths) < 200:
            return float('inf')   # don't gate until burn-in
        return np.quantile(self.recent_widths, self.target_coverage)

    def admit(self, p_hat, x_features) -> bool:
        w = self.width(p_hat, x_features)
        self.recent_widths.append(w)
        return w <= self.threshold()
```

Wire into `polyagent/strategies/combined_trader.py` *before* the fee-adjusted edge check:

```python
if not selective_gate.admit(p_combined, features):
    continue   # abstain
```

### Effort, lift, failure modes
- Effort: 2-3 days, ~400 LOC including unit tests on cached resolved markets.
- Sharpe lift: **+0.4 to +0.8** (most likely single biggest lever; high-confidence claim because Brier-vs-market gap data already shows where the mean edge concentrates).
- Failure modes: (a) over-abstention on novel-event markets where conformal width is structurally large but edge is real; mitigate with category-conditional thresholds. (b) selective-bias amplification across groups (Jones et al. 2021, ICLR) — log per-category coverage and accuracy.

---

## 2. Sharpe-Optimal Position Sizing Beyond Kelly (DRO + Vol-Target Stack)

**One-liner:** Replace the current full-Kelly + drawdown-conditioned scale with a three-layer stack: (a) Wasserstein-DRO Kelly fraction from the conformal interval, (b) per-trade vol-target overlay, (c) portfolio-level vol-target.

### Theoretical rationale
- **Conformal-Kelly is conservative-correct on the lower bound** (you've already shipped it via Vovk 2025 conformal e-prediction), but classical Kelly with a *point* estimate p̂ over-bets when p̂ is overconfident. The Wasserstein-Kelly DRO formulation (Li 2023, arXiv 2302.13979; Sun & Boyd 2018, "Distributional Robust Kelly Gambling," arXiv 1812.10371) maximizes worst-case log-growth over a Wasserstein ball of radius ε around the empirical p̂. For binary contracts this collapses to a closed form:

  f*_DRO = max(0, (p_lo · b − (1 − p_lo)) / b)

  where p_lo is the conformal lower bound and b is the binary payoff. **You already do this.** The improvement is to *additionally* layer:

- **Vol-target overlay (Moreira & Muir 2017, JoF; Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert 2018):** scale each trade's notional inversely to recent realized vol of similar trades (same category, same edge bucket). Moreira-Muir documented that for risk assets, vol-targeting raises factor Sharpe by 0.1-0.4 because changes in volatility are not fully offset by proportional changes in expected return. Cederburg, O'Doherty, Wang & Yan (2020, JFE) caution this is *not* universal — but for trading systems with empirically *negatively-skewed* returns (Polyagent's slippage-burn profile), the conditional-vol-target variant of Bongaerts et al. (2020) and Xu (2024, "Improving Volatility-Managed Portfolios in Real Time") consistently helps. The paper-trading context guarantees no fee-induced turnover penalty, removing the largest objection to vol-targeting.

- **Portfolio-level vol cap:** post-summation cap so total daily P&L vol stays below a target σ_target (e.g. 1.5% NAV). This is where most of the documented session variance (−5% session) was unforced.

### Why it raises Sharpe
Mean-variance theory: max E[r]/σ(r) is achieved by *equalizing risk contribution* across periods, not equalizing capital. Polyagent currently equalizes capital (via Kelly cap) which means high-vol periods dominate σ.

### Implementation sketch

```python
# polyagent/sizing/vol_target.py  (NEW)
class VolTargetStack:
    def __init__(self, target_per_trade_vol=0.008, target_portfolio_vol=0.015,
                 ewma_halflife=20):
        self.tpt = target_per_trade_vol
        self.tpv = target_portfolio_vol
        self.cat_vol = {}   # category -> EWMA realized P&L vol per $ traded

    def update_realized(self, category, pnl_per_dollar):
        prev = self.cat_vol.get(category, 0.01)
        a = 1 - 0.5 ** (1 / 20)
        self.cat_vol[category] = (1 - a) * prev + a * abs(pnl_per_dollar)

    def trade_scale(self, category):
        v = max(self.cat_vol.get(category, 0.01), 1e-4)
        return min(1.0, self.tpt / v)        # cap at 1× (don't lever in paper)

    def portfolio_scale(self, current_open_positions):
        # estimate aggregate vol assuming intra-category corr ≈ 0.6, cross ≈ 0.1
        ... (Markowitz with shrinkage)
```

Hook in `polyagent/sizing/kelly.py`:

```python
f_dro = conformal_kelly_lower(p_lo, b)
f_voltarget = vt.trade_scale(category) * f_dro
f_final = vt.portfolio_scale(open_positions) * f_voltarget
```

### Effort, lift, failure modes
- Effort: 1-2 days, ~150 LOC.
- Sharpe lift: **+0.3 to +0.6**.
- Failure modes: vol-target chases its own tail in regime shifts (mitigate with BOCPD reset, §9). Cederburg et al. caution: gains may come from a long-short *combination* of scaled and unscaled, which is not implementable as a directional signal in a binary contract — so expect the lower end of the range.

---

## 3. Portfolio Construction for Correlated Binary Contracts

**One-liner:** Treat each `event_id` as a cluster; apply equal-risk-contribution sizing across clusters; use NegRisk-group block-diagonal covariance with Ledoit-Wolf shrinkage.

### Theoretical rationale
Correlation-aware sizing is risk parity (Maillard, Roncalli & Teïletche 2010; Roncalli 2013). For binary contracts inside a NegRisk group of N tokens, payoffs are *exactly anti-correlated* (Σpᵢ = 1), so the within-group covariance is rank-deficient and known in closed form. Across NegRisk groups, correlation is dominated by event-cluster (e.g., all "election" markets co-move on poll surprises). Paolella et al. (2025, JTSA) develop fat-tailed risk-parity that is robust to volatility shocks — the relevant regime here.

The Akey et al. (2026) finding — top 1% capture 76.5% of profit, mostly via a few high-conviction event-clusters — implies *concentration* is rewarded for skill but punishes naive bots. Risk parity gives Polyagent the diversification benefit *without* requiring it to identify which event cluster has true edge.

### Why it raises Sharpe
The −5% session was driven by a single event-cluster blowup. ERC sizing would have capped that cluster's contribution to portfolio σ at 1/K of total, mechanically reducing tail draw.

### Implementation sketch

```python
# polyagent/portfolio/erc.py (NEW)
def erc_weights(sigma_matrix, event_clusters):
    # block ERC: solve f s.t. f_i * (Σf)_i = const ∀ i
    # closed form for diagonal-dominant case; iterate for full case
    n = sigma_matrix.shape[0]
    f = np.ones(n) / n
    for _ in range(50):
        marginal = sigma_matrix @ f
        f = f * (1 / marginal)
        f = f / f.sum()
    return f

# Hook in polyagent/strategies/combined_trader.py before order send:
proposed = collect_proposed_trades()
sigma = build_block_cov(proposed, event_clusters, intra_corr=-1.0/N_neg, inter_corr=0.15)
weights = erc_weights(sigma, [t.event_id for t in proposed])
final_sizes = [t.kelly_size * w for t, w in zip(proposed, weights)]
```

### Effort, lift, failure modes
- Effort: 2 days, ~200 LOC (includes Ledoit-Wolf shrinkage from sklearn).
- Sharpe lift: **+0.15 to +0.35**.
- Failure modes: covariance estimation with N≈120 fills is unstable; *use shrinkage with high λ ≈ 0.8 toward block-diagonal prior*.

---

## 4. NegRisk + Sum-to-1 Arbitrage Scanner at Scale

**One-liner:** Build an O(N log N) cross-market NegRisk scanner that checks `|Σpᵢ − 1| > θ_min` continuously across all 500 streamed markets and books arb pairs in paper mode whenever the simultaneous-fill assumption holds.

### Theoretical rationale
The IMDEA Software Institute analysis (Karkare et al., "Unravelling the Probabilistic Forest," arXiv 2508.03474, 2025) documented **$39.59M of arbitrage** extracted from Polymarket between April 2024 and April 2025, with **$29M (73%) coming from NegRisk rebalancing**. The top performer extracted $2,009,632 across 4,049 transactions averaging $496 per trade — a setup *paper trading can simulate honestly* because the arbitrage involves a NegRisk-group price sum exceeding 1.0 by more than fees, not a queue-position effect.

For NegRisk markets the legs are mutually exclusive ⇒ buying ALL N "NO" tokens (or equivalently, all N "YES" tokens with NegRisk conversion) costs Σpᵢ and pays $1 with certainty. Paper mode: assume fill at displayed best ask, conservatively haircut by 1× the spread to model adverse selection — this matches Cont & Kukanov (2013) optimal placement assumptions.

The IMDEA window closing characteristics (median 200ms) means *real-money* execution requires <50ms latency. **In paper mode the latency assumption can be tuned: assume "fill if opportunity persists ≥ T_fill seconds at displayed price" with T_fill ∈ {0.5, 2, 5} as a sensitivity sweep. This is honest because it makes the resulting Sharpe a function of an explicit assumption.**

### Why it raises Sharpe
Arb returns are by construction near-zero variance per trade (the only variance is fill timing and resolution risk). Even with conservative paper-fill haircuts, a 0.5-2% per-trade return at 5-50 events per day yields Sharpe well above 1, dwarfing directional edge.

### Implementation sketch

```python
# polyagent/strategies/combinatorial_arb_v2.py (REPLACEMENT)
class NegRiskArbScanner:
    def __init__(self, fill_persistence_sec=2.0, haircut_bps=20):
        self.fill_persistence = fill_persistence_sec
        self.haircut = haircut_bps / 10000

    async def scan(self, neg_risk_groups):
        for group in neg_risk_groups:
            best_asks = [book.best_ask() for book in group.books]
            sum_p = sum(best_asks) + len(group) * self.haircut
            if sum_p < 1.0 - FEE_RATE:
                edge = 1.0 - sum_p
                # liquidity-constrained sizing
                max_size = min(book.depth_at_best for book in group.books)
                size = max_size * 0.5  # take half the thinnest leg
                if self._persistent(group, persistence=self.fill_persistence):
                    self._book_paper_arb(group, best_asks, size, edge)
```

Important details:
1. **Persistence check** (`_persistent`): require all N legs to be available at quoted prices for ≥ T_fill seconds — this is Polyagent's honest paper-mode proxy for real execution latency.
2. **Liquidity-floor**: size = min(depth across all N legs) — exactly matches the IMDEA paper's documented constraint.
3. **NegRisk-augmented markets**: query the `enableNegRisk` field; for those, add the conversion path that lets you go from N-1 NO → 1 YES + (N-2) USDC.
4. **Mind Della Vedova's caveat**: of 13 LLM-detected combinatorial dependencies in the 2024 election, only 5 generated profit — a 62% false-positive rate. Use only the IMDEA-validated NegRisk class for the first version; add LLM-detected combinatorial pairs in a v2 with much higher confidence thresholds.

### Effort, lift, failure modes
- Effort: 4-6 days, ~700 LOC.
- Sharpe lift: **+0.5 to +1.0** in paper mode (conservatively); more if you accept aggressive fill assumptions.
- Failure modes: paper-fill assumption is *the* fragile thing. Be explicit: report Sharpe at three persistence levels (0.5s, 2s, 5s) and at three haircut levels (0, 20bps, 50bps). The user will see how Sharpe degrades with realism.

---

## 5. Execution Quality (Paper-Side)

**One-liner:** Add OFI-conditional entry timing, multi-level OFI features (Xu, Cont, Stavrinou), and Almgren-Chriss-style child-order scheduling for paper-mode TWAP slicing.

### Theoretical rationale
- **Cont, Kukanov & Stoikov (2014)** order placement model — already shipped.
- **Multi-level OFI** (Xu, Cont & Stavrinou, "Multi-Level Order Flow Imbalance in a Limit Order Book") extends top-of-book OFI to 10+ depths and improves short-horizon prediction. Add multi-level OFI as features 423–433.
- **VPIN (Easley, López de Prado & O'Hara 2012)** — flow toxicity. Polyagent should *not* take liquidity when VPIN is in the top quartile (informed-trader-driven move likely against you). Ye et al. (2025, ScienceDirect S0275531925004192) confirmed VPIN predicts price jumps in crypto with positive serial correlation; the same mechanism applies to politically-driven Polymarket prices.
- **Avellaneda-Stoikov for binary contracts:** the reservation price r = S − qγσ²(T−t) collapses for binary contracts because q is bounded and T is the resolution date — meaning *as resolution approaches, the inventory penalty grows quadratically*. Polyagent should size *down* for trades with short time-to-resolution unless edge is large; this is the opposite of what most directional bots do.

### Why it raises Sharpe
Reduces adverse-selection slippage (which was 22% of notional in worst session). VPIN-conditional skip is a near-pure variance cut.

### Implementation sketch

```python
# polyagent/features/microstructure.py (NEW)
def multi_level_ofi(book, depth=10):
    bid_levels = book.bids[:depth]
    ask_levels = book.asks[:depth]
    # Cont-style: ΔBid - ΔAsk weighted by 1/(level+1)
    return sum((b.size_change - a.size_change) / (i + 1)
               for i, (b, a) in enumerate(zip(bid_levels, ask_levels)))

def vpin(trade_buckets, n=50):
    # Bulk Volume Classification per Easley-López-O'Hara
    return np.mean([abs(b.buy_vol - b.sell_vol) / b.total_vol
                    for b in trade_buckets[-n:]])

# polyagent/risk/microstructure_gate.py (NEW)
class VPINGate:
    def admit(self, token_id):
        v = vpin(recent_buckets[token_id])
        return v < 0.7
```

### Effort, lift, failure modes
- Effort: 3 days, ~400 LOC.
- Sharpe lift: **+0.2 to +0.4** (driven mostly by avoidance of toxic flow).
- Failure modes: VPIN bucket size needs calibration per category; election markets vs sports vs crypto have very different toxic-flow profiles.

---

## 6. Ensemble Diversification — TabPFN-2.5 as the 5th Expert

**One-liner:** Add TabPFN-2.5 (Grinsztajn et al. 2025, arXiv 2511.08667) as a 5th expert in the log-pool combiner, with decorrelation-loss training of an attention-based stacking layer.

### Theoretical rationale
- Deep ensembles (Lakshminarayanan et al. 2017; Fort, Hu, Lakshminarayanan 2019, "Deep Ensembles: A Loss Landscape Perspective," arXiv 1912.02757) gain almost all their benefit from *diverse mode coverage*, not from many similar models. Polyagent's current four-expert log-pool likely covers a small mode-set because all four experts are trained on overlapping features.
- **TabPFN-2.5** (released Nov 2025) provides a transformer pre-trained on 130M synthetic tabular tasks; it learns a different inductive bias than LightGBM (gradient-boosted trees) and provides genuinely orthogonal signal on small-sample tabular problems. Hollmann et al. (2025, *Nature*) demonstrate TabPFN-v2 matches AutoGluon on small data; v2.5 scales to 50K rows × 2K features — comfortably within Polyagent's 10,212 resolved markets × 422 features.
- **BatchEnsemble** (Wen et al. 2020, ICLR) is *not* recommended as a substitute — recent work (arXiv 2601.16936) finds BatchEnsemble's rank-1 perturbations explore a measure-zero subset of the deep-ensemble parameter space and underperform on calibration.

### Why it raises Sharpe
Combining a tree-based (LightGBM), a transformer (TabPFN-2.5), an LLM-RAG (Phi-4 forecaster), and a market-prior expert in the log-pool — each pulling from a *different* inductive bias — reduces ensemble variance by ~15-30% per the Breiman bias-variance decomposition, with at-worst flat mean error. The Phi-4 expert has been shown by AIA Forecaster (arXiv 2511.07678, Nov 2025) to add information *additive* to market consensus — corroborating the log-pool design.

### Implementation sketch

```python
# polyagent/models/tabpfn_expert.py (NEW)
from tabpfn import TabPFNClassifier   # tabpfn==2.5

class TabPFN25Expert:
    def __init__(self, max_context=8000):
        self.model = TabPFNClassifier(device='cuda', n_estimators=8)
        self.max_context = max_context

    def fit(self, X, y, day_groups):
        # day-grouped CPCV holdout matches existing CV scheme
        ...
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
```

Wire into `polyagent/calibration/log_pool.py`:
```python
experts = [lgbm_expert, phi4_expert, news_expert, market_expert, tabpfn25_expert]
```
RTX 5070 Ti (16GB, sm_120) handles TabPFN-2.5 at 50K-row context comfortably.

### Effort, lift, failure modes
- Effort: 2 days, ~200 LOC.
- Sharpe lift: **+0.15 to +0.35**.
- Failure modes: TabPFN-2.5 pretraining was on synthetic tabular tasks; question-feature distribution shift may be substantial. Use Real-TabPFN-2.5 variant fine-tuned on real data when accuracy matters more than speed.

---

## 7. Novel Features That Actually Add Sharpe

**One-liner:** Add four feature classes — smart-money replication, multi-level OFI, copula-based event-cluster co-movement, and Twitter/Bluesky velocity — using only features validated in 2024-2026 prediction-market literature.

### A. Smart-money replication signal
Della Vedova (SSRN 6191618) flagged 6,291 of 483,002 wallets (1.30%) as having predictive accuracy too high to be luck (p<0.01). Akey et al. (2026) confirmed top-1% wallets capture 76.5% of profit. **Build a feature: `smart_money_pressure_t = Σ(signed_volume_smart_wallets in last K mins) / total_volume`.** This is essentially the wallet-level orthogonality test from Della Vedova converted into a real-time feature. The smart-money registry (Goldsky subgraph, already shipped) is the data source.

Caveat from Polymarket's "Copytrade Wars" (Oracle, Nov 2025): top traders use secondary wallets to hide. Counter: cluster wallets by behavioral signature (markets touched, sizing pattern), not by visible label.

### B. Multi-level OFI features
Cont, Stavrinou, Xu — see §5.

### C. Copula-based event-cluster co-movement
Polymarket markets within an event share a latent common factor. Use a Gaussian-copula factor model: residualize each market's price-change against the event-mean price-change, and use the residual as a "true private edge" feature.

### D. Social velocity
Twitter/Bluesky volume *velocity* (not level) on the question text. Recent work (Reichenbach & Walther 2025, SSRN 5910522, "Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket") confirms social-velocity signals lead price changes by minutes for politics markets. Use FinBERT or general-purpose sentiment plus a Hawkes-process velocity estimator.

### Why it raises Sharpe
- Smart-money signal is one of the few sources of *informational* edge that is empirically documented in prediction markets.
- OFI+copula features cut variance via better timing.
- Social velocity is an early-warning input that gives the bot 30-300s lead time on news pipeline.

### Effort & lift
3 days total, ~300 LOC. Lift: **+0.2 to +0.4** combined.

### Failure modes
Smart-money replication has crowding risk (PolyClawster, KreoPoly, Polycop are already crowding it). Build an *anti-replication filter*: skip if the smart wallet's market is already saturated by copy-traders (depth-at-best > 5× normal).

---

## 8. Calibration & UQ Beyond Conformal

**One-liner:** Add online conformal (Gibbs-Candès 2024) on top of the existing three-tier static calibrator, with adaptive coverage parameter.

### Theoretical rationale
Static conformal (which Polyagent ships) has guaranteed marginal coverage *if data are exchangeable*. Polymarket data are obviously not exchangeable: regimes shift, new event types appear. Gibbs & Candès (2024, JMLR 25:162, "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," arXiv 2208.08401) developed Adaptive Conformal Inference (ACI) where the quantile β_t is updated by online gradient descent: β_{t+1} = β_t + γ(α − err_t). Their financial application explicitly demonstrated this on stock-market volatility prediction.

Combined with deep-ensemble uncertainty (Lakshminarayanan et al. 2017) for the LightGBM expert (via random initialization × 5), Polyagent gets epistemic + aleatoric decomposition — sizing should be more aggressive when uncertainty is *aleatoric* (irreducible) and more conservative when epistemic (reducible by more data).

### Why it raises Sharpe
Static conformal under-covers in regime shift → conformal-Kelly over-bets on poorly-calibrated probabilities → variance goes up. Online conformal corrects this within ~50-200 trades of a regime change.

### Implementation sketch
```python
# polyagent/calibration/online_conformal.py (NEW)
class GibbsCandesACI:
    def __init__(self, alpha=0.10, gamma=0.005):
        self.alpha = alpha; self.gamma = gamma
        self.beta = alpha
        self.score_history = collections.deque(maxlen=2000)

    def predict_set(self, p_hat, score_fn):
        scores = [score_fn(p, y) for (p, y) in self.score_history]
        q = np.quantile(scores, 1 - self.beta) if scores else 0.5
        return p_hat - q, p_hat + q

    def update(self, p_hat, y_true, score_fn):
        # was the realized y in the predicted set?
        lo, hi = self.predict_set(p_hat, score_fn)
        in_set = (lo <= y_true <= hi)
        err = 0.0 if in_set else 1.0
        self.beta = max(0.001, min(0.999, self.beta + self.gamma * (self.alpha - err)))
        self.score_history.append((p_hat, y_true))
```

### Effort & lift
2 days, ~200 LOC. Lift: **+0.1 to +0.3**.

---

## 9. Regime Detection — BOCPD with Auto-Deleverage

**One-liner:** Bayesian Online Changepoint Detection (Adams & MacKay 2007; Tsaknaki, Lillo & Mazzarisi 2024 autoregressive variant) on rolling Brier-loss series; on detected changepoint, halve max position size for K trades; require recovery signal to restore.

### Theoretical rationale
- BOCPD (Adams & MacKay 2007, arXiv 0710.3742) computes the posterior over run-length given streaming data. It has been used for financial order flow (Tsaknaki, Lillo & Mazzarisi 2023, arXiv 2307.02375; 2024 extension arXiv 2407.16376).
- PSI (Population Stability Index) is the workhorse drift metric in finance; threshold 0.1 = moderate drift, 0.2 = severe (Yurdakul 2018; Lin 2017 SAS Conference).
- Apply BOCPD to *Brier loss* itself, not raw P&L. Brier-loss regime change is the most direct measure of "edge degradation."

### Why it raises Sharpe
The −5% session was a regime-change event undetected in real time. BOCPD with a 30-trade window would have flagged it within ~12 trades, halving size and saving ~half the drawdown. Net: variance cut at minimal mean cost.

### Implementation sketch
```python
# polyagent/risk/bocpd_gate.py (NEW)
class BOCPDGate:
    def __init__(self, hazard=1/200, alpha0=2, beta0=2):
        self.posterior = np.array([1.0])   # P(run_length = 0)
        self.alpha = np.array([alpha0]); self.beta = np.array([beta0])

    def update(self, brier_loss):
        # Beta-conjugate UPM for bounded loss
        pred_prob = self.posterior * beta_pdf(brier_loss, self.alpha, self.beta)
        cp_prob = (pred_prob * (1 - self.hazard))
        new_run = pred_prob * self.hazard
        self.posterior = np.concatenate([[new_run.sum()], cp_prob])
        self.posterior /= self.posterior.sum()
        # update sufficient stats...
        return self.posterior[0]   # P(changepoint at t)

# polyagent/risk/throttle.py: integrate
if bocpd.posterior[0] > 0.7:
    self.global_size_multiplier = 0.5
    self.cooldown_trades = 30
```

### Effort & lift
3 days, ~350 LOC. Lift: **+0.2 to +0.5** (variance cut).

---

## 10. Stop-Loss Redesign

**One-liner:** Replace fixed-percentage stops with event-driven exits; only retain price-based stops where Kaminski-Lo's φ ≥ SR_daily condition is met.

### Theoretical rationale
Kaminski & Lo (2014, "When Do Stop-Loss Rules Stop Losses?" *Journal of Financial Markets*) prove that stop-losses raise Sharpe **iff the AR(1) coefficient φ of the strategy's returns ≥ the daily-frequency Sharpe**. Otherwise they bleed mean without proportionally cutting variance. Polyagent's daily Sharpe is approximately 0 (by hypothesis); its return autocorrelation is unmeasured but plausibly negative (mean-reverting prediction-market returns post-news).

For binary contracts specifically, the *early resolution of uncertainty* principle says that exiting a position before resolution at a price that has converged on the eventual outcome is bad: you've taken on the variance without claiming the expected return. **Use event-driven exits: close on news arrival that resolves the question, not on price moves.**

### Why it raises Sharpe
Removes a documented mean-bleed. The current stop-loss + re-entry blacklist creates a "sell low, blocked from re-entering" pattern that empirically hurts Polyagent's sessions.

### Implementation sketch
```python
# polyagent/strategies/stop_loss.py (REWRITE)
class EventDrivenExit:
    def should_exit(self, position, market_state, news_state):
        # 1. Hard exit: news has resolved the question
        if news_state.resolution_signal_strength(position.market_id) > 0.85:
            return True, "news_resolution"
        # 2. Soft exit: time-to-resolution < threshold AND price > 0.92 in our favor
        if market_state.time_to_resolution_h < 4 and position.unrealized_pct > 0.85:
            return True, "near_resolution_locked_in"
        # 3. Kaminski-Lo gate: only price-stop if φ ≥ SR
        if self.measured_phi >= self.measured_sr_daily:
            if position.unrealized_pct < -self.stop_pct:
                return True, "kaminski_lo_stop"
        return False, None
```

### Effort & lift
1 day, ~100 LOC. Lift: **+0.05 to +0.25** (mostly removes mean drag).

---

## 11. NegRisk Structure Exploitation (Beyond §4)

**One-liner:** Build a real-time NegRisk consistency dashboard that tracks `Σpᵢ vs 1` per group, detects systematic mispricing patterns from retail concentration on favorites, and books cross-group hedges when `corr(group_A_drift, group_B_drift) > 0.5`.

### Theoretical rationale
Karkare et al. (IMDEA, arXiv 2508.03474) document that NegRisk rebalancing arbitrage was 73% of all extracted profit ($29M of $39.59M), generated by 8.6% of opportunities — a **29× capital-efficiency advantage** over single-condition arbitrage. The mechanism: retail concentrates on 1-2 favorites within a NegRisk group, leaving the tail conditions (probabilities 1-5%) systematically underpriced; arbitrageurs buy the entire group's tail-NO tokens.

Polymarket's NegRisk Adapter design (audited by ChainSecurity) means a position of `k` NO tokens converts atomically to (1 YES on the kth + k−1 USDC). This is a paper-mode-friendly mechanism because the conversion is deterministic.

### Why it raises Sharpe
Same as §4 but specifically for NegRisk-augmented markets where new outcomes can appear post-launch. The augmented-NegRisk class typically runs *wider* spreads at launch, creating larger arbitrage windows.

### Implementation
Extend `polyagent/strategies/combinatorial_arb.py` with augmented-NegRisk handling:
```python
if market.negRisk and market.enableNegRisk:   # augmented variant
    # treat 'Other' placeholder as a residual leg with implied prob 1 - Σ(named)
    legs_extended = legs + [SyntheticLeg('OTHER', 1 - sum(p for p in legs))]
```

### Effort & lift
2 days, ~250 LOC. Lift: included in §4 estimate; do not double-count.

---

## 12. Sharpe Estimation Honesty — DSR / PSR / MTRL Live Harness

**One-liner:** Compute Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, and Minimum Track Record Length on every nightly evaluation; refuse to deploy any new strategy variant whose DSR p-value > 0.10.

### Theoretical rationale
- **PSR** (Bailey & López de Prado 2012): probability that true SR ≥ benchmark SR*, accounting for skewness (γ₃), excess kurtosis (γ₄), and sample size. Formula:
  PSR(SR*) = Φ((SR̂ − SR*)·√(n−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²))
- **DSR** (Bailey & López de Prado 2014, *J. Portfolio Management* 40(5), SSRN 2460551): subtracts the expected maximum-SR-from-N-trials inflation. With Polyagent's likely tens-to-hundreds of strategy variants tested, DSR is the only honest metric.
- **MTRL** (Minimum Track Record Length): the n at which PSR = 0.95. For a true SR of 0.5 with γ₃ = −0.5, γ₄ = 6, MTRL ≈ 200 trades. Polyagent's 120 fills are well below MTRL.

### Why this matters for Sharpe (indirectly)
You cannot improve what you cannot measure honestly. Without DSR, every iteration of feature engineering will *appear* to help by luck. The DSR harness sets a hard discipline: features must clear DSR > 0.5 (i.e., the deflated SR is positive) on hold-out data.

### Implementation
```python
# polyagent/eval/sharpe_honesty.py (NEW)
def deflated_sharpe(returns, sr_trials, n_independent_trials):
    sr_obs = np.mean(returns) / np.std(returns)
    g3 = scipy.stats.skew(returns)
    g4 = scipy.stats.kurtosis(returns, fisher=False)
    T = len(returns)
    sr_trials = np.array(sr_trials)
    # expected max SR after N trials (Bailey 2014 Eq. 6)
    emc = 0.5772156649
    expected_max_sr = (np.std(sr_trials) *
        ((1 - emc) * scipy.stats.norm.ppf(1 - 1/n_independent_trials)
         + emc * scipy.stats.norm.ppf(1 - 1/(n_independent_trials * np.e))))
    # PSR with deflated benchmark
    num = (sr_obs - expected_max_sr) * np.sqrt(T - 1)
    denom = np.sqrt(1 - g3 * sr_obs + ((g4 - 1) / 4) * sr_obs ** 2)
    return scipy.stats.norm.cdf(num / denom)
```

Use López de Prado's ONC (Optimal Number of Clusters) to estimate effective `n_independent_trials` from variant correlations.

### Effort & lift
2 days, ~250 LOC. Lift: not Sharpe-additive but prevents anti-Sharpe overfit. **This is the single most valuable methodology change in the document.**

---

## 13. Summary of 2024-2026 Prediction-Market Literature Used

- **Whelan (2024, *Economica* 91:188-209)** — favorite-longshot bias from risk-aversion in fixed-odds markets.
- **Bürgi, Deng & Whelan (CESifo WP 12122, 2026)** — Kalshi makers earn ~−10% returns; takers ~−32%; both subject to favorite-longshot bias.
- **Yang (Augusta, March 2026)** — top-5% skilled traders earn $121/market making, $63 taking; trader skill is the gating variable.
- **Karkare et al. (IMDEA, arXiv 2508.03474, 2025)** — $39.59M arbitrage, $29M from NegRisk; 200ms arbitrage windows.
- **Della Vedova (SSRN 6191618, Feb 2026)** — bots +$136M, retail −$82M; profits via execution; 1.3% of wallets show predictive accuracy too high to be luck.
- **Akey, Grégoire, Harvie, Martineau (SSRN 6443103, March 2026)** — top 1% capture 76.5% of profit; 9.3 ppt loss-prob reduction per 1-σ maker share.
- **Reichenbach & Walther (SSRN 5910522, Dec 2025)** — Polymarket prices broadly track realized; small "Yes-bias" but no general longshot bias.
- **Mitts & Ofir (March 2026, Harvard Corp Gov)** — $143M of anomalous profits identified as informed trading; 6,291 wallets flagged; 69.9% win rate on flagged trades.
- **AIA Forecaster (Bridgewater, arXiv 2511.07678, Nov 2025)** — LLM ensemble matches superforecasters; underperforms market consensus alone but the ensemble of LLM + market consensus beats consensus → orthogonal signal exists.
- **Turtel et al. (arXiv 2502.05253, 2025)** — DPO self-play on Phi-4 → +7-10% Brier on Polymarket questions.

---

## ⚠️ WARNINGS — Things That Cannot Be Improved In Paper Mode

This section is brutally honest because the user is sophisticated and deserves clarity on the structural ceiling.

1. **Maker LP rewards are unattainable.** Polymarket pays ~$5M/month in maker rebates for sports/esports markets and 20-25% of taker fees redistributed to makers. **Paper trading cannot collect these.** Your "passive_poster" simulator can model fills but cannot accrue real USDC rebates. Yang (2026) and Akey et al. (2026) document this is the single largest source of real edge — *and you cannot capture it in paper*.

2. **On-chain redemption arbitrage requires real wallet operations.** Buying NO tokens, calling `convert()` on the NegRiskAdapter, redeeming USDC — all require EIP-712 signing. Paper mode can simulate the *math* but cannot capture the on-chain timing risk or gas costs (which the IMDEA paper documents as <$0.02 per tx, but is real).

3. **Real-money latency arbitrage is unreachable.** IMDEA documents 200ms windows for same-market arbitrage. Standard Python signing is ~1s/sig. A paper simulator can assume any latency; the resulting Sharpe is a function of an unverifiable assumption. Be explicit about this.

4. **Partial-fill dynamics are unmodeled.** Real CLOB orders can fill 30%, leaving you 70% exposed at a worse price. Paper VWAP fills typically assume binary all-or-nothing. The Cont-Kukanov-Stoikov queue model you've shipped helps but can't fully model the joint distribution of partial-fill probabilities across N legs of a NegRisk arb.

5. **Queue position effects in real CLOB are fundamentally unobservable from paper.** When you place a limit order at the best bid, your real position in queue is a hidden variable; paper sims cannot reproduce queue-jumping dynamics that real makers exploit (sub-tick price-time priority manipulations).

6. **Insider/informed-trading edge is unattainable for a bot.** Mitts & Ofir (2026) document $143M of informed-trading profits attributable to leaked classified intelligence (Iran strike, Venezuela operation, Biden pardons). Polyagent has no information channel to such signals; this is structurally inaccessible.

7. **Wash-trading data contamination distorts your training set.** Columbia (Nov 2025) found 25% of Polymarket volume is wash trades, peaking at 60% in Dec 2024 (Della Vedova; Della Vedova 2026). Your wash-trade filter (Dubach 2026) helps, but residual contamination biases ML targets. There is no clean ground truth for "what would the price have been without wash trading."

8. **Brier-gap floor.** Bot Brier 0.13-0.18 vs market 0.05-0.10 is a 2-3× gap. **No amount of model engineering closes this gap structurally**, because the market price *includes* informed trading, smart money, and aggregated retail knowledge. The bot is one of those traders, not a privileged observer. Recognize this and stop trying to close the gap directly; instead route around it via §1 (selectivity), §4 (arb), §6 (orthogonal signal).

9. **The CFTC fee transition (March-April 2026) makes historical data partly obsolete.** As of March 6, 2026, Polymarket charges taker fees on most categories (peak 1.80% for crypto, 0.75% for sports). Pre-fee training data has different microstructure than post-fee live data. Re-validate every conclusion on post-fee data only.

10. **Polymarket V2 cutover (April 28, 2026)** wiped open orders and replaced collateral token (USDC.e → pUSD). The CLOB API endpoint changed. Any cached order-book snapshots before that date have a structural break.

---

## Prioritized Roadmap

**Week 1 (highest ROI, low risk)**
- Day 1-2: Ship §12 (DSR/PSR harness). Without this, you cannot measure whether subsequent changes help.
- Day 3-4: Ship §10 (event-driven stops, remove Kaminski-Lo-violating stops).
- Day 5: Ship §3 (ERC sizing on event clusters).

**Week 2**
- Day 1-3: Ship §1 (selective abstention).
- Day 4-5: Ship §2 (vol-target stack).

**Week 3**
- Day 1-3: Ship §9 (BOCPD regime gate).
- Day 4-5: Ship §8 (online conformal).

**Week 4-5**
- Ship §4 (NegRisk arb scanner v2). Largest single Sharpe lift but most subtle paper-fill assumptions.
- Ship §6 (TabPFN-2.5 expert).

**Week 6-7**
- Ship §5 (microstructure features), §7 (smart-money + copula features).

**Continuous**
- Run DSR-controlled experiments only; refuse to ship anything with deflated p-value > 0.10.

---

## References

- Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.* arXiv:0710.3742.
- Akey, P., Grégoire, V., Harvie, N., & Martineau, C. (2026). *Who Wins and Who Loses In Prediction Markets? Evidence from Polymarket.* SSRN 6443103.
- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Bai, T., & Jin, Y. (2026). *Conformal Selective Prediction with General Risk Control.* arXiv:2603.24704.
- Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk. SSRN 1821643.
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management 40(5), 94-107. SSRN 2460551.
- Bürgi, C., Deng, W., & Whelan, K. (2026). *Makers and Takers: The Economics of the Kalshi Prediction Market.* CESifo WP 12122. SSRN 5502658.
- Busseti, E., Ryu, E. K., & Boyd, S. (2016). *Risk-Constrained Kelly Gambling.* arXiv:1603.06183.
- Cederburg, S., O'Doherty, M. S., Wang, F., & Yan, X. S. (2020). *On the performance of volatility-managed portfolios.* Journal of Financial Economics.
- Chalkidis, N., & Savani, R. (2021). *Trading via Selective Classification.* arXiv:2110.14914.
- Cont, R., & Kukanov, A. (2017). *Optimal order placement in limit order markets.* Quantitative Finance. arXiv:1210.1625.
- Della Vedova, J. (2026). *Who Profits from Prediction Markets? Execution, not Information.* SSRN 6191618.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). *Flow toxicity and liquidity in a high-frequency world.* RFS 25(5):1457-1493.
- El-Yaniv, R., & Wiener, Y. (2010). *On the foundations of noise-free selective classification.* JMLR.
- Fort, S., Hu, H., & Lakshminarayanan, B. (2019). *Deep Ensembles: A Loss Landscape Perspective.* arXiv:1912.02757.
- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NIPS.
- Gibbs, I., & Candès, E. J. (2024). *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts.* JMLR 25:162. arXiv:2208.08401.
- Grinsztajn, L., et al. (2025). *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.* arXiv:2511.08667.
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., & Van Hemert, O. (2018). *The Impact of Volatility Targeting.* Journal of Portfolio Management.
- Hollmann, N., et al. (2025). *Accurate Predictions on Small Data with a Tabular Foundation Model.* Nature.
- Jones, E., Sagawa, S., Koh, P. W., Kumar, A., & Liang, P. (2021). *Selective Classification Can Magnify Disparities Across Groups.* ICLR.
- Kaminski, K. M., & Lo, A. W. (2014). *When Do Stop-Loss Rules Stop Losses?* Journal of Financial Markets.
- Karkare et al. (IMDEA Networks Institute) (2025). *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets.* arXiv:2508.03474.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NIPS.
- Li, J. Y.-M. (2023). *Wasserstein-Kelly Portfolios.* arXiv:2302.13979.
- Mitts, J., & Ofir, M. (2026). *From Iran to Taylor Swift: Informed Trading in Prediction Markets.* Columbia Law / U. Haifa.
- Moreira, A., & Muir, T. (2017). *Volatility-Managed Portfolios.* Journal of Finance. SSRN 2659431.
- Paolella, M. S., et al. (2025). *Risk parity portfolio optimization under heavy-tailed returns and dynamic correlations.* JTSA.
- Reichenbach, F., & Walther, M. (2025). *Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket.* SSRN 5910522.
- Sun, Q., & Boyd, S. (2018). *Distributional Robust Kelly Gambling.* arXiv:1812.10371.
- Tsaknaki, I.-Y., Lillo, F., & Mazzarisi, P. (2024). *Bayesian Autoregressive Online Change-Point Detection with Time-Varying Parameters.* arXiv:2407.16376.
- Turtel, B., Franklin, D., & Schoenegger, P. (2025). *LLMs Can Teach Themselves to Better Predict the Future.* arXiv:2502.05253.
- Vovk, V. (2025). *Conformal e-prediction.* arXiv:2001.05989 (rev. May 2025).
- Wen, Y., Tran, D., & Ba, J. (2020). *BatchEnsemble: An Alternative Approach to Efficient Ensemble.* ICLR.
- Whelan, K. (2024). *Risk aversion and favourite–longshot bias in a competitive fixed-odds betting market.* Economica 91(361):188-209.
- Xu, K., Cont, R., & Stavrinou, P. (2023). *Multi-Level Order-Flow Imbalance in a Limit Order Book.*
- Xu, X. (2024). *Improving Volatility-Managed Portfolios in Real Time.*
- Yang, H.-C. (Alex) (2026). *Skilled Liquidity Provision in Prediction Markets: Evidence from 150 Million Trades.* Augusta University working paper.

---

*End of design document. v1 draft. All Sharpe-lift figures are estimates derived from analogous literature, not promises. Ship §12 first; everything else is conditional on having an honest measurement framework in place.*