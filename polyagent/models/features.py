"""Feature engineering for the statistical layer.

Two feature scopes:

1. Question-only features (available the moment a market is created — no
   price history needed). Used at inference time on every active market.
2. Live-market features (depend on current book state). Adds book midprice,
   spread, and a longshot-bucket flag from the current YES price.

Both feature sets are computed for training too: at training time, the
"current YES price" is taken from the last available point of the historical
prices-history series for the market — i.e. the resolution price, which for
clean resolutions is 0 or 1. To avoid leaking the label, we use the question-
only features for training the base-rate prior, and the price-history features
will feed a separate time-aware model later.

This module returns plain dicts; the LightGBM wrapper converts them to a
DataFrame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lightweight category keyword lookup. Each tuple is (category_label, regex).
_CATEGORY_RES: list[tuple[str, re.Pattern]] = [
    ("crypto", re.compile(r"\b(bitcoin|btc|ether|eth|crypto|solana|sol|altcoin|stablecoin|memecoin|fdv|launch|token|defi)\b", re.I)),
    ("politics_us", re.compile(r"\b(trump|biden|harris|vance|congress|senate|house|election|primary|impeach|powell|fed|fomc|cfpb|scotus|supreme court|federal)\b", re.I)),
    ("politics_global", re.compile(r"\b(putin|xi|jinping|netanyahu|zelensky|orban|macron|sunak|starmer|merz|modi|erdogan|kim jong)\b", re.I)),
    ("geopolitics", re.compile(r"\b(iran|israel|gaza|hamas|hezbollah|ukraine|russia|china|taiwan|north korea|nato|hormuz|red sea|cease[- ]?fire|war|invasion|sanctions)\b", re.I)),
    ("sports_us", re.compile(r"\b(nba|nfl|mlb|nhl|lakers|warriors|celtics|patriots|cowboys|yankees|dodgers|bruins|rangers|heat|knicks)\b", re.I)),
    ("sports_global", re.compile(r"\b(world cup|fifa|champions league|premier league|formula 1|f1|tennis|cricket|olympics|rugby)\b", re.I)),
    ("entertainment", re.compile(r"\b(oscar|grammy|emmy|movie|film|album|netflix|disney|spotify|taylor swift|kardashian)\b", re.I)),
    ("economy", re.compile(r"\b(cpi|inflation|gdp|unemployment|nfp|recession|fed rate|interest rate|treasury|yield|bond|bls|earnings)\b", re.I)),
    ("ai", re.compile(r"\b(openai|anthropic|claude|gpt|gemini|llama|mistral|deepseek|chatgpt|model|hugging[\s-]?face|sora)\b", re.I)),
    ("weather", re.compile(r"\b(hurricane|tornado|earthquake|wildfire|flood|temperature)\b", re.I)),
]


def _category_flags(question: str) -> dict[str, int]:
    out: dict[str, int] = {f"cat_{name}": 0 for name, _ in _CATEGORY_RES}
    for name, rx in _CATEGORY_RES:
        if rx.search(question):
            out[f"cat_{name}"] = 1
    return out


# Negative-event keywords (analogous to direction.py polarity).
_NEG_PAT = re.compile(
    r"\b(out|fired|resign|impeach|leave|lose|loses|fail|fails|default|defaults|fall|falls|drop|drops|ban|bans|war|conflict|invasion|recession|crash|collapse|bankruptcy|strike|attack)\b",
    re.I,
)
_POS_PAT = re.compile(
    r"\b(win|wins|succeed|approve|pass|passes|rise|rises|deal|agreement|ceasefire|peace|elected|confirm|confirms|above|exceed|hit|hits)\b",
    re.I,
)
_NUM_PAT = re.compile(r"\b(\d[\d,]*\.?\d*)\b")
_PCT_PAT = re.compile(r"\b\d+(\.\d+)?%")
_DATE_PAT = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|2024|2025|2026|2027|q1|q2|q3|q4)\b", re.I)
_DOLLAR_PAT = re.compile(r"\$\s?\d[\d,]*(\.\d+)?[kKmMbB]?\b")
_K_M_B_PAT = re.compile(r"\b\d+(\.\d+)?\s?[kKmMbB]\b")
_OVERUNDER_PAT = re.compile(r"\b(o/?u|over|under|spread|moneyline|moneyl)\b", re.I)
_VS_PAT = re.compile(r"\bvs\.?\b", re.I)
_COMPARE_PAT = re.compile(r"\b(more than|less than|at least|at most|exceeds?|fall(?:s)? below|hit|reach(?:es)?)\b", re.I)
_COMPOUND_PAT = re.compile(r"\b(and|or)\b", re.I)
_NEGATION_PAT = re.compile(r"\b(not|never|won't|wont|fail to)\b", re.I)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")  # rough capitalized-word heuristic

# ── Sports-specific patterns (sports_global certified at +0.128 log-loss;
# these are surgical additions to widen the existing edge) ────────────────
# Soccer/football clubs and league cues — most sports_global volume is
# soccer match win/draw markets where the question wording is highly
# templated.
_SPORTS_GLOBAL_LEAGUE_RE = re.compile(
    r"\b(world cup|fifa|champions league|premier league|la liga|bundesliga|"
    r"serie a|ligue 1|copa|europa|euro|copa america|uefa|cup final|"
    r"f(?:ormula)?\s?1|grand prix|gp|nascar|"
    r"atp|wta|roland garros|wimbledon|us open|australian open|grand slam|"
    r"icc|t20|odi|test match|cricket world|"
    r"olympics?|paralympic|"
    r"six nations|world rugby|super rugby|"
    r"ufc|mma)\b",
    re.I,
)
# Football club name suffixes (FC, United, City, Athletic, Real, etc.) —
# strong signal that a question is a soccer match-win prediction
_FOOTBALL_CLUB_RE = re.compile(
    r"\b(FC|United|City|Athletic|Real|Bayern|PSG|Madrid|Barcelona|Liverpool|"
    r"Chelsea|Arsenal|Tottenham|Juventus|Milan|Inter|Napoli|Roma|Dortmund|"
    r"Atletico|Sevilla|Porto|Benfica|Ajax|Celtic|Rangers)\b"
)
# Match-type cues
_MATCH_WIN_RE = re.compile(r"\bwin\s+(?:the|on|in|at)\b", re.I)
_TOURNAMENT_WIN_RE = re.compile(r"\bwin\s+the\s+(world cup|champions league|premier league|cup|tournament|final)\b", re.I)
# UFC/MMA fight format markers
_UFC_RE = re.compile(r"\bUFC\b|\b(?:vs?\.?|\bmiddleweight|lightweight|welterweight|featherweight|bantamweight|heavyweight)\b", re.I)
# ISO date pattern in question (lots of sports markets are dated)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def question_features(
    question: str,
    *,
    liquidity: float = 0.0,
    volume: float = 0.0,
    days_to_resolution: float | None = None,
) -> dict[str, float]:
    q = question or ""
    q_lower = q.lower()
    feats: dict[str, float] = {
        "q_len_chars": float(len(q)),
        "q_len_words": float(len(q.split())),
        "q_has_above": 1.0 if "above" in q_lower else 0.0,
        "q_has_below": 1.0 if "below" in q_lower else 0.0,
        "q_has_by": 1.0 if " by " in q_lower else 0.0,
        "q_has_question_mark": 1.0 if q.endswith("?") else 0.0,
        "q_neg_event_hits": float(len(_NEG_PAT.findall(q_lower))),
        "q_pos_event_hits": float(len(_POS_PAT.findall(q_lower))),
        "q_num_count": float(len(_NUM_PAT.findall(q))),
        "q_pct_count": float(len(_PCT_PAT.findall(q))),
        "q_date_token_count": float(len(_DATE_PAT.findall(q))),
        "liquidity": float(liquidity or 0.0),
        "volume": float(volume or 0.0),
        "log_liquidity": float(__import__("math").log1p(max(0.0, float(liquidity or 0.0)))),
        "log_volume": float(__import__("math").log1p(max(0.0, float(volume or 0.0)))),
        # New features (improvements 11 + 12 + extras)
        "q_has_dollar": 1.0 if _DOLLAR_PAT.search(q) or _K_M_B_PAT.search(q) else 0.0,
        "q_dollar_count": float(len(_DOLLAR_PAT.findall(q))),
        "q_has_overunder": 1.0 if _OVERUNDER_PAT.search(q) else 0.0,
        "q_has_vs": 1.0 if _VS_PAT.search(q) else 0.0,
        "q_starts_will": 1.0 if q_lower.startswith("will ") else 0.0,
        "q_compare_count": float(len(_COMPARE_PAT.findall(q))),
        "q_has_and_or": 1.0 if _COMPOUND_PAT.search(q) else 0.0,
        "q_has_negation": 1.0 if _NEGATION_PAT.search(q) else 0.0,
        "q_proper_nouns": float(len(_PROPER_NOUN_RE.findall(q))),
        "q_polarity_diff": float(
            len(_NEG_PAT.findall(q_lower)) - len(_POS_PAT.findall(q_lower))
        ),
        # Time-to-resolution: how far from now (positive = future). NaN-safe.
        "days_to_resolution": float(days_to_resolution) if days_to_resolution is not None else 0.0,
        "has_ttr": 1.0 if days_to_resolution is not None else 0.0,
        "log1p_ttr": float(__import__("math").log1p(max(0.0, float(days_to_resolution or 0.0)))),
        # Sports-specific (surgical additions for the certified sports_global
        # slice; these features are zero on non-sports questions, so they only
        # cost a few extra splits on irrelevant rows but give the model
        # match-type / tournament discrimination on the sports slice).
        "sports_league_hits": float(len(_SPORTS_GLOBAL_LEAGUE_RE.findall(q))),
        "sports_has_league": 1.0 if _SPORTS_GLOBAL_LEAGUE_RE.search(q) else 0.0,
        "sports_football_clubs": float(len(_FOOTBALL_CLUB_RE.findall(q))),
        "sports_has_football_club": 1.0 if _FOOTBALL_CLUB_RE.search(q) else 0.0,
        "sports_match_win": 1.0 if _MATCH_WIN_RE.search(q) else 0.0,
        "sports_tournament_win": 1.0 if _TOURNAMENT_WIN_RE.search(q) else 0.0,
        "sports_has_ufc": 1.0 if _UFC_RE.search(q) else 0.0,
        "sports_has_iso_date": 1.0 if _ISO_DATE_RE.search(q) else 0.0,
    }
    feats.update({k: float(v) for k, v in _category_flags(q).items()})
    return feats


@dataclass
class BookSnapshot:
    yes_mid: float | None = None
    no_mid: float | None = None
    yes_spread: float | None = None
    yes_best_ask: float | None = None
    yes_best_bid: float | None = None
    # Optional reference to the live YES OrderBook for micro-structure
    # feature extraction (pmwhybetter.md Problem-10 fix #4: micro-price,
    # VAMP, queue-imbalance — Gould & Bonart 2015, Stoikov 2017).
    # When provided, live_features() pulls these features in addition to
    # the legacy mid/spread/longshot triple.
    yes_book: object | None = None
    # Optional condition_id + sqlite handle so live_features() can pull
    # Polymarket native-endpoint features (comment-count delta, top-trader
    # 24h net inflow, 1h unique traders) from the polled cache table.
    condition_id: str | None = None
    native_features_conn: object | None = None


def live_features(question: str, snap: BookSnapshot, *, liquidity: float = 0.0, volume: float = 0.0) -> dict[str, float]:
    feats = question_features(question, liquidity=liquidity, volume=volume)
    if snap.yes_mid is not None:
        feats["mkt_yes_mid"] = float(snap.yes_mid)
        # Longshot bucket from current price
        feats["mkt_longshot_yes"] = 1.0 if snap.yes_mid <= 0.10 else 0.0
        feats["mkt_favorite_yes"] = 1.0 if snap.yes_mid >= 0.90 else 0.0
        feats["mkt_midrange_yes"] = 1.0 if 0.40 <= snap.yes_mid <= 0.60 else 0.0
        # Akey 2026 finding: 63% of all Polymarket trades happen at
        # extreme prices (<10c or >90c) where spread is largest as %
        # of stake. Wider "extreme band" feature captures the broader
        # longshot zone where adverse-selection from retail is highest.
        feats["mkt_in_extreme_band"] = (
            1.0 if (snap.yes_mid <= 0.15 or snap.yes_mid >= 0.85) else 0.0
        )
        # Distance from 0.50 — proxy for trade-cost-as-fraction-of-stake
        feats["mkt_distance_from_half"] = float(abs(snap.yes_mid - 0.50))
    if snap.yes_spread is not None:
        feats["mkt_spread"] = float(snap.yes_spread)

    # Microstructure features (pmwhybetter.md Problem-10 fix #4). These are
    # the cross-asset SHAP-stable feature family from arXiv 2602.00776
    # (Dec 2025) — order-book-imbalance + adverse-selection features rank
    # stably across BTC/LTC/ETC/ENJ/ROSE. We compute them best-effort and
    # leave them out of the feature dict if the book reference is absent
    # so legacy training data isn't disturbed.
    if snap.yes_book is not None:
        try:
            from polyagent.models.microprice import compute_features as _ms_features
            ms = _ms_features(snap.yes_book, vamp_notional=500.0)
            if ms.micro is not None:
                feats["mkt_micro_price"] = float(ms.micro)
                # Micro-vs-mid difference is a queue-pressure signal in bps.
                if snap.yes_mid is not None:
                    feats["mkt_micro_minus_mid_bps"] = float(
                        (ms.micro - snap.yes_mid) * 10_000.0
                    )
            if ms.queue_imbalance is not None:
                feats["mkt_queue_imbalance"] = float(ms.queue_imbalance)
            if ms.vamp_buy is not None and snap.yes_mid is not None:
                feats["mkt_vamp_buy_minus_mid_bps"] = float(
                    (ms.vamp_buy - snap.yes_mid) * 10_000.0
                )
            if ms.vamp_sell is not None and snap.yes_mid is not None:
                feats["mkt_vamp_sell_minus_mid_bps"] = float(
                    (ms.vamp_sell - snap.yes_mid) * 10_000.0
                )
            feats["mkt_bid_levels"] = float(ms.bid_levels)
            feats["mkt_ask_levels"] = float(ms.ask_levels)
        except Exception:
            # Don't let feature extraction errors crash the predictor — we
            # silently drop the microstructure features.
            pass

    # Polymarket native-endpoint features (Strategy #10 from the May-11
    # research playbook): comment-count delta as a retail-attention
    # proxy, top-trader 24h net inflow as a smart-money flow proxy,
    # 1h unique-trader count as a market-activity proxy. All sourced
    # from the polled cache in `polymarket_native_features` so the
    # signal-eval path doesn't hit the API.
    if snap.native_features_conn is not None and snap.condition_id:
        try:
            from polyagent.data.polymarket_native import lookup_features
            nf = lookup_features(snap.native_features_conn, snap.condition_id)
            if nf is not None:
                feats["mkt_comment_count_delta_6h"] = float(nf.comment_count_delta_6h)
                feats["mkt_top_trader_inflow_24h"] = float(nf.top_trader_inflow_24h)
                feats["mkt_unique_traders_1h"] = float(nf.unique_traders_1h)
                # Log-scaled versions for the LGBM
                import math as _m
                feats["mkt_log_comment_count_6h"] = float(_m.log1p(max(0, nf.comment_count_6h)))
                feats["mkt_log_unique_traders_1h"] = float(_m.log1p(max(0, nf.unique_traders_1h)))
        except Exception:
            pass
    return feats
