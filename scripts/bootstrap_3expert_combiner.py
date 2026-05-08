"""Bootstrap a 3-expert combiner from the existing 2-expert one.

Use this only as a placeholder until enough live `signal_outcomes` rows have
both `p_news_match` AND `p_market_<horizon>` populated to retrain via
`train_combiner_per_category --experts stat_lgbm news_match p_market_6h`.

Strategy: keep the 2-expert weights exactly, slot a small fixed weight for
news_match by shrinking the existing two proportionally. This gives news
*some* runtime impact while preserving the trained ratio between the
existing two experts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import structlog

from polyagent import logging_setup
from polyagent.config import settings

log = logging_setup.configure()


def _adjust(entry: dict, news_weight: float) -> dict:
    weights = list(entry["weights"])
    expert_names = list(entry["expert_names"])
    if "news_match" in expert_names:
        return entry  # already has news
    keep = 1.0 - news_weight
    weights = [w * keep for w in weights]
    weights.append(news_weight)
    expert_names.append("news_match")
    return {"weights": weights, "expert_names": expert_names}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", default=str(Path(settings.db_path).parent / "combiner.joblib"))
    p.add_argument("--out", default=None)
    p.add_argument(
        "--news-weight-default",
        type=float,
        default=0.10,
        help="Bootstrap weight for news_match in the default combiner (0.0-0.5)",
    )
    p.add_argument(
        "--news-weight-geopolitics",
        type=float,
        default=0.20,
        help="Heavier news weight on geopolitics where news matters most",
    )
    args = p.parse_args()

    in_path = args.input
    out_path = args.out or in_path
    bundle = joblib.load(in_path)
    if bundle.get("version", 1) != 2:
        log.error("expected_v2_bundle", got_version=bundle.get("version"))
        raise SystemExit(2)

    new_default = _adjust(bundle["default"], args.news_weight_default)
    new_by_cat: dict[str, dict] = {}
    for cat, entry in (bundle.get("by_category") or {}).items():
        w = (
            args.news_weight_geopolitics
            if cat in ("geopolitics", "politics_us", "politics_global")
            else args.news_weight_default
        )
        new_by_cat[cat] = _adjust(entry, w)

    bundle["default"] = new_default
    bundle["by_category"] = new_by_cat
    bundle["bootstrapped_news_match"] = True
    joblib.dump(bundle, out_path)
    log.info(
        "bootstrap_done",
        path=out_path,
        default_weights=[round(w, 3) for w in new_default["weights"]],
        default_experts=new_default["expert_names"],
        n_categories=len(new_by_cat),
    )


if __name__ == "__main__":
    main()
