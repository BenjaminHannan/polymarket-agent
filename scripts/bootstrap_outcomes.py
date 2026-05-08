"""Populate signal_outcomes from the existing resolutions table.

Each row gets a stat_lgbm probability computed from the question. News and
market columns are NULL until those signals accumulate live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.models.lgbm import Predictor
from polyagent.models.outcomes import bootstrap_from_resolutions

log = logging_setup.configure()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.db_path).parent / "lgbm_model.joblib"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not Path(args.model).exists():
        log.error("model_missing", path=args.model)
        raise SystemExit(2)

    predictor = Predictor(model_path=args.model)
    predictor.load()

    summary = bootstrap_from_resolutions(predictor=predictor, limit=args.limit)
    log.info("bootstrap_done", **summary)


if __name__ == "__main__":
    main()
