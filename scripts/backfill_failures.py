"""Backfill the model_failures table from signal_outcomes + resolutions.

Usage: python -m scripts.backfill_failures

Walks every resolved row in signal_outcomes, classifies the prediction
against the actual outcome, joins to fills/resolutions for the
trade-side notional+pnl, and writes one row per failure type into
`model_failures`. Idempotent: a unique index on (condition_id,
failure_type) means re-running doesn't double-count.
"""
from __future__ import annotations

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.models.failure_tracker import backfill_failures

log = logging_setup.configure()


def main() -> None:
    summary = backfill_failures(settings.db_path)
    log.info("failure_backfill_summary", **summary)


if __name__ == "__main__":
    main()
