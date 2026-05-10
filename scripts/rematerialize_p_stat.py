"""Re-run the LightGBM predictor against all signal_outcomes rows to refresh
p_stat_lgbm with the latest trained model. The default `bootstrap_outcomes`
script skips rows whose condition_id already exists in signal_outcomes; this
script bypasses that and updates in place.

Usage: python -m scripts.rematerialize_p_stat --model <path>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import structlog

from polyagent import logging_setup
from polyagent.config import settings
from polyagent.models.lgbm import Predictor

log = logging_setup.configure()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(settings.db_path).parent / "lgbm_model.joblib"))
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    if not Path(args.model).exists():
        log.error("model_missing", path=args.model)
        raise SystemExit(2)

    predictor = Predictor(model_path=args.model)
    predictor.load()

    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        """SELECT s.condition_id, r.detail
           FROM signal_outcomes s
           INNER JOIN resolutions r ON r.condition_id = s.condition_id"""
    ).fetchall()
    log.info("rematerialize_start", n=len(rows))

    work: list[tuple[str, str, float, float]] = []
    for cid, detail in rows:
        try:
            d = json.loads(detail or "{}")
        except json.JSONDecodeError:
            continue
        q = d.get("question") or ""
        if not q:
            continue
        work.append((cid, q, d.get("liquidity") or 0.0, d.get("volume") or 0.0))

    log.info("rematerialize_pending", n=len(work))
    updated = 0
    for start in range(0, len(work), args.batch_size):
        chunk = work[start : start + args.batch_size]
        feats = [(q, liq, vol, None) for _, q, liq, vol in chunk]
        try:
            preds = predictor.predict_batch(feats)
        except Exception as e:
            log.warning("predict_batch_error", err=str(e), start=start)
            continue
        for (cid, _, _, _), p in zip(chunk, preds):
            p_cal = p.get("calibrated") if isinstance(p, dict) else None
            if p_cal is None:
                continue
            conn.execute(
                "UPDATE signal_outcomes SET p_stat_lgbm = ? WHERE condition_id = ?",
                (float(p_cal), cid),
            )
            updated += 1
        if (start // args.batch_size) % 10 == 0:
            conn.commit()
            log.info("rematerialize_progress", done=start + len(chunk), of=len(work), updated=updated)

    conn.commit()
    conn.close()
    log.info("rematerialize_done", updated=updated, total=len(work))


if __name__ == "__main__":
    main()
