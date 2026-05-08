"""Statistical-prior signal layer.

For each tracked market, periodically compute the LightGBM-predicted P(YES)
from question-only features, compare it to the live YES mid, and persist a
stat_signal row whenever the gap exceeds an edge threshold. Log-only for
now — calibration on out-of-sample resolved markets is needed before we
trust this enough to size a position from it.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from polyagent.config import settings
from polyagent.gamma import Market
from polyagent.models.lgbm import Predictor
from polyagent.news_store import NewsStore
from polyagent.orderbook import BookStore

log = structlog.get_logger()


@dataclass
class StatSignaler:
    book_store: BookStore
    markets: list[Market]
    news_store: NewsStore
    model_path: str
    poll_sec: float = 60.0
    min_edge: float = 0.10  # log only when |p_model - p_market| >= this

    _predictor: Predictor | None = None

    async def run(self) -> None:
        if not Path(self.model_path).exists():
            log.warning("stat_signal_no_model", path=self.model_path)
            await asyncio.Event().wait()
            return
        self._predictor = Predictor(model_path=self.model_path)
        self._predictor.load()
        is_calibrated = self._predictor._calibrator is not None
        log.info(
            "stat_signal_start",
            model_path=self.model_path,
            n_markets=len(self.markets),
            calibrated=is_calibrated,
        )

        while True:
            await asyncio.sleep(self.poll_sec)
            # Pre-filter and batch-predict off the event loop.
            quoted: list[tuple[object, float]] = []
            for m in self.markets:
                book = self.book_store.books.get(m.yes_token_id)
                if book is None:
                    continue
                mid = book.mid()
                if mid is None:
                    continue
                quoted.append((m, mid))
            if not quoted:
                continue
            from polyagent.gamma import days_to_resolution as _ttr
            features = [
                (m.question, m.liquidity, m.volume_24h, _ttr(m.end_date_iso))
                for (m, _) in quoted
            ]
            preds = await asyncio.to_thread(self._predictor.predict_batch, features)
            for (m, mid), pred in zip(quoted, preds):
                p_raw = pred["raw"]
                p_cal = pred["calibrated"]
                # Edge uses calibrated prob; that's what we'd actually trade on.
                edge = p_cal - mid
                if abs(edge) < self.min_edge:
                    continue
                direction = "yes" if edge > 0 else "no"
                detail = {
                    "p_model_raw": round(p_raw, 4),
                    "p_model_calibrated": round(p_cal, 4),
                    "p_market": round(mid, 4),
                    "edge": round(edge, 4),
                    "question": m.question[:160],
                    "category": m.category,
                    "source": "stat_lgbm",
                }
                await self.news_store.insert_signal(
                    strategy="stat_lgbm",
                    condition_id=m.condition_id,
                    direction=direction,
                    score=abs(edge),
                    news_hash="",
                    detail=detail,
                )
                log.info(
                    "stat_signal",
                    condition_id=m.condition_id,
                    question=m.question[:80],
                    p_raw=round(p_raw, 3),
                    p_cal=round(p_cal, 3),
                    p_mkt=round(mid, 3),
                    edge=round(edge, 3),
                    direction=direction,
                )
