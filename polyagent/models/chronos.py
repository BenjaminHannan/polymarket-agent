"""Chronos-Bolt-Small zero-shot price-series forecaster (CPU-friendly, ~50M params).

Off by default. Set ENABLE_CHRONOS=1 to load. First run downloads ~150 MB
from HuggingFace (no auth token needed for the public model).

Usage at inference:
    forecaster = ChronosForecaster()
    forecaster.load()
    yhat = forecaster.predict(prices=[0.42, 0.41, 0.43, ...], horizon=24)

`yhat` is a list of point forecasts (median of the quantile heads).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import structlog

log = structlog.get_logger()


@dataclass
class ChronosForecaster:
    model_id: str = "amazon/chronos-bolt-small"
    device: str = "cpu"
    _pipe: object | None = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from chronos import BaseChronosPipeline  # type: ignore
        except ImportError:
            log.warning("chronos_not_installed", note="pip install chronos-forecasting to enable")
            self._pipe = None
            return
        try:
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype="float32"
            )
            log.info("chronos_loaded", model=self.model_id, device=self.device)
        except Exception as e:
            log.warning("chronos_load_failed", err=str(e))
            self._pipe = None

    def predict(self, prices: list[float], horizon: int = 24) -> list[float] | None:
        if self._pipe is None:
            return None
        try:
            import torch  # type: ignore

            ctx = torch.tensor(prices, dtype=torch.float32)
            quantiles, mean = self._pipe.predict_quantiles(  # type: ignore[attr-defined]
                context=ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
            )
            median = quantiles[0, :, 1].tolist()
            return median
        except Exception as e:
            log.warning("chronos_predict_failed", err=str(e))
            return None


def is_enabled() -> bool:
    return os.getenv("ENABLE_CHRONOS", "0") == "1"
