"""
Composite feature sets, one per ML model (amplitude, direction, opportunity).

Each set receives the base indicators it is built on, so a strategy controls their parameters
(e.g. the volume window) explicitly. Output order is the model's input column order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantlab.features.base import Computed, Feature
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio


class _FeatureSet(Feature):
    prefix: str
    columns: tuple[str, ...]
    required_columns: tuple[str, ...] = ("close",)

    @property
    def name(self) -> str:
        return self.prefix

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_{c}" for c in self.columns)

    def _frame(self, values: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame({f"{self.prefix}_{c}": values[c] for c in self.columns})


class AmplitudeFeatures(_FeatureSet):
    """Magnitude of moves, volatility and activity."""

    prefix = "amplitude"
    columns = ("volatility", "dist_sma", "bb_width", "rsi_extreme", "macd_hist_abs", "momentum_5", "volume_ratio")

    def __init__(
        self,
        atr: ATR | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
    ) -> None:
        self.atr = atr or ATR(14)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        return 5  # pct_change(5) on the close

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        return self._frame(
            {
                "volatility": computed[self.atr.name] / close,
                "dist_sma": abs(close - computed[self.sma.name]) / close,
                "bb_width": (computed[self.bollinger.upper] - computed[self.bollinger.lower]) / close,
                "rsi_extreme": abs(computed[self.rsi.name] - 50),
                "macd_hist_abs": abs(computed[self.macd.histogram]) / close,
                "momentum_5": abs(close.pct_change(5)),
                "volume_ratio": computed[self.volume.name],
            }
        )


class DirectionFeatures(_FeatureSet):
    """Stationarized trend and momentum measures (ATR-normalized)."""

    prefix = "direction"
    columns = (
        "rsi",
        "macd_hist",
        "dist_sma",
        "sma_slope",
        "bb_pos",
        "vol_trend",
        "volatility",
        "mom_5",
        "mom_10",
        "mom_20",
    )

    def __init__(
        self,
        atr: ATR | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
    ) -> None:
        self.atr = atr or ATR(14)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        return 20  # pct_change(20) on the close (the SMA slope adds 3 to the SMA's own warmup)

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        atr = computed[self.atr.name]
        sma = computed[self.sma.name]
        upper, lower = computed[self.bollinger.upper], computed[self.bollinger.lower]
        return self._frame(
            {
                "rsi": computed[self.rsi.name],
                "macd_hist": computed[self.macd.histogram] / atr,
                "dist_sma": (close - sma) / atr,
                "sma_slope": sma.diff(3) / atr,
                "bb_pos": (close - lower) / (upper - lower),
                "vol_trend": computed[self.volume.name] * np.sign(close.diff()),
                "volatility": atr / close,
                "mom_5": close.pct_change(5),
                "mom_10": close.pct_change(10),
                "mom_20": close.pct_change(20),
            }
        )


class OpportunityFeatures(_FeatureSet):
    """Volatility regime, activity and distance to the bands."""

    prefix = "opportunity"
    columns = (
        "bb_width",
        "vol_ratio",
        "vol_change",
        "volume_ratio",
        "volume_trend",
        "momentum_align",
        "dist_lower",
        "dist_upper",
        "dist_sma",
        "price_accel",
    )
    required_columns = ("close", "volume")

    def __init__(
        self,
        atr: ATR | None = None,
        sma_fast: SMA | None = None,
        sma: SMA | None = None,
        bollinger: BollingerBands | None = None,
        rsi: RSI | None = None,
        macd: MACD | None = None,
        volume: VolumeRatio | None = None,
    ) -> None:
        self.atr = atr or ATR(14)
        self.sma_fast = sma_fast or SMA(5)
        self.sma = sma or SMA(20)
        self.bollinger = bollinger or BollingerBands(20, 2)
        self.rsi = rsi or RSI(14)
        self.macd = macd or MACD()
        self.volume = volume or VolumeRatio(14)

    def inputs(self) -> Sequence[Feature]:
        return (self.atr, self.sma_fast, self.sma, self.bollinger, self.rsi, self.macd, self.volume)

    @property
    def warmup(self) -> int:
        return 20  # rolling(20) on the ATR and on the raw volume

    def compute(self, data: pd.DataFrame, computed: Computed) -> pd.DataFrame:
        close = data["close"]
        atr = computed[self.atr.name]
        upper, lower = computed[self.bollinger.upper], computed[self.bollinger.lower]
        raw_volume = data["volume"]
        momentum_alignment = (
            (computed[self.macd.line] > computed[self.macd.signal]).astype(int)
            + (computed[self.rsi.name] > 50).astype(int)
            + (close > computed[self.sma_fast.name]).astype(int)
        )
        return self._frame(
            {
                "bb_width": (upper - lower) / close,
                "vol_ratio": atr / atr.rolling(window=20).mean(),
                "vol_change": atr.pct_change(3),
                "volume_ratio": computed[self.volume.name],
                "volume_trend": raw_volume.rolling(window=5).mean() / raw_volume.rolling(window=20).mean(),
                "momentum_align": momentum_alignment,
                "dist_lower": (close - lower) / atr,
                "dist_upper": (upper - close) / atr,
                "dist_sma": (close - computed[self.sma.name]) / atr,
                "price_accel": close.pct_change(3),
            }
        )
