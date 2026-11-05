"""Composite feature sets, one per ML model (amplitude, direction, opportunity)."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from quantlab.features.base import FeatureInterface, ResolvedDependencies
from quantlab.features.feature_id import FeatureID
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, Volume

T = TypeVar("T", bound=FeatureInterface)


def _dep(resolved: ResolvedDependencies | None, feature_id: FeatureID, kind: type[T], owner: str) -> T:
    dep = (resolved or {}).get(feature_id)
    if dep is None:
        raise ValueError(f"{owner}: dependency {feature_id} is not resolved.")
    return cast(T, dep)


class AmplitudeFeatures(FeatureInterface):
    """Inputs of the amplitude model: magnitude of moves, volatility and activity."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.clear_cache()
        self.data = data if data is not None else pd.DataFrame()

    def dependencies(self) -> set[FeatureID]:
        return {
            FeatureID.ATR,
            FeatureID.SMA_20,
            FeatureID.BOLLINGER_BANDS,
            FeatureID.RSI,
            FeatureID.MACD,
            FeatureID.VOLUME,
        }

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        owner = "AmplitudeFeatures"
        atr = _dep(resolved_dependencies, FeatureID.ATR, ATR, owner)
        sma_20 = _dep(resolved_dependencies, FeatureID.SMA_20, SMA, owner)
        bollinger = _dep(resolved_dependencies, FeatureID.BOLLINGER_BANDS, BollingerBands, owner)
        rsi = _dep(resolved_dependencies, FeatureID.RSI, RSI, owner)
        macd = _dep(resolved_dependencies, FeatureID.MACD, MACD, owner)
        volume = _dep(resolved_dependencies, FeatureID.VOLUME, Volume, owner)

        close = pd.to_numeric(self.data["close"], errors="coerce")

        self.volatility = atr.atr / close
        self.dist_sma = abs(close - sma_20.sma) / close
        self.bb_width = (bollinger.upper_band - bollinger.lower_band) / close
        self.rsi_extreme = abs(rsi.rsi - 50)
        self.macd_hist_abs = abs(macd.macd_histogram) / close
        self.momentum_5 = abs(close.pct_change(5))
        self.volume_ratio = volume.volume_ratio

    def update(self, new_data: pd.DataFrame) -> None:
        self.data = new_data

    def clear_cache(self) -> None:
        self.volatility: pd.Series | None = None
        self.dist_sma: pd.Series | None = None
        self.bb_width: pd.Series | None = None
        self.rsi_extreme: pd.Series | None = None
        self.macd_hist_abs: pd.Series | None = None
        self.momentum_5: pd.Series | None = None
        self.volume_ratio: pd.Series | None = None
        self.data = pd.DataFrame()


class DirectionFeatures(FeatureInterface):
    """Inputs of the direction model: stationarized trend and momentum measures."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.clear_cache()
        self.data = data if data is not None else pd.DataFrame()

    def dependencies(self) -> set[FeatureID]:
        return {
            FeatureID.ATR,
            FeatureID.SMA_20,
            FeatureID.BOLLINGER_BANDS,
            FeatureID.RSI,
            FeatureID.MACD,
            FeatureID.VOLUME,
        }

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        owner = "DirectionFeatures"
        atr = _dep(resolved_dependencies, FeatureID.ATR, ATR, owner)
        sma_20 = _dep(resolved_dependencies, FeatureID.SMA_20, SMA, owner)
        bollinger = _dep(resolved_dependencies, FeatureID.BOLLINGER_BANDS, BollingerBands, owner)
        rsi = _dep(resolved_dependencies, FeatureID.RSI, RSI, owner)
        macd = _dep(resolved_dependencies, FeatureID.MACD, MACD, owner)
        volume = _dep(resolved_dependencies, FeatureID.VOLUME, Volume, owner)

        close = pd.to_numeric(self.data["close"], errors="coerce")

        self.rsi = rsi.rsi  # already bounded in [0, 100]
        self.macd_hist_norm = macd.macd_histogram / atr.atr  # ATR-normalized for stationarity
        self.price_dist_sma = (close - sma_20.sma) / atr.atr
        self.sma_slope_norm = sma_20.sma.diff(3) / atr.atr
        self.bb_position = (close - bollinger.lower_band) / (bollinger.upper_band - bollinger.lower_band)
        self.volume_trend = volume.volume_ratio * np.sign(close.diff())
        self.volatility = atr.atr / close
        self.mom_5 = close.pct_change(5)
        self.mom_10 = close.pct_change(10)
        self.mom_20 = close.pct_change(20)

    def update(self, new_data: pd.DataFrame) -> None:
        self.data = new_data

    def clear_cache(self) -> None:
        self.rsi: pd.Series | None = None
        self.macd_hist_norm: pd.Series | None = None
        self.price_dist_sma: pd.Series | None = None
        self.sma_slope_norm: pd.Series | None = None
        self.bb_position: pd.Series | None = None
        self.volume_trend: pd.Series | None = None
        self.volatility: pd.Series | None = None
        self.mom_5: pd.Series | None = None
        self.mom_10: pd.Series | None = None
        self.mom_20: pd.Series | None = None
        self.data = pd.DataFrame()


class OpportunityFeatures(FeatureInterface):
    """Inputs of the opportunity model: volatility regime, activity and distance to bands."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.clear_cache()
        self.data = data if data is not None else pd.DataFrame()

    def dependencies(self) -> set[FeatureID]:
        return {
            FeatureID.ATR,
            FeatureID.SMA_5,
            FeatureID.SMA_20,
            FeatureID.BOLLINGER_BANDS,
            FeatureID.RSI,
            FeatureID.MACD,
            FeatureID.VOLUME,
        }

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        owner = "OpportunityFeatures"
        atr = _dep(resolved_dependencies, FeatureID.ATR, ATR, owner)
        sma_5 = _dep(resolved_dependencies, FeatureID.SMA_5, SMA, owner)
        sma_20 = _dep(resolved_dependencies, FeatureID.SMA_20, SMA, owner)
        bollinger = _dep(resolved_dependencies, FeatureID.BOLLINGER_BANDS, BollingerBands, owner)
        rsi = _dep(resolved_dependencies, FeatureID.RSI, RSI, owner)
        macd = _dep(resolved_dependencies, FeatureID.MACD, MACD, owner)
        volume = _dep(resolved_dependencies, FeatureID.VOLUME, Volume, owner)

        close = pd.to_numeric(self.data["close"], errors="coerce")

        self.bb_width_norm = (bollinger.upper_band - bollinger.lower_band) / close
        self.volatility_ratio = atr.atr / atr.atr.rolling(window=20).mean()
        self.volatility_change = atr.atr.pct_change(3)
        self.volume_ratio = volume.volume_ratio
        self.volume_trend = volume.volume.rolling(window=5).mean() / volume.volume.rolling(window=20).mean()

        # Momentum alignment on a 0-3 scale
        self.momentum_alignment = (
            (macd.macd > macd.signal_line).astype(int) + (rsi.rsi > 50).astype(int) + (close > sma_5.sma).astype(int)
        )

        self.dist_lower_bb = (close - bollinger.lower_band) / atr.atr
        self.dist_upper_bb = (bollinger.upper_band - close) / atr.atr
        self.dist_sma = (close - sma_20.sma) / atr.atr
        self.price_accel = close.pct_change(3)

    def update(self, new_data: pd.DataFrame) -> None:
        self.data = new_data

    def clear_cache(self) -> None:
        self.bb_width_norm: pd.Series | None = None
        self.volatility_ratio: pd.Series | None = None
        self.volatility_change: pd.Series | None = None
        self.volume_ratio: pd.Series | None = None
        self.volume_trend: pd.Series | None = None
        self.momentum_alignment: pd.Series | None = None
        self.dist_lower_bb: pd.Series | None = None
        self.dist_upper_bb: pd.Series | None = None
        self.dist_sma: pd.Series | None = None
        self.price_accel: pd.Series | None = None
        self.data = pd.DataFrame()
