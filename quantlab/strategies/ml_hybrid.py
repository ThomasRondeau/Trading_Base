from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.composite import DirectionFeatures
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio
from quantlab.models.sklearn_model import SklearnModel
from quantlab.strategies.base import Strategy
from quantlab.targets import ForwardTarget

logger = logging.getLogger(__name__)


class MLHybridStrategy(Strategy):
    """Mean reversion entries (Bollinger + RSI + volume) confirmed by an XGBoost 5-bar direction model."""

    def __init__(self, max_depth: int = 4, symbol: str = "AAPL", **kwargs: str) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.training_data_shift = 30
        self.bollinger, self.rsi, self.volume = BollingerBands(20, 2), RSI(14), VolumeRatio(20)
        self.direction_features = DirectionFeatures(
            ATR(14), SMA(20), self.bollinger, self.rsi, MACD(12, 26, 9), self.volume
        )
        self.direction_model = SklearnModel(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            self.direction_features.outputs,
            ForwardTarget(5, "direction"),
            accuracy_score,
        )
        self._dataset: tuple[np.ndarray, np.ndarray] | None = None

    def features(self) -> Sequence[Feature]:
        return (self.direction_features,)

    def train(self, history: pd.DataFrame) -> None:
        features = self.feed.pipeline.compute(history)
        self._dataset = self.direction_model.dataset(history, features, drop_last=self.training_data_shift)
        self.direction_model.fit(*self._dataset)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        price = bar["close"]
        rsi, volume_ratio = row[self.rsi.name], row[self.volume.name]
        upper, lower = row[self.bollinger.upper], row[self.bollinger.lower]

        tech_buy = price <= lower and rsi < 35 and volume_ratio > 1.1
        tech_sell = price >= upper and rsi > 65 and volume_ratio > 1.1
        if not (tech_buy or tech_sell):
            return []

        up = self.direction_model.predict_row(row) == 1
        self.diagnostics = {"direction_up": int(up)}
        if tech_buy and up:
            return [self.order(Side.BUY, 10)]
        if tech_sell and not up:
            return [self.order(Side.SELL, 10)]
        return []
