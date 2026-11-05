from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.composite import AmplitudeFeatures, DirectionFeatures, OpportunityFeatures
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, VolumeRatio
from quantlab.models.sklearn_model import SklearnModel
from quantlab.strategies.base import Strategy
from quantlab.targets import ForwardTarget

logger = logging.getLogger(__name__)

HORIZON = 5  # bars ahead predicted by the three models


class CascadeStrategy(Strategy):
    """
    Three models in cascade: is there an opportunity? -> which direction? -> how large?
    A trade is taken only when the three agree and the predicted amplitude is large enough.
    """

    def __init__(self, min_amplitude_threshold: float = 0.05, symbol: str = "AAPL", **kwargs: str) -> None:
        """
        :param min_amplitude_threshold: minimum predicted amplitude to trade, in percent (same unit as the
                                        `return_pct` and `move` targets).
        """
        super().__init__(symbol=symbol, **kwargs)
        self.min_amplitude_threshold = min_amplitude_threshold
        self.training_data_shift = 30  # look-ahead safety margin of the training sets (>= HORIZON)

        atr, rsi, macd, bollinger, volume = ATR(14), RSI(14), MACD(12, 26, 9), BollingerBands(20, 2), VolumeRatio(14)
        sma_5, sma_20 = SMA(5), SMA(20)
        self.amplitude_features = AmplitudeFeatures(atr, sma_20, bollinger, rsi, macd, volume)
        self.direction_features = DirectionFeatures(atr, sma_20, bollinger, rsi, macd, volume)
        self.opportunity_features = OpportunityFeatures(atr, sma_5, sma_20, bollinger, rsi, macd, volume)

        self.amplitude_model = SklearnModel(
            RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1),
            self.amplitude_features.outputs,
            ForwardTarget(HORIZON, "return_pct"),
            r2_score,
        )
        self.direction_model = SklearnModel(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            self.direction_features.outputs,
            ForwardTarget(HORIZON, "direction"),
            accuracy_score,
        )
        self.opportunity_model = SklearnModel(
            RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=15, random_state=42, n_jobs=-1, class_weight="balanced"
            ),
            self.opportunity_features.outputs,
            ForwardTarget(HORIZON, "move", threshold_pct=0.05),
            balanced_accuracy_score,
        )
        self._datasets: dict[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]] = {}

    @property
    def models(self) -> dict[str, SklearnModel]:
        return {
            "amplitude": self.amplitude_model,
            "direction": self.direction_model,
            "opportunity": self.opportunity_model,
        }

    def features(self) -> Sequence[Feature]:
        return (self.amplitude_features, self.direction_features, self.opportunity_features)

    def train(self, history: pd.DataFrame) -> None:
        features = self.feed.pipeline.compute(history)
        for name, model in self.models.items():
            X, y = model.dataset(history, features, drop_last=self.training_data_shift)
            model.fit(X, y)
            self._datasets[name] = (X, y)

    def evaluate(self, n_splits: int = 5) -> dict[str, dict[str, float]]:
        """In-sample vs time-ordered cross-validated scores of the three models (overfitting check)."""
        report: dict[str, dict[str, float]] = {}
        for name, model in self.models.items():
            X, y = self._datasets[name]
            cv_mean, _ = model.cross_validate(X, y, n_splits=n_splits)
            report[name] = {"in_sample": model.in_sample_score(X, y), "cv": cv_mean}
            logger.info("%-11s model | in-sample %.2f | CV %.2f", name, report[name]["in_sample"], cv_mean)
        return report

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        opportunity = int(self.opportunity_model.predict_row(row))
        self.diagnostics = {"opportunity": opportunity}
        if opportunity != 1:
            return []

        up = int(self.direction_model.predict_row(row)) == 1
        direction = "UP" if up else "DOWN"
        amplitude = float(self.amplitude_model.predict_row(row))  # percent
        self.diagnostics.update(direction_up=int(up), amplitude_pct=amplitude)
        if abs(amplitude) < self.min_amplitude_threshold:
            return []
        # The direction and the sign of the amplitude must agree
        if not ((direction == "UP" and amplitude > 0) or (direction == "DOWN" and amplitude < 0)):
            return []

        confidence = min(abs(amplitude) / 5.0, 1.0)  # capped at 5 %
        return [self.order(Side.BUY if direction == "UP" else Side.SELL, max(1, int(confidence * 10)))]
