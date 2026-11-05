from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.composite import AmplitudeFeatures, DirectionFeatures, OpportunityFeatures
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, Volume
from quantlab.models.cascade_models import AmplitudeModel, DirectionModel, OpportunityModel
from quantlab.strategies.base import TradingStrategyInterface
from quantlab.targets.forward import (
    TargetAmplitude,
    TargetDirection,
    TargetOpportunity,
    TargetPlus1,
    TargetPlus5,
    TargetPlus10,
    TargetPlus20,
)
from quantlab.targets.manager import TargetManager
from quantlab.targets.target_id import TargetID

logger = logging.getLogger(__name__)


class CascadeStrategy(TradingStrategyInterface):
    """
    Three models in cascade: is there an opportunity? -> which direction? -> how large?
    A trade is taken only when the three agree and the predicted amplitude is large enough.
    """

    def __init__(self, min_amplitude_threshold: float = 0.05) -> None:
        """
        :param min_amplitude_threshold: minimum predicted amplitude to trade, in percent (same unit as
                                        TargetAmplitude and TargetOpportunity.threshold_pct).
        """
        super().__init__()
        self.is_running = False
        self.strategy_id = 1
        self.symbol = "AAPL"
        self.min_amplitude_threshold = min_amplitude_threshold

        self.feature_manager = FeatureManager()
        self.target_manager = TargetManager()

        self.amplitude_model = AmplitudeModel()
        self.direction_model = DirectionModel()
        self.opportunity_model = OpportunityModel()

        self.training_data_shift = 30
        self.current_data = pd.DataFrame()

    def train(self, training_data: pd.DataFrame) -> None:
        fm = self.feature_manager
        # Base indicators
        fm.register_indicator(FeatureID.SMA_5, SMA(window=5))
        fm.register_indicator(FeatureID.SMA_10, SMA(window=10))
        fm.register_indicator(FeatureID.SMA_20, SMA(window=20))
        fm.register_indicator(FeatureID.RSI, RSI(window=14))
        fm.register_indicator(FeatureID.MACD, MACD(short_window=12, long_window=26, signal_window=9))
        fm.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        fm.register_indicator(FeatureID.ATR, ATR(window=14))
        fm.register_indicator(FeatureID.VOLUME, Volume(window=14))
        # Composite feature sets
        fm.register_indicator(FeatureID.AMPLITUDE, AmplitudeFeatures())
        fm.register_indicator(FeatureID.DIRECTION, DirectionFeatures())
        fm.register_indicator(FeatureID.OPPORTUNITY, OpportunityFeatures())
        fm.clear_cache()

        tm = self.target_manager
        tm.register_target(TargetID.SHIFTED_PLUS_1, TargetPlus1())
        tm.register_target(TargetID.SHIFTED_PLUS_5, TargetPlus5())
        tm.register_target(TargetID.SHIFTED_PLUS_10, TargetPlus10())
        tm.register_target(TargetID.SHIFTED_PLUS_20, TargetPlus20())
        tm.register_target(TargetID.DIRECTION, TargetDirection())
        tm.register_target(TargetID.OPPORTUNITY, TargetOpportunity())
        tm.register_target(TargetID.AMPLITUDE, TargetAmplitude())
        tm.clear_cache()

        # Look-ahead protection: features stop `training_data_shift` bars before the targets' data ends
        self.current_data = training_data.copy().iloc[: -self.training_data_shift]
        fm.update(self.current_data)
        tm.update(training_data, training_data_shift=self.training_data_shift)

        for model in (self.amplitude_model, self.direction_model, self.opportunity_model):
            model.set_feature_manager(fm)
            model.set_target_manager(tm)
            model.train()

    def test(self) -> None:
        amplitude_accuracy = self.amplitude_model.test()
        direction_accuracy = self.direction_model.test()
        opportunity_accuracy = self.opportunity_model.test()

        amplitude_cv, _ = self.amplitude_model.cross_validate()
        direction_cv, _ = self.direction_model.cross_validate()
        opportunity_cv, _ = self.opportunity_model.cross_validate()

        logger.info("Amplitude model   | train R2: %.2f | CV R2: %.2f", amplitude_accuracy, amplitude_cv)
        logger.info("Direction model   | train acc: %.2f | CV acc: %.2f", direction_accuracy, direction_cv)
        logger.info("Opportunity model | train acc: %.2f | CV acc: %.2f", opportunity_accuracy, opportunity_cv)
        logger.info("A train score close to 1.0 means memorization: rely on the cross-validation (CV) scores.")

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            raise RuntimeError("The strategy must be started before receiving bars.")

        # Rolling window keeps the per-bar recomputation bounded
        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-200:]
        self.feature_manager.update(self.current_data)

        if not self.opportunity_model.predict():
            return []

        direction = self.direction_model.predict()  # 'UP' or 'DOWN'
        amplitude = self.amplitude_model.predict()  # in percent

        if abs(amplitude) < self.min_amplitude_threshold:
            return []

        # The direction and the sign of the amplitude must agree
        if not ((direction == "UP" and amplitude > 0) or (direction == "DOWN" and amplitude < 0)):
            return []

        side = "BUY" if direction == "UP" else "SELL"
        confidence = min(abs(amplitude) / 5.0, 1.0)  # capped at 5 %
        quantity = max(1, int(confidence * 10))
        return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side=side, quantity=quantity)]

    def stop(self) -> None:
        self.is_running = False

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running

    def get_status(self) -> bool:
        return self.is_running
