from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.composite import DirectionFeatures
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ATR, MACD, RSI, SMA, BollingerBands, Volume
from quantlab.models.cascade_models import DirectionModel
from quantlab.strategies.base import TradingStrategyInterface
from quantlab.targets.forward import TargetDirection
from quantlab.targets.manager import TargetManager
from quantlab.targets.target_id import TargetID

logger = logging.getLogger(__name__)


class MLHybridStrategy(TradingStrategyInterface):
    """Mean reversion entries (Bollinger + RSI + volume) confirmed by the XGBoost direction model."""

    def __init__(self, max_depth: int = 4) -> None:
        super().__init__()
        self.strategy_id = 6
        self.symbol = "AAPL"
        self.is_running = False
        self.feature_manager = FeatureManager()
        self.target_manager = TargetManager()
        self.direction_model = DirectionModel(max_depth=max_depth)
        self.current_data = pd.DataFrame()
        self.training_data_shift = 30

    def train(self, training_data: pd.DataFrame) -> None:
        fm = self.feature_manager
        fm.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        fm.register_indicator(FeatureID.RSI, RSI(window=14))
        fm.register_indicator(FeatureID.VOLUME, Volume(window=20))
        fm.register_indicator(FeatureID.ATR, ATR(window=14))
        fm.register_indicator(FeatureID.MACD, MACD(short_window=12, long_window=26, signal_window=9))
        fm.register_indicator(FeatureID.SMA_20, SMA(window=20))
        fm.register_indicator(FeatureID.DIRECTION, DirectionFeatures())

        self.target_manager.register_target(TargetID.DIRECTION, TargetDirection())

        # Look-ahead protection: see TargetInterface
        self.current_data = training_data.copy().iloc[: -self.training_data_shift]
        fm.update(self.current_data)
        self.target_manager.update(training_data, training_data_shift=self.training_data_shift)

        self.direction_model.set_feature_manager(fm)
        self.direction_model.set_target_manager(self.target_manager)
        self.direction_model.train()

    def test(self) -> None:
        logger.info("MLHybrid: direction model train accuracy %.2f", self.direction_model.test())

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-200:]
        self.feature_manager.update(self.current_data)

        bb = self.feature_manager.get_feature(FeatureID.BOLLINGER_BANDS)
        rsi = self.feature_manager.get_feature(FeatureID.RSI).rsi
        vol = self.feature_manager.get_feature(FeatureID.VOLUME).volume_ratio

        price = bar["close"]
        curr_rsi = rsi.iloc[-1]
        curr_vol = vol.iloc[-1]
        upper = bb.upper_band.iloc[-1]
        lower = bb.lower_band.iloc[-1]

        tech_buy = price <= lower and curr_rsi < 35 and curr_vol > 1.1
        tech_sell = price >= upper and curr_rsi > 65 and curr_vol > 1.1
        if not (tech_buy or tech_sell):
            return []

        ml_direction = self.direction_model.predict()  # 5-bar direction: 'UP' or 'DOWN'
        if tech_buy and ml_direction == "UP":
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="BUY", quantity=10)]
        if tech_sell and ml_direction == "DOWN":
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="SELL", quantity=10)]
        return []

    def stop(self) -> None:
        self.is_running = False

    def get_status(self) -> bool:
        return self.is_running

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running
