from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ATR, SMA
from quantlab.strategies.base import TradingStrategyInterface

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(TradingStrategyInterface):
    """Fast/slow SMA crossover."""

    def __init__(self, sma_fast: int = 5, sma_slow: int = 20) -> None:
        super().__init__()
        self.strategy_id = 4
        self.symbol = "AAPL"
        self.is_running = False
        self.feature_manager = FeatureManager()
        self.current_data = pd.DataFrame()
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow

    def train(self, training_data: pd.DataFrame) -> None:
        fm = self.feature_manager
        fm.register_indicator(FeatureID.SMA_FAST, SMA(window=self.sma_fast))
        fm.register_indicator(FeatureID.SMA_SLOW, SMA(window=self.sma_slow))
        fm.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        fm.update(self.current_data)

    def test(self) -> None:
        logger.info("TrendFollowing: SMA %s/%s crossover.", self.sma_fast, self.sma_slow)

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-100:]
        self.feature_manager.update(self.current_data)

        fast = self.feature_manager.get_feature(FeatureID.SMA_FAST).sma
        slow = self.feature_manager.get_feature(FeatureID.SMA_SLOW).sma
        if len(fast) < 5:
            return []

        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

        if crossed_up:
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="BUY", quantity=10)]
        if crossed_down:
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="SELL", quantity=10)]
        return []

    def stop(self) -> None:
        self.is_running = False

    def get_status(self) -> bool:
        return self.is_running

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running
