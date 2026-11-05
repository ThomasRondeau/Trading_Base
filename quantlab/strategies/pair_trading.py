from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ZScore
from quantlab.strategies.base import TradingStrategyInterface

logger = logging.getLogger(__name__)


class PairTradingStrategy(TradingStrategyInterface):
    """
    AAPL vs MSFT: sell the outperformer and buy the underperformer when the price ratio's
    smoothed z-score moves beyond `zscore_threshold`. Bars must carry `close_<SYMBOL>` columns.
    """

    def __init__(self, zscore_threshold: float = 2.2) -> None:
        super().__init__()
        self.strategy_id = 3
        self.symbol_a = "AAPL"
        self.symbol_b = "MSFT"
        self.is_running = False
        self.feature_manager = FeatureManager()
        self.current_data = pd.DataFrame()
        self.zscore_threshold = zscore_threshold
        self.zscore_window = 30
        self.smoothing_window = 3
        # History kept: enough for the smoothed z-score, without unbounded growth
        self.max_history = 4 * (self.zscore_window + self.smoothing_window)

    def _ratio(self, df: pd.DataFrame) -> pd.Series:
        return df[f"close_{self.symbol_a}"] / df[f"close_{self.symbol_b}"]

    def train(self, training_data: pd.DataFrame) -> None:
        self.current_data = training_data.copy()
        if "ratio" not in self.current_data.columns:
            self.current_data["ratio"] = self._ratio(self.current_data)

        self.feature_manager.register_indicator(FeatureID.ZSCORE, ZScore(window=self.zscore_window))
        self.feature_manager.update(self.current_data)

    def test(self) -> None:
        logger.info("PairTrading: smoothed mean reversion on the ratio z-score.")

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            return []

        new_row = bar.to_frame().T
        if "ratio" not in new_row.columns:
            new_row["ratio"] = self._ratio(new_row)

        self.current_data = pd.concat([self.current_data, new_row]).iloc[-self.max_history :]
        self.feature_manager.update(self.current_data)

        zscore_feature = self.feature_manager.get_feature(FeatureID.ZSCORE)
        if zscore_feature.zscore is None or len(zscore_feature.zscore) < 3:
            return []

        # Smooth over a few bars to avoid reacting to spikes
        z = zscore_feature.zscore.rolling(window=self.smoothing_window).mean().iloc[-1]

        if z >= self.zscore_threshold:
            return [
                Order(strategy_id=self.strategy_id, symbol=self.symbol_a, side="SELL", quantity=10),
                Order(strategy_id=self.strategy_id, symbol=self.symbol_b, side="BUY", quantity=10),
            ]
        if z <= -self.zscore_threshold:
            return [
                Order(strategy_id=self.strategy_id, symbol=self.symbol_a, side="BUY", quantity=10),
                Order(strategy_id=self.strategy_id, symbol=self.symbol_b, side="SELL", quantity=10),
            ]
        return []

    def stop(self) -> None:
        self.is_running = False

    def get_status(self) -> bool:
        return self.is_running

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running
