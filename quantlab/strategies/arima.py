from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ATR
from quantlab.models.arima import ArimaModel
from quantlab.strategies.base import TradingStrategyInterface
from quantlab.targets.manager import TargetManager

logger = logging.getLogger(__name__)


class ArimaStrategy(TradingStrategyInterface):
    """Trade the direction of the ARIMA one-bar forecast when it beats trading costs."""

    def __init__(self, order: tuple[int, int, int] = (2, 1, 0), min_edge_bps: float = 2.0) -> None:
        """
        :param order: ARIMA (p, d, q) order.
        :param min_edge_bps: minimum predicted move (in basis points of the price) to trade;
                             defaults to the round-trip fees (2 x 1 bp).
        """
        super().__init__()
        self.strategy_id = 2
        self.symbol = "AAPL"
        self.is_running = False
        self.feature_manager = FeatureManager()
        self.target_manager = TargetManager()
        self.model = ArimaModel(order=order)
        self.min_edge_bps = min_edge_bps
        self.current_data = pd.DataFrame()

    def train(self, training_data: pd.DataFrame) -> None:
        self.feature_manager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        self.feature_manager.update(self.current_data)
        self.model.set_feature_manager(self.feature_manager)
        self.model.train()

    def test(self) -> None:
        logger.info("ARIMA: no training metrics (rolling forecast).")

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-100:]
        self.feature_manager.update(self.current_data)

        prediction = self.model.predict()  # predicted price change
        edge_bps = abs(prediction) / float(bar["close"]) * 1e4
        if edge_bps < self.min_edge_bps:
            return []

        side = "BUY" if prediction > 0 else "SELL"
        return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side=side, quantity=5)]

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running

    def stop(self) -> None:
        self.is_running = False

    def get_status(self) -> bool:
        return self.is_running
