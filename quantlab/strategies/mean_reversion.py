from __future__ import annotations

import logging

import pandas as pd

from quantlab.execution.orders import Order
from quantlab.features.feature_id import FeatureID
from quantlab.features.manager import FeatureManager
from quantlab.features.technical import ATR, RSI, BollingerBands, Volume
from quantlab.strategies.base import TradingStrategyInterface

logger = logging.getLogger(__name__)


class MeanReversionStrategy(TradingStrategyInterface):
    """Fade moves well outside the Bollinger Bands when RSI is extreme and volume spikes."""

    def __init__(self, rsi_threshold: float = 30) -> None:
        super().__init__()
        self.strategy_id = 5
        self.symbol = "AAPL"
        self.is_running = False
        self.feature_manager = FeatureManager()
        self.current_data = pd.DataFrame()
        self.rsi_threshold = rsi_threshold

    def train(self, training_data: pd.DataFrame) -> None:
        fm = self.feature_manager
        fm.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        fm.register_indicator(FeatureID.RSI, RSI(window=14))
        fm.register_indicator(FeatureID.VOLUME, Volume(window=20))
        fm.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        fm.update(self.current_data)

    def test(self) -> None:
        logger.info("MeanReversion: BB + RSI(%s) + volume spike + ATR filter.", self.rsi_threshold)

    def on_bar(self, bar: pd.Series) -> list[Order]:
        if not self.is_running:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-100:]
        self.feature_manager.update(self.current_data)

        bb = self.feature_manager.get_feature(FeatureID.BOLLINGER_BANDS)
        rsi = self.feature_manager.get_feature(FeatureID.RSI).rsi
        vol = self.feature_manager.get_feature(FeatureID.VOLUME).volume_ratio
        atr = self.feature_manager.get_feature(FeatureID.ATR).atr

        if len(rsi) < 1 or bb.upper_band is None or len(vol) < 1:
            return []

        price = bar["close"]
        curr_rsi = rsi.iloc[-1]
        curr_vol = vol.iloc[-1]
        curr_atr = atr.iloc[-1]
        upper = bb.upper_band.iloc[-1]
        lower = bb.lower_band.iloc[-1]

        vol_spike = curr_vol > 1.2  # volume confirms the potential reversal
        # The price must be significantly outside the bands
        oversold = price < (lower - 0.2 * curr_atr)
        overbought = price > (upper + 0.2 * curr_atr)

        if oversold and curr_rsi < self.rsi_threshold and vol_spike:
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="BUY", quantity=10)]
        if overbought and curr_rsi > (100 - self.rsi_threshold) and vol_spike:
            return [Order(strategy_id=self.strategy_id, symbol=self.symbol, side="SELL", quantity=10)]
        return []

    def stop(self) -> None:
        self.is_running = False

    def get_status(self) -> bool:
        return self.is_running

    def update_status(self, is_running: bool) -> None:
        self.is_running = is_running
