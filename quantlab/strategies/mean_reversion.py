from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from quantlab.execution.exits import Level
from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import ATR, RSI, BollingerBands, VolumeRatio
from quantlab.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """
    Fade moves well outside the Bollinger Bands when the RSI is extreme and volume spikes.

    With `target_middle_band`, the target of each position is the Bollinger middle band at the signal
    (the mean the price should revert to) instead of `take_profit_pct`.
    """

    def __init__(
        self, rsi_threshold: float = 30, symbol: str = "AAPL", target_middle_band: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.rsi_threshold = rsi_threshold
        self.target_middle_band = target_middle_band
        self.bollinger, self.rsi, self.volume, self.atr = BollingerBands(20, 2), RSI(14), VolumeRatio(20), ATR(14)

    def features(self) -> Sequence[Feature]:
        return (self.bollinger, self.rsi, self.volume, self.atr)

    def decide(self, bar: pd.Series) -> list[Order]:
        row = self.feed.row()
        price, rsi, atr = bar["close"], row[self.rsi.name], row[self.atr.name]
        upper, lower = row[self.bollinger.upper], row[self.bollinger.lower]

        volume_spike = row[self.volume.name] > 1.2  # volume confirms the potential reversal
        # The price must be significantly outside the bands
        oversold = price < (lower - 0.2 * atr)
        overbought = price > (upper + 0.2 * atr)

        plan = self.exit_plan()
        if self.target_middle_band:
            plan = replace(plan, target=Level.at((upper + lower) / 2))
        if oversold and rsi < self.rsi_threshold and volume_spike:
            return [self.order(Side.BUY, 10, exit_plan=plan)]
        if overbought and rsi > (100 - self.rsi_threshold) and volume_spike:
            return [self.order(Side.SELL, 10, exit_plan=plan)]
        return []
