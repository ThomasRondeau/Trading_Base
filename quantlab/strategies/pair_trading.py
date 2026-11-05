from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import SMA, Ratio, ZScore
from quantlab.strategies.base import Strategy


class PairTradingStrategy(Strategy):
    """
    Sell the outperformer and buy the underperformer when the smoothed z-score of the price ratio
    moves beyond `zscore_threshold`. Bars must carry `close_<SYMBOL>` columns for both symbols.
    """

    def __init__(
        self,
        zscore_threshold: float = 2.2,
        symbol_a: str = "AAPL",
        symbol_b: str = "MSFT",
        zscore_window: int = 30,
        smoothing_window: int = 3,
        **kwargs: str,
    ) -> None:
        super().__init__(symbol=symbol_a, **kwargs)
        self.zscore_threshold = zscore_threshold
        self.symbol_a, self.symbol_b = symbol_a, symbol_b
        ratio = Ratio(f"close_{symbol_a}", f"close_{symbol_b}")
        # Smoothed over a few bars to avoid reacting to spikes
        self.zscore = SMA(smoothing_window, source=ZScore(ratio, zscore_window))

    def features(self) -> Sequence[Feature]:
        return (self.zscore,)

    def decide(self, bar: pd.Series) -> list[Order]:
        z = self.feed.row()[self.zscore.name]
        if z >= self.zscore_threshold:
            return [self.order(Side.SELL, 10, self.symbol_a), self.order(Side.BUY, 10, self.symbol_b)]
        if z <= -self.zscore_threshold:
            return [self.order(Side.BUY, 10, self.symbol_a), self.order(Side.SELL, 10, self.symbol_b)]
        return []
