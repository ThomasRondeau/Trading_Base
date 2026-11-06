from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from quantlab.execution.exits import ExitPlan, Level
from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import SMA, Ratio, ZScore
from quantlab.strategies.base import Strategy


class PairTradingStrategy(Strategy):
    """
    Sell the outperformer and buy the underperformer when the smoothed z-score of the price ratio
    moves beyond `zscore_threshold`. Bars must carry `close_<SYMBOL>` columns for both symbols.

    Both legs form one group: entered all or nothing and closed together, when the z-score comes back
    within `exit_zscore`, on the time exit, or when the pair's combined P&L reaches `stop_loss_pct` /
    `take_profit_pct` of its gross notional (checked at each bar close). Each leg also carries a wide
    protective stop at the broker (`leg_stop_pct`); when it fires, the other leg is closed at once.
    """

    def __init__(
        self,
        zscore_threshold: float = 2.2,
        symbol_a: str = "AAPL",
        symbol_b: str = "MSFT",
        zscore_window: int = 30,
        smoothing_window: int = 3,
        exit_zscore: float | None = 0.5,
        leg_stop_pct: float | None = 5.0,
        **kwargs: Any,
    ) -> None:
        """
        :param exit_zscore: close the pair when |z-score| <= this (None: no z-score exit).
        :param leg_stop_pct: protective stop of each leg at the broker, in percent (0 / None: none).
        """
        super().__init__(symbol=symbol_a, **kwargs)
        if self.stop_atr or self.take_profit_atr:
            raise ValueError("Pair trading: the stop / target apply to the combined P&L, in percent (no ATR).")
        self.zscore_threshold = zscore_threshold
        self.exit_zscore = exit_zscore
        self.leg_stop_pct = leg_stop_pct if leg_stop_pct and leg_stop_pct > 0 else None
        self.symbol_a, self.symbol_b = symbol_a, symbol_b
        ratio = Ratio(f"close_{symbol_a}", f"close_{symbol_b}")
        # Smoothed over a few bars to avoid reacting to spikes
        self.zscore = SMA(smoothing_window, source=ZScore(ratio, zscore_window))

    def features(self) -> Sequence[Feature]:
        return (self.zscore,)

    def decide(self, bar: pd.Series) -> list[Order]:
        z = self.feed.row()[self.zscore.name]
        if self.exit_zscore is not None and abs(z) <= self.exit_zscore:
            self.close_positions(reason="zscore")
        if z >= self.zscore_threshold:
            return self.pair(Side.SELL, Side.BUY)
        if z <= -self.zscore_threshold:
            return self.pair(Side.BUY, Side.SELL)
        return []

    def pair(self, side_a: Side, side_b: Side) -> list[Order]:
        plan = ExitPlan(
            stop=Level.pct(self.stop_loss_pct / 100) if self.stop_loss_pct else None,
            target=Level.pct(self.take_profit_pct / 100) if self.take_profit_pct else None,
            holding_bars=self.holding_bars,
            leg_stop=self.leg_stop_pct / 100 if self.leg_stop_pct else None,
        )
        legs = [self.order(side_a, 10, self.symbol_a, plan), self.order(side_b, 10, self.symbol_b, plan)]
        return self.group(legs, plan)
