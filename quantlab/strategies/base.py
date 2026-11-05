"""
Strategy contract shared by backtest and live.

A strategy declares its features (`features()`), optionally fits models in `fit`, and decides on
each bar in `decide`. The base class owns the running state and the `FeatureFeed`, so a concrete
strategy never manages buffers or caches and runs unchanged on historical or live bars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import pandas as pd

from quantlab.execution.orders import Fill, Order, Side
from quantlab.features.base import Feature
from quantlab.features.pipeline import FeaturePipeline
from quantlab.runtime.feature_feed import FeatureFeed


class Strategy(ABC):
    #: raw bars the strategy reads itself through `self.feed.bars(n)` (beyond the features' warmup)
    min_history: int = 0
    #: bars after which each position is closed at market (None: only the protective stop / target close it)
    holding_bars: int | None = 5

    def __init__(self, symbol: str = "AAPL", strategy_id: str | None = None) -> None:
        self.symbol = symbol
        self.strategy_id = strategy_id or type(self).__name__
        self.is_running = False
        self._feed: FeatureFeed | None = None

    # --- to implement ----------------------------------------------------------------------------------

    def features(self) -> Sequence[Feature]:
        """Features read by `decide` (and by the models)."""
        return ()

    @abstractmethod
    def decide(self, bar: pd.Series) -> list[Order]:
        """Orders for the current bar; `self.feed` already includes it."""

    def train(self, history: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Fit models on `history` (no-op for rule-based strategies)."""

    # --- lifecycle -------------------------------------------------------------------------------------

    @property
    def feed(self) -> FeatureFeed:
        if self._feed is None:
            raise RuntimeError(f"{self.strategy_id}: call fit(history) before using the feature feed.")
        return self._feed

    def fit(self, history: pd.DataFrame) -> None:
        """Warm the feature buffer with `history`, then train the models."""
        self._feed = FeatureFeed(FeaturePipeline(self.features()), min_history=self.min_history)
        self._feed.warm_start(history)
        self.train(history)

    def prepare(self, upcoming: pd.DataFrame) -> None:
        """Backtest only: precompute the features of the upcoming bars (same values as bar by bar)."""
        self.feed.precompute(upcoming)

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def on_bar(self, bar: pd.Series) -> list[Order]:
        """Feed the bar and return the orders to send (none while stopped or not warmed up)."""
        if not self.is_running:
            return []
        self.feed.update(bar)
        if not self.feed.ready:
            return []
        return self.decide(bar)

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 - optional hook
        """Called by the broker when one of this strategy's orders is (partially) filled."""

    # --- persistence (live restarts) -------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "symbol": self.symbol, "is_running": self.is_running}

    def set_state(self, state: dict[str, Any]) -> None:
        self.is_running = bool(state.get("is_running", False))

    # --- helpers ---------------------------------------------------------------------------------------

    def order(self, side: Side, quantity: int, symbol: str | None = None) -> Order:
        return Order(symbol=symbol or self.symbol, side=side, quantity=quantity, strategy_id=self.strategy_id)
