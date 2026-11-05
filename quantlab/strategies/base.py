from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quantlab.execution.orders import Order


class TradingStrategyInterface(ABC):
    """Contract between a strategy and the engine that drives it."""

    def __init__(self) -> None:
        self.name = type(self).__name__

    @abstractmethod
    def train(self, training_data: pd.DataFrame) -> None:
        """Register features/targets and fit models on the training data."""

    @abstractmethod
    def test(self) -> Any:
        """Report diagnostics on the trained models."""

    @abstractmethod
    def update_status(self, is_running: bool) -> None:
        """Start (True) or pause (False) the strategy."""

    @abstractmethod
    def on_bar(self, bar: pd.Series) -> list[Order]:
        """Process a new bar and return the orders to send (empty list for no trade)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the strategy."""

    @abstractmethod
    def get_status(self) -> bool:
        """Whether the strategy is running."""
