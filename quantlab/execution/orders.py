"""Order emitted by a strategy."""

from __future__ import annotations


class Order:
    def __init__(self, strategy_id: int, symbol: str, side: str, quantity: int) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.side = side  # 'BUY' or 'SELL'
        self.quantity = quantity

    def __repr__(self) -> str:
        return f"Order(strategy_id={self.strategy_id}, symbol={self.symbol!r}, side={self.side!r}, qty={self.quantity})"
