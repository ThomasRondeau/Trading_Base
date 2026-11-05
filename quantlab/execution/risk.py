"""Risk limits. Only `stop_loss_pct` and `take_profit_pct` are used by the backtest engine so far."""

from __future__ import annotations

from typing import Any


class RiskManager:
    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_size_pct: float = 0.1,  # 10% of capital per position
        max_risk_per_trade_pct: float = 0.02,  # 2% of capital at risk per trade
        max_total_exposure_pct: float = 1.0,  # 100% of capital in total
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.05,  # 5% take profit
    ) -> None:
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Tracking
        self.positions: dict[str, dict[str, Any]] = {}  # {symbol: {quantity, entry_price, stop_price}}
        self.trades_history: list[dict[str, Any]] = []
        self.pnl: float = 0
