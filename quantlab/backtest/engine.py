"""
Bar-by-bar backtest.

Known limitations (fixed in step 4 of the refactor plan): orders fill at the close of the signal bar,
exits are hard-coded (5 bars / SL / TP checked on closes only), and the metrics sum per-trade
percentages instead of using an equity curve.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from quantlab.execution.risk import RiskManager
from quantlab.strategies.base import TradingStrategyInterface

logger = logging.getLogger(__name__)

BASIS_POINT = 0.0001
MAX_BARS_HELD = 5


def _price_column(bar: pd.Series, symbol: str) -> str:
    """Multi-symbol bars carry `close_<SYMBOL>`; single-symbol bars just `close`."""
    return f"close_{symbol}" if f"close_{symbol}" in bar else "close"


def _fees(quantity: float, entry_price: float, exit_price: float) -> float:
    """|position| * basis point * (entry + exit)."""
    return quantity * BASIS_POINT * (entry_price + exit_price)


class BacktestEngine:
    def __init__(self) -> None:
        self.strategies: list[TradingStrategyInterface] = []
        self.is_running = False
        self.risk_manager = RiskManager()
        self.training_data: pd.DataFrame | None = None
        self.backtest_data: pd.DataFrame | None = None

    def add_strategy(self, strategy: TradingStrategyInterface) -> None:
        self.strategies.append(strategy)
        logger.info("Strategy %s added to engine.", type(strategy).__name__)

    def set_backtest_data(self, backtest_data: pd.DataFrame) -> None:
        self.backtest_data = backtest_data
        logger.info("Backtest data set with %d bars.", len(backtest_data))

    def set_training_data(self, training_data: pd.DataFrame) -> None:
        self.training_data = training_data
        logger.info("Training data set with %d bars.", len(training_data))

    def train_strategies(self) -> None:
        if self.training_data is None:
            raise ValueError("Training data must be set before training.")
        for strategy in self.strategies:
            strategy.train(self.training_data)

    def test_strategies(self) -> None:
        for strategy in self.strategies:
            strategy.test()

    def run(self) -> dict[str, float]:
        if self.backtest_data is None or not self.strategies:
            raise ValueError("Data and strategies must be set before running the backtest.")

        logger.info("Backtest started with strategies: %s", ", ".join(type(s).__name__ for s in self.strategies))
        self.is_running = True
        for strategy in self.strategies:
            strategy.update_status(True)

        trades: list[dict[str, Any]] = []
        active_positions: list[dict[str, Any]] = []  # entry_price, side, quantity, bars_held, symbol

        stop_loss_pct = self.risk_manager.stop_loss_pct
        take_profit_pct = self.risk_manager.take_profit_pct

        total_bars = len(self.backtest_data)
        for count, (timestamp, bar) in enumerate(self.backtest_data.iterrows(), start=1):
            if count % 1000 == 0:
                logger.info("Progress: %d/%d bars processed...", count, total_bars)

            # 1. Exits: time (MAX_BARS_HELD bars), stop loss or take profit
            remaining_positions = []
            for pos in active_positions:
                pos["bars_held"] += 1
                symbol = pos.get("symbol", "unknown")
                current_price = bar[_price_column(bar, symbol)]

                if pos["side"] == "BUY":
                    pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                else:
                    pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

                if pos["bars_held"] >= MAX_BARS_HELD:
                    exit_reason = "Time"
                elif pnl_pct <= -stop_loss_pct:
                    exit_reason = "Stop Loss"
                elif pnl_pct >= take_profit_pct:
                    exit_reason = "Take Profit"
                else:
                    remaining_positions.append(pos)
                    continue

                trades.append(self._close_trade(pos, current_price, timestamp, exit_reason))
            active_positions = remaining_positions

            # 2. New orders, filled immediately at the current bar's close
            orders = [order for strategy in self.strategies for order in strategy.on_bar(bar)]
            for order in orders:
                active_positions.append(
                    {
                        "symbol": order.symbol,
                        "entry_price": bar[_price_column(bar, order.symbol)],
                        "side": order.side,
                        "quantity": order.quantity,
                        "bars_held": 0,
                    }
                )

        # Close remaining positions at the last price
        last_bar = self.backtest_data.iloc[-1]
        last_timestamp = self.backtest_data.index[-1]
        for pos in active_positions:
            last_price = last_bar[_price_column(last_bar, pos.get("symbol", "unknown"))]
            trades.append(self._close_trade(pos, last_price, last_timestamp, reason=None))

        return self.print_summary(trades)

    @staticmethod
    def _close_trade(pos: dict[str, Any], exit_price: float, timestamp: Any, reason: str | None) -> dict[str, Any]:
        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        unit_pnl = (exit_price - entry_price) if pos["side"] == "BUY" else (entry_price - exit_price)
        pnl_total = unit_pnl * quantity
        fees = _fees(quantity, entry_price, exit_price)
        pnl_net = pnl_total - fees
        trade = {
            "timestamp": timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "side": pos["side"],
            "symbol": pos.get("symbol", "unknown"),
            "quantity": quantity,
            "pnl": pnl_total,
            "pnl_net": pnl_net,
            # Positions still open at the end historically report the per-share percentage
            "pnl_percent": (pnl_total / (quantity * entry_price)) * 100
            if reason is not None
            else (unit_pnl / entry_price) * 100,
            "pnl_net_percent": (pnl_net / (quantity * entry_price)) * 100,
            "fees": fees,
        }
        if reason is not None:
            trade["reason"] = reason
        return trade

    def print_summary(self, trades: list[dict[str, Any]]) -> dict[str, float]:
        if not trades:
            logger.info("Backtest summary: no trades executed.")
            return {"total_trades": 0, "win_rate": 0, "total_pnl_net_percent": 0, "sharpe_ratio": 0, "max_drawdown": 0}

        df_trades = pd.DataFrame(trades)

        total_trades = len(trades)
        win_rate = (len(df_trades[df_trades["pnl_net"] > 0]) / total_trades) * 100
        total_pnl_gross = df_trades["pnl_percent"].sum()
        total_pnl_net = df_trades["pnl_net_percent"].sum()
        avg_pnl_net = df_trades["pnl_net_percent"].mean()
        total_fees = df_trades["fees"].sum()

        # Simplified daily Sharpe: per-trade percentages summed by date
        df_trades["date"] = pd.to_datetime(df_trades["timestamp"]).dt.date
        daily_returns = df_trades.groupby("date")["pnl_net_percent"].sum()

        sharpe_ratio = 0.0
        if len(daily_returns) > 1:
            std = daily_returns.std()
            if std != 0:
                sharpe_ratio = (daily_returns.mean() / std) * (252**0.5)

        cumulative_returns = (1 + daily_returns / 100).cumprod()
        drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
        max_drawdown = drawdown.min() * 100

        logger.info(
            "Backtest (net of fees): trades=%d win_rate=%.2f%% fees=%.2f pnl_gross=%.2f%% pnl_net=%.2f%% "
            "avg_pnl_net=%.2f%% sharpe=%.2f max_dd=%.2f%%",
            total_trades,
            win_rate,
            total_fees,
            total_pnl_gross,
            total_pnl_net,
            avg_pnl_net,
            sharpe_ratio,
            max_drawdown,
        )
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_fees": total_fees,
            "total_pnl_gross_percent": total_pnl_gross,
            "total_pnl_net_percent": total_pnl_net,
            "avg_pnl_net_percent": avg_pnl_net,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    def stop(self) -> None:
        self.is_running = False
        logger.info("BacktestEngine stopped.")
