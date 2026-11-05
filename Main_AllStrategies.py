import logging

import pandas as pd

from quantlab.backtest.engine import BacktestEngine
from quantlab.data.massive import MassiveAPIClient
from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.base import TradingStrategyInterface
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

SYMBOL = "SPY"  # S&P 500 ETF: deep intraday liquidity
IN_SAMPLE_FRACTION = 0.83  # ~5 months in-sample, ~1 month out-of-sample


def all_strategies() -> list[TradingStrategyInterface]:
    strategies: list[TradingStrategyInterface] = [
        CascadeStrategy(),
        ArimaStrategy(),
        MeanReversionStrategy(),
        MLHybridStrategy(),
        TrendFollowingStrategy(),
    ]
    for strategy in strategies:
        strategy.symbol = SYMBOL  # type: ignore[attr-defined]
    return strategies


def run_backtest(strategies: list[TradingStrategyInterface], train: pd.DataFrame, test: pd.DataFrame) -> None:
    engine = BacktestEngine()
    engine.set_training_data(train)
    engine.set_backtest_data(test)
    for strategy in strategies:
        engine.add_strategy(strategy)
    engine.train_strategies()
    engine.run()


def main() -> None:
    """Every strategy on ~6 months of 1-minute Massive data: one by one, then together as a portfolio."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        full_data = MassiveAPIClient().fetch_data(SYMBOL, days_back=180, until=0)
        if full_data.empty:
            raise ValueError("No data received from Massive API.")

        split_idx = int(len(full_data) * IN_SAMPLE_FRACTION)
        train, test = full_data.iloc[:split_idx], full_data.iloc[split_idx:]
        logger.info("In-sample:     %s to %s (%d bars)", train.index[0], train.index[-1], len(train))
        logger.info("Out-of-sample: %s to %s (%d bars)", test.index[0], test.index[-1], len(test))

        for strategy in all_strategies():
            logger.info("=== Strategy: %s ===", type(strategy).__name__)
            run_backtest([strategy], train, test)

        logger.info("=== Portfolio: all strategies ===")
        run_backtest(all_strategies(), train, test)
    except Exception:
        logger.exception("Run failed")


if __name__ == "__main__":
    main()
