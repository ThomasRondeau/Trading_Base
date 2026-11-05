import logging

import pandas as pd

from quantlab.backtest.engine import BacktestEngine
from quantlab.data.data_manager import DataManager
from quantlab.strategies.pair_trading import PairTradingStrategy

logger = logging.getLogger(__name__)


def prepare_pair_data(symbol_a: str, symbol_b: str, days_back: int, until: int) -> pd.DataFrame:
    """Bars of both symbols aligned on timestamps, with `close_<SYMBOL>` columns."""
    data_manager = DataManager()
    data_a = data_manager.collect_data_yfinance(symbol_a, days_back, until)
    data_b = data_manager.collect_data_yfinance(symbol_b, days_back, until)
    combined = pd.merge(data_a, data_b, left_index=True, right_index=True, suffixes=(f"_{symbol_a}", f"_{symbol_b}"))
    combined["close"] = combined[f"close_{symbol_a}"]  # generic column expected by single-symbol code
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting pair trading backtest (AAPL vs MSFT)...")
    try:
        engine = BacktestEngine()
        engine.set_training_data(prepare_pair_data("AAPL", "MSFT", 50, 5))
        engine.set_backtest_data(prepare_pair_data("AAPL", "MSFT", 5, 0))
        engine.add_strategy(PairTradingStrategy())
        engine.train_strategies()
        engine.run()
    except Exception:
        logger.exception("Backtest failed")


if __name__ == "__main__":
    main()
