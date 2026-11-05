import logging

from quantlab.backtest.engine import BacktestEngine
from quantlab.data.data_manager import DataManager
from quantlab.strategies.cascade import CascadeStrategy

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting cascade backtest (yfinance)...")
    try:
        engine = BacktestEngine()
        data_manager = DataManager()
        engine.set_training_data(data_manager.collect_data_yfinance("AAPL", 50, 5))
        engine.set_backtest_data(data_manager.collect_data_yfinance("AAPL", 5, 0))
        engine.add_strategy(CascadeStrategy())
        engine.train_strategies()
        engine.test_strategies()
        engine.run()
    except Exception:
        logger.exception("Backtest failed")


if __name__ == "__main__":
    main()
