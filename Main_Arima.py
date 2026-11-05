import logging

from quantlab.backtest.run import backtest
from quantlab.data.data_manager import DataManager
from quantlab.strategies.arima import ArimaStrategy

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting ARIMA backtest (yfinance)...")
    try:
        data_manager = DataManager()
        backtest(
            [ArimaStrategy()],
            data_manager.collect_data_yfinance("AAPL", 50, 5),
            data_manager.collect_data_yfinance("AAPL", 5, 0),
        )
    except Exception:
        logger.exception("Backtest failed")


if __name__ == "__main__":
    main()
