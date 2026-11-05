import logging
import threading
import time

from quantlab.backtest.engine import BacktestEngine
from quantlab.data.data_manager import DataManager
from quantlab.execution.ibkr.trading_app import TradingApp
from quantlab.strategies.cascade import CascadeStrategy

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = TradingApp.get_instance()
    app.connect("127.0.0.1", 7497, clientId=5)
    threading.Thread(target=app.run, daemon=True).start()

    timeout = 10
    start_time = time.time()
    while not app.connected and (time.time() - start_time) < timeout:
        logger.info("Waiting for the TWS/IB Gateway connection...")
        time.sleep(1)
    if not app.connected:
        logger.error("Not connected to TWS/IB Gateway")
        return

    try:
        data_manager = DataManager()
        engine = BacktestEngine()
        engine.set_training_data(data_manager.collect_training_data("AAPL", 10, 3))
        engine.set_backtest_data(data_manager.collect_training_data("AAPL", 3))
        engine.add_strategy(CascadeStrategy())
        engine.train_strategies()
        engine.run()
    except Exception:
        logger.exception("Strategy run failed")
    finally:
        if app.connected:
            app.disconnect()
            logger.info("Disconnected")


if __name__ == "__main__":
    main()
