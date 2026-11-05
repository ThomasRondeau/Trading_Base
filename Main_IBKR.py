"""Backtest of the cascade strategy on 1-minute IBKR history (read-only: data requests only)."""

import logging

from quantlab.backtest.run import backtest
from quantlab.data.data_manager import DataManager
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.strategies.cascade import CascadeStrategy

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = IBKRClient(auto_reconnect=False)
    try:
        client.connect()
    except ConnectionError as exc:
        logger.error("%s", exc)
        return

    try:
        data_manager = DataManager()
        backtest(
            [CascadeStrategy()],
            data_manager.collect_data_ibkr(client, "AAPL", 10, 3),
            data_manager.collect_data_ibkr(client, "AAPL", 3),
        )
    except Exception:
        logger.exception("Strategy run failed")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
