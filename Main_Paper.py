"""
Run a strategy on live IBKR bars with the same runner as the backtests.

- IBKR_READ_ONLY=true (default): *shadow* mode. Live bars, fills simulated by the SimulatedBroker,
  no order ever reaches TWS.
- IBKR_READ_ONLY=false: paper trading through IBKRBroker. Orders are refused on a real-money account
  unless IBKR_ALLOW_LIVE=true.

Stop with Ctrl+C. Journal, persisted state and replay come with step 5 of the refactor plan.
"""

import logging

from quantlab.execution.broker import Broker
from quantlab.execution.ibkr.bar_feed import IBKRBarFeed
from quantlab.execution.ibkr.broker import IBKRBroker
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.runner import StrategyRunner
from quantlab.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

SYMBOL = "AAPL"
RISK = RiskLimits(max_order_quantity=10, max_position_quantity=50, max_daily_loss_pct=0.01)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = IBKRClient()
    try:
        client.connect()
    except ConnectionError as exc:
        logger.error("%s", exc)
        return

    try:
        broker: Broker
        if client.settings.read_only:
            logger.info("Shadow mode (read-only): fills are simulated, no order is sent")
            broker = SimulatedBroker()
        else:
            client.ensure_account_allowed()
            broker = IBKRBroker(client)
            mismatches = broker.reconcile({})
            if mismatches:
                raise RuntimeError(f"IBKR positions differ from the runner's (flat) book: {mismatches}")

        feed = IBKRBarFeed(client, [SYMBOL], bar_seconds=60)
        strategy = TrendFollowingStrategy(symbol=SYMBOL)
        history = feed.history("2 D")
        logger.info("Warm-up history: %d bars up to %s", len(history), history.index[-1])
        strategy.fit(history)

        runner = StrategyRunner([strategy], broker, RiskManager(RISK))
        try:
            runner.run(feed)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        logger.info("Equity %.2f, positions %s", runner.portfolio.equity, runner.portfolio.net_positions())
    except Exception:
        logger.exception("Live run failed")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
