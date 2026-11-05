"""
Journaled session on live IBKR bars, with the same runner as the backtests.

    py -3.10 Main_Paper.py --strategy trend --symbol AAPL --session trend_aapl
    py -3.10 Main_Paper.py --session trend_aapl --resume       # after a restart / Ctrl+C
    py -3.10 Main_Replay.py --session trend_aapl               # live vs backtest comparison

Mode from the settings: IBKR_READ_ONLY=true (default) -> shadow (simulated fills, no order sent);
IBKR_READ_ONLY=false -> paper trading on a DU... account. A real-money account also needs --live and
typing its account id.
"""

import argparse
import dataclasses
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

from quantlab.config import IBKRSettings
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.execution.ibkr.live import LiveConfig, run_live_session
from quantlab.execution.risk import RiskLimits
from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.base import Strategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

STRATEGIES: dict[str, Callable[[str], Strategy]] = {
    "trend": lambda symbol: TrendFollowingStrategy(symbol=symbol),
    "mean_reversion": lambda symbol: MeanReversionStrategy(symbol=symbol),
    "cascade": lambda symbol: CascadeStrategy(symbol=symbol),
    "ml_hybrid": lambda symbol: MLHybridStrategy(symbol=symbol),
    "arima": lambda symbol: ArimaStrategy(symbol=symbol),
}
# History used to fit / warm the strategies (the ML ones train on it)
HISTORY = {"trend": "2 D", "mean_reversion": "2 D", "arima": "2 D", "cascade": "10 D", "ml_hybrid": "10 D"}
RISK = RiskLimits(max_order_quantity=10, max_position_quantity=50, max_daily_loss_pct=0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="trend")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--session", help="session name (default: <strategy>_<symbol>_<date>)")
    parser.add_argument("--sessions-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--resume", action="store_true", help="resume an interrupted session")
    parser.add_argument("--live", action="store_true", help="allow a real-money account (asks for confirmation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = IBKRSettings.from_env()
    client = IBKRClient(settings)
    try:
        client.connect()
    except ConnectionError as exc:
        logger.error("%s", exc)
        return

    try:
        if args.live and not settings.read_only and not client.is_paper:
            typed = input(f"REAL MONEY on {', '.join(client.accounts)}. Type the account id to confirm: ")
            if typed.strip() not in client.accounts:
                logger.error("Not confirmed: aborting")
                return
            client.settings = dataclasses.replace(settings, allow_live_account=True)
        name = args.session or f"{args.strategy}_{args.symbol}_{date.today():%Y%m%d}"
        run_live_session(
            client,
            lambda: [STRATEGIES[args.strategy](args.symbol)],
            args.sessions_dir / name,
            LiveConfig(symbols=[args.symbol], history_duration=HISTORY[args.strategy], risk=RISK),
            resume=args.resume,
        )
    except Exception:
        logger.exception("Session failed")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
