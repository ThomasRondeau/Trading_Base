"""Replay a journaled session (Main_Paper.py) in the backtester and report the live / backtest gap."""

import argparse
import logging
from pathlib import Path

from quantlab.analysis.replay import replay_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--sessions-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--slippage-bps", type=float, help="simulated slippage (default: the session's)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    report = replay_session(args.sessions_dir / args.session, slippage_bps=args.slippage_bps)
    report.log()
    if not report.entry_fills.empty:
        logging.getLogger(__name__).info("Entry fills:\n%s", report.entry_fills.to_string())


if __name__ == "__main__":
    main()
