"""
quantlab command line: one entry point for research and trading.

    quantlab backtest --strategy trend --symbol AAPL --source yfinance --days 55 --test-days 5
    quantlab backtest --strategy cascade --strategy trend --symbol SPY --source massive --days 180 --each
    quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --param zscore_threshold=2.0
    quantlab optimize --strategy trend --symbol AAPL --grid sma_fast=5,10,20 --grid sma_slow=30,50,100 \\
                      --train-size 3000 --validation-size 1000 --test-size 1000 --embargo 30
    quantlab paper --strategy trend --symbol AAPL --session trend_aapl [--resume] [--live]
    quantlab replay --session trend_aapl
    quantlab plot --symbol SPY --source ibkr --days 3
    quantlab ui                                     (graphical interface in the browser)

Data sources: yfinance (2-minute bars, ~60 days), massive (1-minute, MASSIVE_API_KEY), ibkr (1-minute,
needs TWS / IB Gateway), file (--file bars.parquet|csv). IBKR connection settings come from IBKR_* (see
.env.example); `paper` runs in shadow mode (no order sent) while IBKR_READ_ONLY=true.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data import loading
from quantlab.data.loading import SOURCES, ibkr_client, split
from quantlab.execution.risk import RiskLimits
from quantlab.strategies.base import Strategy
from quantlab.strategies.registry import STRATEGIES, build_strategy

logger = logging.getLogger("quantlab")

__all__ = ["SOURCES", "ibkr_client", "main", "parse_grid", "parse_params", "parse_value", "split"]


# --- argument helpers --------------------------------------------------------------------------------------


def parse_value(text: str) -> Any:
    """'10' -> 10, '2.5' -> 2.5, 'true' -> True, anything else stays a string."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_params(items: Sequence[str], strategies: Sequence[str]) -> dict[str, dict[str, Any]]:
    """`--param key=value` applies to every strategy; `--param name.key=value` to one."""
    params: dict[str, dict[str, Any]] = {name: {} for name in strategies}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--param expects key=value, got {item!r}")
        target, dot, name = key.rpartition(".")
        if dot and target not in params:
            raise argparse.ArgumentTypeError(f"--param {item!r}: {target!r} is not a selected strategy")
        for strategy in [target] if dot else list(params):
            params[strategy][name] = parse_value(value)
    return params


def parse_grid(items: Sequence[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not values:
            raise argparse.ArgumentTypeError(f"--grid expects key=v1,v2,..., got {item!r}")
        grid[key] = [parse_value(v) for v in values.split(",")]
    return grid


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    try:
        return loading.load_bars(args.symbol, args.source, args.days, args.file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_strategies(args: argparse.Namespace) -> list[Strategy]:
    params = parse_params(args.param, args.strategy)
    return [build_strategy(name, args.symbol, params[name]) for name in args.strategy]


def risk_limits(args: argparse.Namespace) -> RiskLimits:
    def pct(value: float | None) -> float | None:
        return None if value is None or value <= 0 else value / 100

    return RiskLimits(
        stop_loss_pct=pct(args.stop_loss),
        take_profit_pct=pct(args.take_profit),
        max_order_quantity=args.max_order,
        max_position_quantity=args.max_position,
        max_daily_loss_pct=pct(args.max_daily_loss),
    )


# --- commands ----------------------------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantlab.backtest.run import backtest

    data = load_bars(args)
    train, test = split(data, args.test_days, args.test_fraction)
    logger.info("Training %s to %s (%d bars)", train.index[0], train.index[-1], len(train))
    logger.info("Test     %s to %s (%d bars)", test.index[0], test.index[-1], len(test))

    runs: list[list[str]] = [[name] for name in args.strategy] if args.each and len(args.strategy) > 1 else []
    runs.append(list(args.strategy))
    rows = {}
    for names in runs:
        sub = argparse.Namespace(**{**vars(args), "strategy": names})
        result = backtest(
            build_strategies(sub),
            train,
            test,
            capital=args.capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            risk_limits=risk_limits(args),
            precompute_features=not args.bar_by_bar,
        )
        label = "+".join(names)
        rows[label] = result.metrics
        if len(names) == len(args.strategy):
            if args.trades:
                result.trades_frame().to_csv(args.trades, index=False)
                logger.info("Trades written to %s", args.trades)
            if args.equity:
                result.equity.to_csv(args.equity)
                logger.info("Equity curve written to %s", args.equity)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.4f}"))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from quantlab.backtest.walk_forward import WalkForwardOptimizer

    if len(args.strategy) != 1:
        raise SystemExit("optimize takes exactly one --strategy")
    name = args.strategy[0]
    fixed = parse_params(args.param, [name])[name]
    data = load_bars(args)
    optimizer = WalkForwardOptimizer(
        strategy_factory=lambda **p: build_strategy(name, args.symbol, {**fixed, **p}),
        param_grid=parse_grid(args.grid),
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step=args.step,
        embargo=args.embargo,
        anchored=args.anchored,
        objective=args.objective,
        min_trades=args.min_trades,
    )
    result = optimizer.run(data)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.summary().to_string())
    print("\nStability of the chosen parameters:")
    print(result.param_stability().to_string())
    print(f"\nMean out-of-sample {result.objective}: {result.mean_test_score:.4f}")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from quantlab.execution.ibkr.live import (
        LiveConfig,
        authorize_live_account,
        needs_live_confirmation,
        run_live_session,
    )

    if len(args.strategy) != 1:
        raise SystemExit("paper takes exactly one --strategy")
    name = args.strategy[0]
    spec = STRATEGIES[name]
    with ibkr_client(auto_reconnect=True) as client:
        if needs_live_confirmation(client):
            if not args.live:
                raise SystemExit(f"{', '.join(client.accounts)} is a real-money account: pass --live to trade it.")
            typed = input(f"REAL MONEY on {', '.join(client.accounts)}. Type the account id to confirm: ")
            try:
                authorize_live_account(client, typed)
            except PermissionError as exc:
                raise SystemExit(str(exc)) from exc
        session = args.session or f"{name}_{'_'.join(args.symbol)}_{date.today():%Y%m%d}"
        params = parse_params(args.param, [name])[name]
        config = LiveConfig(
            symbols=list(args.symbol),
            bar_seconds=args.bar_seconds,
            history_duration=args.history or spec.live_history,
            use_rth=args.rth,
            capital=args.capital,
            risk=risk_limits(args),
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        run_live_session(
            client, lambda: [spec.build(args.symbol, params)], args.sessions_dir / session, config, args.resume
        )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from quantlab.analysis.replay import replay_session

    report = replay_session(args.sessions_dir / args.session, slippage_bps=args.slippage_bps)
    print(pd.Series(report.summary).to_string(float_format=lambda v: f"{v:,.4f}"))
    mismatched = report.signals[report.signals["match"] != "both"]
    if not mismatched.empty:
        print("\nSignals not matched:\n" + mismatched.to_string())
    if not report.entry_fills.empty:
        print("\nEntry fills:\n" + report.entry_fills.to_string())
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from quantlab.analysis.plot import plot_levels, session_bars

    try:
        session, day = session_bars(load_bars(args), args.date)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    plot_levels(session, f"{args.symbol[0]} {day} (New York time)", output=args.output)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    return subprocess.call(command)


# --- parser ------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    commands = parser.add_subparsers(dest="command", required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--symbol", action="append", required=True, help="repeat for multi-symbol strategies")
    data.add_argument("--source", choices=SOURCES, default="yfinance")
    data.add_argument("--file", help="bars file for --source file (parquet or csv)")
    data.add_argument("--days", type=int, default=55, help="history window in calendar days")

    strategy = argparse.ArgumentParser(add_help=False)
    strategy.add_argument("--strategy", action="append", required=True, choices=sorted(STRATEGIES))
    strategy.add_argument("--param", action="append", default=[], help="key=value or strategy.key=value")

    execution = argparse.ArgumentParser(add_help=False)
    execution.add_argument("--capital", type=float, default=100_000.0)
    execution.add_argument("--fee-bps", type=float, default=1.0)
    execution.add_argument("--slippage-bps", type=float, default=1.0, help="simulated slippage")
    execution.add_argument("--stop-loss", type=float, default=2.0, help="percent, 0 = none")
    execution.add_argument("--take-profit", type=float, default=5.0, help="percent, 0 = none")
    execution.add_argument("--max-order", type=int, help="max shares per entry order")
    execution.add_argument("--max-position", type=int, help="max absolute shares per symbol")
    execution.add_argument("--max-daily-loss", type=float, help="kill switch, percent of the day's equity")

    sessions = argparse.ArgumentParser(add_help=False)
    sessions.add_argument("--sessions-dir", type=Path, default=Path("sessions"))

    p = commands.add_parser("backtest", parents=[data, strategy, execution], help="backtest strategies")
    p.add_argument("--test-days", type=float, default=5, help="last N calendar days are the test block")
    p.add_argument("--test-fraction", type=float, help="or: the last fraction of the bars")
    p.add_argument("--each", action="store_true", help="with several strategies: also run each alone")
    p.add_argument("--bar-by-bar", action="store_true", help="live feature path instead of the batch mode")
    p.add_argument("--trades", help="write the trades to this CSV")
    p.add_argument("--equity", help="write the equity curve to this CSV")
    p.set_defaults(func=cmd_backtest)

    p = commands.add_parser("optimize", parents=[data, strategy], help="walk-forward hyperparameter search")
    p.add_argument("--grid", action="append", required=True, help="key=v1,v2,...")
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--validation-size", type=int, required=True)
    p.add_argument("--test-size", type=int, required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--embargo", type=int, default=30)
    p.add_argument("--anchored", action="store_true")
    p.add_argument("--objective", default="total_return_pct")
    p.add_argument("--min-trades", type=int, default=5)
    p.set_defaults(func=cmd_optimize)

    p = commands.add_parser("paper", parents=[strategy, execution, sessions], help="journaled IBKR session")
    p.add_argument("--symbol", action="append", required=True)
    p.add_argument("--session", help="name (default: <strategy>_<symbols>_<date>)")
    p.add_argument("--resume", action="store_true", help="resume an interrupted session")
    p.add_argument("--live", action="store_true", help="allow a real-money account (asks for confirmation)")
    p.add_argument("--bar-seconds", type=int, default=60)
    p.add_argument("--history", help="IBKR warm-up history, e.g. '2 D' (default per strategy)")
    p.add_argument("--rth", action="store_true", help="regular trading hours only")
    p.set_defaults(func=cmd_paper, max_order=10, max_position=50, max_daily_loss=1.0)

    p = commands.add_parser("replay", parents=[sessions], help="replay a session in the backtester")
    p.add_argument("--session", required=True)
    p.add_argument("--slippage-bps", type=float, help="simulated slippage (default: the session's)")
    p.set_defaults(func=cmd_replay)

    p = commands.add_parser("plot", parents=[data], help="chart one session with VWAP and support/resistance")
    p.add_argument("--date", help="session date (default: the last one)")
    p.add_argument("--output", help="save to this image file instead of showing")
    p.set_defaults(func=cmd_plot)

    p = commands.add_parser("ui", help="graphical interface in the browser (Streamlit)")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("ibapi").setLevel(logging.WARNING)
    try:
        code: int = args.func(args)
    except ConnectionError as exc:
        logger.error("%s", exc)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
