"""
Run (or resume) a journaled session on live IBKR bars.

Modes, from the connection settings:
- ``shadow`` (IBKR_READ_ONLY=true): live bars, fills simulated by the `SimulatedBroker`, no order sent;
- ``paper``: orders through `IBKRBroker` on a paper account (DU...);
- ``live``: a real-money account, only when the settings allow it (`IBKR_ALLOW_LIVE`, set by the
  entry script after an explicit confirmation).

Start: fit the strategies on recent IBKR history, snapshot them, then run. Resume: reload the snapshot
(same models), rebuild the feature buffers from fresh history, restore the runner state, recover the
IBKR orders and executions missed while down, catch up on the missed bars, check the positions and run.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from quantlab.execution.broker import Broker
from quantlab.execution.ibkr.bar_feed import IBKRBarFeed
from quantlab.execution.ibkr.broker import IBKRBroker
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.journal import Journal
from quantlab.runtime.runner import StrategyRunner
from quantlab.runtime.session import SessionFiles
from quantlab.runtime.state import StateStore
from quantlab.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveConfig:
    symbols: Sequence[str]
    bar_seconds: int = 60
    history_duration: str = "2 D"
    use_rth: bool = False
    capital: float = 100_000.0
    risk: RiskLimits = field(default_factory=RiskLimits)
    #: costs of the simulated fills (shadow mode, and the replay)
    fee_bps: float = 1.0
    slippage_bps: float = 1.0


def session_mode(client: IBKRClient) -> str:
    if client.settings.read_only:
        return "shadow"
    return "paper" if client.is_paper else "live"


@dataclass
class LiveSession:
    runner: StrategyRunner
    feed: IBKRBarFeed
    journal: Journal

    def run(self) -> StrategyRunner:
        """Blocks until `stop()` or Ctrl+C; the state is saved after every bar and on exit."""
        try:
            self.runner.run(self.feed)
        except KeyboardInterrupt:
            logger.info("Interrupted: state saved, resume with the same session name")
        finally:
            self.feed.stop()
            self.journal.event("session", {"action": "stop", "equity": self.runner.portfolio.equity})
            self.journal.close()
        portfolio = self.runner.portfolio
        logger.info("Equity %.2f, positions %s", portfolio.equity, portfolio.net_positions())
        return self.runner

    def stop(self) -> None:
        """Thread-safe: the run returns at the next heartbeat."""
        self.feed.stop()


def run_live_session(
    client: IBKRClient,
    make_strategies: Callable[[], Sequence[Strategy]],
    session_dir: str | Path,
    config: LiveConfig,
    resume: bool = False,
) -> StrategyRunner:
    return open_live_session(client, make_strategies, session_dir, config, resume).run()


def open_live_session(
    client: IBKRClient,
    make_strategies: Callable[[], Sequence[Strategy]],
    session_dir: str | Path,
    config: LiveConfig,
    resume: bool = False,
) -> LiveSession:
    """Prepare (or restore) a session up to the first live bar; `run()` it."""
    files = SessionFiles(session_dir)
    mode = session_mode(client)
    broker: Broker
    if mode == "shadow":
        broker = SimulatedBroker(fee_bps=config.fee_bps, slippage_bps=config.slippage_bps)
    else:
        client.ensure_account_allowed()
        broker = IBKRBroker(client)
    logger.info("Session %s: %s mode on %s", files.root, mode, ", ".join(config.symbols))

    feed = IBKRBarFeed(client, config.symbols, config.bar_seconds, use_rth=config.use_rth)
    history = feed.history(config.history_duration)
    if history.empty:
        raise RuntimeError("No IBKR history for the warm-up.")
    store = StateStore(files.state_path)
    state = None
    if resume:
        state = store.load()
        if state is None or not files.exists():
            raise FileNotFoundError(f"No session to resume in {files.root}.")
        if files.meta()["mode"] != mode:
            raise RuntimeError(f"Session started in {files.meta()['mode']} mode, now {mode}: refusing to resume.")
        strategies = files.load_strategies()
        for strategy in strategies:
            strategy.warm(history)  # same models, buffers rebuilt from fresh bars
    else:
        if files.exists():
            raise FileExistsError(f"Session {files.root} already exists: resume it or pick another name.")
        strategies = list(make_strategies())
        for strategy in strategies:
            strategy.fit(history)
        meta = {
            "mode": mode,
            "symbols": list(config.symbols),
            "bar_seconds": config.bar_seconds,
            "capital": config.capital,
            "risk": dataclasses.asdict(config.risk),
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "strategies": [s.strategy_id for s in strategies],
            "accounts_paper": client.is_paper,
        }
        files.save_snapshot(strategies, history, meta)

    journal = Journal(files.journal_path)
    runner = StrategyRunner(
        strategies, broker, RiskManager(config.risk), config.capital, journal=journal, state_store=store
    )
    journal.event("session", {"action": "resume" if resume else "start", "mode": mode, "warmup_bars": len(history)})
    try:
        if state is not None:
            runner.set_state(state)
            if isinstance(broker, IBKRBroker):
                broker.recover(runner.router.orders())
            runner.broker.process_events()
            if runner.last_bar_time is not None:
                runner.catch_up(history[history.index > runner.last_bar_time])
        if isinstance(broker, IBKRBroker):
            mismatches = broker.reconcile(runner.portfolio.net_positions())
            if mismatches:
                raise RuntimeError(
                    f"IBKR positions differ from the session book {{symbol: (ibkr, book)}}: {mismatches}"
                )
    except Exception:
        journal.close()
        raise
    return LiveSession(runner, feed, journal)
