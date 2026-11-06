"""Step 5: journal, persisted state / restart, replay, and a full shadow session on the fake IBKR client."""

from __future__ import annotations

import dataclasses
import pickle
import threading
from pathlib import Path

import pandas as pd
import pytest
from ibkr_fakes import connected, wait_until

from quantlab.analysis.replay import bar_differences, replay_session
from quantlab.execution.ibkr.broker import IBKRBroker
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.execution.ibkr.live import LiveConfig, open_live_session
from quantlab.execution.orders import Order, OrderType, Side
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.execution.router import Trade
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.journal import Journal, read_journal
from quantlab.runtime.runner import StrategyRunner
from quantlab.runtime.session import SessionFiles
from quantlab.runtime.state import StateStore
from quantlab.strategies.base import Strategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

FIXTURES = Path(__file__).parent / "fixtures"
RISK = RiskLimits()
EXITS = {"stop_loss_pct": 0.2, "take_profit_pct": 0.4}  # tight (percent): exercises the protective orders


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "aapl_1m.parquet").iloc[:1600]


def fitted(strategy: Strategy, history: pd.DataFrame) -> Strategy:
    strategy.fit(history)
    return strategy


def run(
    strategies: list[Strategy],
    bars: pd.DataFrame,
    journal: Journal | None = None,
    store: StateStore | None = None,
    slippage_bps: float = 1.0,
) -> StrategyRunner:
    runner = StrategyRunner(
        strategies, SimulatedBroker(slippage_bps=slippage_bps), RiskManager(RISK), journal=journal, state_store=store
    )
    runner.run(HistoricalBarFeed(bars))
    return runner


# --- journal -----------------------------------------------------------------------------------------------


def test_journal_records_every_bar_signal_fill_and_trade(tmp_path: Path, aapl: pd.DataFrame) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    runner = run([fitted(MeanReversionStrategy(**EXITS), aapl.iloc[:1000])], aapl.iloc[1000:], journal)
    journal.close()
    data = read_journal(tmp_path / "j.sqlite")

    assert len(data.bars) == 600 and not data.bars["catch_up"].any()
    assert data.bars.index[0] == pd.Timestamp(aapl.index[1000], tz="UTC")
    assert len(data.kind("equity")) == 600
    assert len(data.kind("trade")) == len(runner.router.trades) > 0
    signals, fills, orders = data.kind("signal"), data.kind("fill"), data.kind("order")
    assert len(signals) > 0 and set(signals["order_id"]) <= set(orders["order_id"])
    assert set(orders["order_type"]) == {"MKT", "STP", "LMT"}  # entries, protective pairs, time exits
    assert len(fills) == 2 * len(runner.router.trades) + len(runner.router.lots)
    features = data.kind("features")
    assert {"bb_upper_20_2", "rsi_14"} <= set(features.columns)


def test_cascade_predictions_are_journaled(tmp_path: Path, aapl: pd.DataFrame) -> None:
    journal = Journal(tmp_path / "j.sqlite")
    run([fitted(CascadeStrategy(**EXITS), aapl.iloc[:1400])], aapl.iloc[1400:1450], journal)
    journal.close()
    decisions = read_journal(tmp_path / "j.sqlite").kind("decision")
    assert len(decisions) == 50 and "opportunity" in decisions


# --- restart -----------------------------------------------------------------------------------------------


def trades_key(trades: list[Trade]) -> list[tuple[object, ...]]:
    return [(t.strategy_id, t.side, t.quantity, t.entry_time, t.exit_time, round(t.pnl_net, 9), t.reason)
            for t in trades]  # fmt: skip


def test_restart_from_saved_state_reproduces_the_uninterrupted_run(tmp_path: Path, aapl: pd.DataFrame) -> None:
    history, live = aapl.iloc[:1000], aapl.iloc[1000:]
    snapshot = pickle.dumps(
        [fitted(MeanReversionStrategy(**EXITS), history), fitted(TrendFollowingStrategy(**EXITS), history)]
    )

    reference = run(pickle.loads(snapshot), live)

    # Cut the session in the middle of a trade
    cut = live.index.get_loc(reference.router.trades[len(reference.router.trades) // 2].entry_time) + 1
    store = StateStore(tmp_path / "state.json")
    first = run(pickle.loads(snapshot), live.iloc[:cut], store=store)
    assert first.router.lots

    # Restart: same models, buffers rebuilt from the bars seen so far, state restored
    strategies = pickle.loads(snapshot)
    for strategy in strategies:
        strategy.warm(pd.concat([history, live.iloc[:cut]]))
    second = StrategyRunner(strategies, SimulatedBroker(slippage_bps=1.0), RiskManager(RISK))
    state = store.load()
    assert state is not None
    second.set_state(state)
    second.run(HistoricalBarFeed(live.iloc[cut:]))

    assert trades_key(first.router.trades + second.router.trades) == trades_key(reference.router.trades)
    assert second.portfolio.equity == pytest.approx(reference.portfolio.equity, abs=1e-6)


def test_catch_up_advances_open_lots_without_deciding(aapl: pd.DataFrame) -> None:
    strategy = fitted(MeanReversionStrategy(**EXITS), aapl.iloc[:1000])
    runner = run([strategy], aapl.iloc[1000:1100])
    open_lots = {lot.lot_id: lot.bars_held for lot in runner.router.lots.values()}
    runner.catch_up(aapl.iloc[1100:1103])
    assert runner.last_bar_time == aapl.index[1102]
    for lot_id, held in open_lots.items():
        if lot_id in runner.router.lots:
            assert runner.router.lots[lot_id].bars_held == held + 3


# --- replay ------------------------------------------------------------------------------------------------


def make_session(root: Path, history: pd.DataFrame, live: pd.DataFrame, live_slippage_bps: float) -> StrategyRunner:
    """A 'live' session whose fills are simulated with `live_slippage_bps` (stands for real fills)."""
    files = SessionFiles(root)
    strategies: list[Strategy] = [fitted(MeanReversionStrategy(**EXITS), history)]
    meta = {"mode": "shadow", "capital": 100_000.0, "risk": dataclasses.asdict(RISK), "fee_bps": 1.0,
            "slippage_bps": 1.0}  # fmt: skip
    files.save_snapshot(strategies, history, meta)
    journal = Journal(files.journal_path)
    runner = run(strategies, live, journal, slippage_bps=live_slippage_bps)
    journal.close()
    return runner


def test_replay_of_a_faithful_session_matches_exactly(tmp_path: Path, aapl: pd.DataFrame) -> None:
    live_runner = make_session(tmp_path, aapl.iloc[:1000], aapl.iloc[1000:], live_slippage_bps=1.0)
    report = replay_session(tmp_path)
    s = report.summary
    assert s["signal_agreement_pct"] == 100.0 and s["signals_live"] == s["signals_replay"] > 0
    assert s["max_feature_diff"] == 0.0
    assert s["entry_slippage_bps_mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["trades_live"] == s["trades_replay"] == len(live_runner.router.trades)
    assert s["pnl_net_live"] == pytest.approx(s["pnl_net_replay"])


def test_replay_measures_live_slippage(tmp_path: Path, aapl: pd.DataFrame) -> None:
    make_session(tmp_path, aapl.iloc[:1000], aapl.iloc[1000:], live_slippage_bps=6.0)
    report = replay_session(tmp_path)  # replay assumes the session's 1 bp
    assert report.summary["entry_slippage_bps_mean"] == pytest.approx(5.0, abs=0.01)
    assert report.summary["pnl_net_live"] < report.summary["pnl_net_replay"]


def test_bar_differences() -> None:
    index = pd.date_range("2026-01-05 15:00", periods=3, freq="min", tz="UTC")
    live = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 10.0, 10.0]}, index=index)
    reference = live.copy()
    reference.iloc[1, 1] = 12.0  # volume corrected afterwards
    assert list(bar_differences(live, reference).index) == [index[1]]


# --- full shadow session on the fake IBKR client -----------------------------------------------------------

T0 = 1_767_625_800


def push_minute(api: object, minute: int, close: float) -> None:
    for i in range(12):
        api.realtime_bar("AAPL", T0 + 60 * minute + 5 * i, close)  # type: ignore[attr-defined]


class AlwaysBuy(Strategy):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol=symbol, stop_loss_pct=None, take_profit_pct=None, holding_bars=2)

    def decide(self, bar: pd.Series) -> list[Order]:
        return [self.order(Side.BUY, 1)]


def open_session(client: IBKRClient, root: Path, resume: bool = False) -> tuple[object, threading.Thread]:
    session = open_live_session(
        client, lambda: [AlwaysBuy(symbol="AAPL")], root,
        LiveConfig(symbols=["AAPL"], risk=RiskLimits(require_stop=False)), resume=resume,
    )  # fmt: skip
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    return session, thread


def test_shadow_session_journals_and_resumes(tmp_path: Path) -> None:
    client, api = connected()  # read-only: shadow mode
    api.history = {"AAPL": [(T0 - 60 * k, 100.0) for k in range(30, -1, -1)]}
    session, thread = open_session(client, tmp_path)
    wait_until(lambda: api.realtime)
    for minute in range(3):
        push_minute(api, minute, 100.0 + minute)
    wait_until(lambda: len(read_journal(tmp_path / "journal.sqlite").bars) == 3)
    session.stop()  # type: ignore[attr-defined]
    thread.join(3)
    assert api.placed == []  # shadow: nothing sent

    state = StateStore(tmp_path / "state.json").load()
    # held: a lot whose time exit is pending, a lot still held, and an entry not filled yet
    assert state is not None and len(state["router"]["lots"]) == 3
    # Restart two minutes later: the missed minutes come from the history and are caught up
    api.history = {"AAPL": [(T0 + 60 * k, 100.0 + k) for k in range(-30, 6)]}
    session, thread = open_session(client, tmp_path, resume=True)
    wait_until(lambda: len(api.realtime) == 1)
    push_minute(api, 5, 106.0)
    wait_until(lambda: len(read_journal(tmp_path / "journal.sqlite").bars) == 6)
    session.stop()  # type: ignore[attr-defined]
    thread.join(3)

    journal = read_journal(tmp_path / "journal.sqlite")
    assert list(journal.bars["catch_up"]) == [False, False, False, True, True, False]
    assert [e["action"] for e in journal.kind("session").to_dict("records")] == ["start", "stop", "resume", "stop"]
    assert len(journal.kind("trade")) >= 2  # lots of the first run closed after the restart
    assert api.placed == []


# --- IBKR broker restart -----------------------------------------------------------------------------------


def test_ibkr_broker_recovers_missed_executions_and_ended_orders() -> None:
    client, api = connected(read_only=False)
    broker = IBKRBroker(client)
    entry = Order("AAPL", Side.BUY, 10, "s")
    stop = Order("AAPL", Side.SELL, 10, "s", OrderType.STOP, stop_price=95.0)
    target = Order("AAPL", Side.SELL, 10, "s", OrderType.LIMIT, limit_price=110.0)
    for order in (entry, stop, target):
        broker.submit(order)
    state = broker.get_state()
    entry_id, stop_id = api.placed[0][0], api.placed[1][0]

    # While down: the stop filled (the target was cancelled by OCA); the entry had filled before
    api.execute(entry_id, 10, 100.0, "e-entry", silent=True)
    api.execute(stop_id, 10, 95.0, "e-stop", silent=True)
    restarted_client, restarted_api = connected(read_only=False)
    restarted_api.placed, restarted_api.past_executions = api.placed, api.past_executions
    restarted = IBKRBroker(restarted_client)
    # the entry fill had been processed before the crash
    restarted.set_state({**state, "seen_execs": ["e-entry"], "executed": {str(entry_id): 10}})
    fills, statuses = [], []
    restarted.set_handlers(fills.append, lambda oid, status, msg: statuses.append((oid, status)))
    restarted.recover([entry, stop, target], timeout=2)
    restarted.process_events()

    assert [(f.order.order_id, f.quantity, f.price) for f in fills] == [(stop.order_id, 10, 95.0)]
    assert [oid for oid, _ in statuses] == [target.order_id]  # no longer working, never executed


@pytest.mark.slow
def test_every_strategy_survives_the_session_snapshot(aapl: pd.DataFrame) -> None:
    """A session snapshot pickles the fitted strategies: they must reload and decide identically."""
    from quantlab.strategies.arima import ArimaStrategy
    from quantlab.strategies.ml_hybrid import MLHybridStrategy

    for factory in (TrendFollowingStrategy, MeanReversionStrategy, MLHybridStrategy, CascadeStrategy, ArimaStrategy):
        strategy = fitted(factory(), aapl.iloc[:1400])
        clone = pickle.loads(pickle.dumps(strategy))
        for s in (strategy, clone):
            s.start()
        for _, bar in aapl.iloc[1400:1420].iterrows():
            # order ids are random: compare the decisions
            decided = [(o.symbol, o.side, o.quantity) for o in strategy.on_bar(bar)]
            assert decided == [(o.symbol, o.side, o.quantity) for o in clone.on_bar(bar)]
            assert strategy.diagnostics == clone.diagnostics
        assert strategy.feed.row().equals(clone.feed.row())
