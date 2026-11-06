"""Streamlit UI: every page renders, and the offline actions work end to end (AppTest, no browser, no TWS)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import altair as alt
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from quantlab.execution.risk import RiskLimits
from quantlab.execution.simulated import SimulatedBroker
from quantlab.runtime.bar_feed import HistoricalBarFeed
from quantlab.runtime.journal import Journal
from quantlab.runtime.runner import StrategyRunner
from quantlab.runtime.session import SessionFiles
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.ui import charts

FIXTURES = Path(__file__).parent / "fixtures"


def page_app(page: str, sessions_dir: str = "sessions") -> AppTest:
    def script(page: str, sessions_dir: str) -> None:
        from pathlib import Path

        import quantlab.ui.app as app

        app.SESSIONS_DIR = Path(sessions_dir)
        getattr(app, page)()

    return AppTest.from_function(script, args=(page, sessions_dir), default_timeout=120)


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", ["page_backtest", "page_optimize", "page_live", "page_sessions", "page_chart"])
def test_every_page_renders(page: str, tmp_path: Path) -> None:
    at = page_app(page, str(tmp_path)).run()
    no_errors(at)


def test_backtest_page_runs_on_a_file(tmp_path: Path) -> None:
    at = page_app("page_backtest").run()
    at.multiselect[0].set_value(["trend", "mean_reversion"]).run()
    at.selectbox(key="bt_source").set_value("file").run()
    at.text_input(key="bt_path").set_value(str(FIXTURES / "aapl_1m.parquet")).run()
    at.radio(key="bt_split").set_value("Dernière fraction").run()
    at.button[0].click().run()
    no_errors(at)
    labels = [m.label for m in at.metric]
    assert {"Rendement", "Trades", "Sharpe"} <= set(labels)
    assert len(at.dataframe) >= 2  # comparison of the runs + trades


def test_strategy_parameters_are_introspected() -> None:
    from quantlab.ui.app import parse_extra, strategy_signature

    assert strategy_signature("trend") == {"sma_fast": 5, "sma_slow": 20}
    assert strategy_signature("arima")["order"] == (2, 1, 0)
    assert "zscore_threshold" in strategy_signature("pair")
    assert parse_extra("sma_fast=10\n\nk=1.5") == {"sma_fast": 10, "k": 1.5}
    with pytest.raises(ValueError):
        parse_extra("oops")


def test_sessions_page_shows_journal_and_replays(tmp_path: Path) -> None:
    data = pd.read_parquet(FIXTURES / "aapl_1m.parquet")
    root = tmp_path / "demo"
    strategy = MeanReversionStrategy()
    strategy.fit(data.iloc[:1000])
    meta = {"mode": "shadow", "symbols": ["AAPL"], "strategies": [strategy.strategy_id], "capital": 100_000.0,
            "risk": dataclasses.asdict(RiskLimits()), "fee_bps": 1.0, "slippage_bps": 1.0}  # fmt: skip
    files = SessionFiles(root)
    files.save_snapshot([strategy], data.iloc[:1000], meta)
    journal = Journal(files.journal_path)
    StrategyRunner([strategy], SimulatedBroker(), journal=journal).run(HistoricalBarFeed(data.iloc[1000:1400]))
    journal.close()

    at = page_app("page_sessions", str(tmp_path)).run()
    no_errors(at)
    assert at.selectbox[0].value == "demo"
    assert any(m.label == "Barres" and m.value == "400" for m in at.metric)
    next(b for b in at.button if b.label == "Lancer le replay").click().run()
    no_errors(at)
    assert any(m.label == "Accord des signaux" and m.value == "100.0 %" for m in at.metric)


def test_chart_page_draws_a_session() -> None:
    at = page_app("page_chart").run()
    at.selectbox(key="chart_source").set_value("file").run()
    at.text_input(key="chart_path").set_value(str(FIXTURES / "aapl_1m.parquet")).run()
    next(b for b in at.button if b.label == "Tracer").click().run()
    no_errors(at)
    assert at.get("imgs") or at.get("download_button")


def test_charts_build_valid_specs() -> None:
    index = pd.date_range("2026-01-05 14:30", periods=10, freq="min", tz="UTC")
    equity = pd.Series(range(100, 110), index=index, dtype=float)
    trades = pd.DataFrame({"pnl_net": [1.0, -2.0], "exit_time": index[:2], "strategy_id": "s", "symbol": "X",
                           "side": "BUY", "reason": "time"})  # fmt: skip
    for chart in (
        charts.equity_chart({"a": equity, "b": equity * 1.01}),
        charts.drawdown_chart(equity),
        charts.trade_pnl_chart(trades),
        charts.value_bars(pd.Series([1.0, 2.0], index=["x", "y"]), "v", "k"),
        charts.time_line(pd.Series(index), equity.reset_index(drop=True), "v"),
    ):
        assert isinstance(chart, alt.TopLevelMixin)
        chart.to_dict()
    # New York wall-clock times, naive
    assert charts.exchange_time(index)[0] == pd.Timestamp("2026-01-05 09:30")


def test_live_sessions_run_in_the_background_and_stop(tmp_path: Path) -> None:
    from ibkr_fakes import FakeEClient, wait_until

    from quantlab.config import IBKRSettings
    from quantlab.execution.ibkr.client import IBKRClient
    from quantlab.execution.ibkr.live import LiveConfig
    from quantlab.ui.live_runner import LiveSessions

    t0 = 1_767_625_800
    clients: list[IBKRClient] = []

    def factory() -> IBKRClient:  # read-only fake TWS: shadow mode
        client = IBKRClient(IBKRSettings(read_only=True), api_factory=FakeEClient)
        client.api.history = {"AAPL": [(t0 - 60 * k, 100.0) for k in range(40, -1, -1)]}
        clients.append(client)
        return client

    registry = LiveSessions(client_factory=factory)
    entry = registry.start("demo", tmp_path / "demo", "trend", ["AAPL"], {}, LiveConfig(symbols=["AAPL"]))
    wait_until(lambda: entry.status == "en cours" and clients and clients[0].api.realtime)
    api = clients[0].api
    for minute in range(2):
        for i in range(12):
            api.realtime_bar("AAPL", t0 + 60 * minute + 5 * i, 100.0 + minute)
    wait_until(lambda: (tmp_path / "demo" / "journal.sqlite").exists())
    with pytest.raises(RuntimeError):  # one run per session name
        registry.start("demo", tmp_path / "demo", "trend", ["AAPL"], {}, LiveConfig(symbols=["AAPL"]))
    entry.stop()
    wait_until(lambda: not entry.alive)
    assert (entry.status, entry.mode, entry.error) == ("arrêtée", "shadow", None)
    assert api.placed == []


def test_live_session_refuses_an_unconfirmed_real_account(tmp_path: Path) -> None:
    from ibkr_fakes import FakeEClient, wait_until

    from quantlab.config import IBKRSettings
    from quantlab.execution.ibkr.client import IBKRClient
    from quantlab.execution.ibkr.live import LiveConfig
    from quantlab.ui.live_runner import LiveSessions

    def factory() -> IBKRClient:
        client = IBKRClient(IBKRSettings(read_only=False), api_factory=FakeEClient)
        client.api.accounts = "U1234567"
        return client

    registry = LiveSessions(client_factory=factory)
    entry = registry.start("real", tmp_path / "real", "trend", ["AAPL"], {}, LiveConfig(symbols=["AAPL"]),
                           allow_live=True, confirm_account="wrong")  # fmt: skip
    wait_until(lambda: not entry.alive)
    assert entry.status == "erreur" and "not confirmed" in (entry.error or "")
