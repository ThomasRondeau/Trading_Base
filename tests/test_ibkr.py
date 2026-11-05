"""
IBKR client, broker and bar feed against a fake EClient: no TWS, no socket, no order can leave the process.

`FakeEClient` answers requests by calling the wrapper callbacks, like TWS would from the network thread.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
import pytest
from ibapi.client import EClient
from ibapi.order import Order as IBOrder
from ibkr_fakes import FakeEClient, connected, make_client, wait_until

from quantlab.config import IBKRSettings
from quantlab.execution.ibkr.bar_feed import BarAggregator, IBKRBarFeed
from quantlab.execution.ibkr.broker import IBKRBroker, to_ib_order
from quantlab.execution.ibkr.client import (
    GuardedEClient,
    IBKRClient,
    IBKRRequestError,
    LiveAccountViolation,
    ReadOnlyViolation,
    stock,
)
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side
from quantlab.execution.risk import RiskLimits, RiskManager
from quantlab.runtime.runner import StrategyRunner
from quantlab.strategies.base import Strategy

# --- safety ------------------------------------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What would reach the real socket (the EClient methods are replaced by recorders)."""
    calls: list[str] = []
    for name in ("placeOrder", "cancelOrder", "reqGlobalCancel"):
        monkeypatch.setattr(EClient, name, lambda self, *a, _n=name, **k: calls.append(_n))
    return calls


def test_read_only_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("IBKR_READ_ONLY", "IBKR_ALLOW_LIVE"):
        monkeypatch.delenv(name, raising=False)
    settings = IBKRSettings.from_env()
    assert settings.read_only is True and settings.allow_live_account is False


@pytest.mark.parametrize(
    "call",
    [
        lambda api: api.placeOrder(1, stock("AAPL"), IBOrder()),
        lambda api: api.cancelOrder(1, None),
        lambda api: api.reqGlobalCancel(),
    ],
)
def test_guarded_eclient_blocks_order_calls_when_read_only(call: Any, sent: list[str]) -> None:
    api = GuardedEClient(IBKRClient(IBKRSettings(read_only=True), api_factory=FakeEClient), IBKRSettings())
    with pytest.raises(ReadOnlyViolation):
        call(api)
    assert sent == []


def test_guarded_eclient_passes_orders_only_when_allowed(sent: list[str]) -> None:
    api = GuardedEClient(IBKRClient(api_factory=FakeEClient), IBKRSettings(read_only=False))
    api.placeOrder(1, stock("AAPL"), IBOrder())
    assert sent == ["placeOrder"]


def test_client_refuses_orders_when_read_only() -> None:
    client, api = connected(read_only=True)
    with pytest.raises(ReadOnlyViolation):
        client.place_order(stock("AAPL"), IBOrder())
    with pytest.raises(ReadOnlyViolation):
        IBKRBroker(client)
    assert api.placed == []


def test_client_refuses_orders_on_a_real_money_account() -> None:
    client, api = make_client(read_only=False)
    api.accounts = "U7654321"
    client.connect(timeout=2)
    assert not client.is_paper
    with pytest.raises(LiveAccountViolation):
        client.place_order(stock("AAPL"), IBOrder())
    assert api.placed == []

    client, api = make_client(read_only=False, allow_live=True)
    api.accounts = "U7654321"
    client.connect(timeout=2)
    client.place_order(stock("AAPL"), IBOrder())
    assert len(api.placed) == 1


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_PORT", "7496")
    monkeypatch.setenv("IBKR_READ_ONLY", "false")
    monkeypatch.setenv("IBKR_ALLOW_LIVE", "true")
    settings = IBKRSettings.from_env()
    assert (settings.port, settings.read_only, settings.allow_live_account) == (7496, False, True)
    monkeypatch.setenv("IBKR_READ_ONLY", "maybe")
    with pytest.raises(ValueError):
        IBKRSettings.from_env()


# --- client ------------------------------------------------------------------------------------------------


def test_connect_failure_raises_a_helpful_error() -> None:
    client, api = make_client()
    api.refuse_connections = True
    with pytest.raises(ConnectionError, match="Enable ActiveX"):
        client.connect(timeout=0.2)


def test_historical_bars_are_validated_utc() -> None:
    client, api = connected()
    api.history["AAPL"] = [(1_767_625_800 + 60 * i, 100.0 + i) for i in range(3)]
    data = client.historical_bars(stock("AAPL"))
    assert list(data["close"]) == [100.0, 101.0, 102.0]
    assert str(data.index.tz) == "UTC" and data.index[0] == pd.Timestamp(1_767_625_800, unit="s", tz="UTC")


def test_historical_bars_error_and_timeout() -> None:
    client, api = connected()
    api.history_error = (162, "Historical Market Data Service error message: no data")
    with pytest.raises(IBKRRequestError) as info:
        client.historical_bars(stock("AAPL"))
    assert info.value.code == 162
    api.history_error, api.answer_history = None, False
    with pytest.raises(TimeoutError):
        client.historical_bars(stock("AAPL"), timeout=0.1)


def test_positions_are_summed_per_symbol() -> None:
    client, api = connected()
    api.positions = [("AAPL", 10), ("AAPL", -4), ("MSFT", 0)]
    assert client.positions() == {"AAPL": 6}


def test_order_ids_are_unique_across_threads() -> None:
    client, _ = connected()
    ids: list[int] = []
    threads = [
        threading.Thread(target=lambda: ids.extend(client.next_order_id() for _ in range(200))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 800 and min(ids) == 100


def test_reconnects_and_restores_subscriptions() -> None:
    client, api = connected(reconnect_delays=(0.01,))
    reconnected = threading.Event()
    client.reconnect_listeners.append(reconnected.set)
    client.subscribe_realtime_bars(stock("AAPL"), lambda bar: None)
    assert api.realtime_requests == 1

    api.drop()
    assert not client.is_connected
    assert reconnected.wait(3)
    assert client.is_connected and api.connect_count == 2
    assert api.realtime_requests == 2  # re-subscribed


def test_no_reconnect_after_an_intentional_disconnect() -> None:
    client, api = connected(reconnect_delays=(0.01,))
    client.disconnect()
    api.drop()
    time.sleep(0.1)
    assert api.connect_count == 1


# --- broker ------------------------------------------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.fills: list[Fill] = []
        self.statuses: list[tuple[str, OrderStatus]] = []

    def on_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

    def on_status(self, order_id: str, status: OrderStatus, message: str) -> None:
        self.statuses.append((order_id, status))


def paper_broker(**kwargs: Any) -> tuple[IBKRBroker, FakeEClient, Recorder]:
    client, api = connected(read_only=False)
    broker, rec = IBKRBroker(client, **kwargs), Recorder()
    broker.set_handlers(rec.on_fill, rec.on_status)
    return broker, api, rec


def test_to_ib_order_maps_types_oca_and_reference() -> None:
    stop = Order("AAPL", Side.SELL, 5, "s", OrderType.STOP, stop_price=98.5, oca_group="g1")
    ib = to_ib_order(stop)
    assert (ib.action, ib.totalQuantity, ib.orderType, ib.auxPrice) == ("SELL", 5, "STP", 98.5)
    assert (ib.ocaGroup, ib.ocaType, ib.tif, ib.orderRef) == ("g1", 2, "GTC", stop.order_id)
    market = to_ib_order(Order("AAPL", Side.BUY, 1, "s"))
    assert (market.orderType, market.tif) == ("MKT", "DAY")


def test_fills_are_delivered_on_the_strategy_thread_with_commission() -> None:
    broker, api, rec = paper_broker()
    order = Order("AAPL", Side.BUY, 10, "s")
    broker.submit(order)
    (ib_id, contract, _), = api.placed  # fmt: skip
    assert contract.symbol == "AAPL"

    api.execute(ib_id, 4, 100.0, "e1", commission=0.5)
    api.execute(ib_id, 6, 100.2, "e2", commission=0.7)
    api.execute(ib_id, 4, 100.0, "e1", commission=0.5)  # duplicate (e.g. re-requested after a reconnect)
    assert rec.fills == []  # nothing until the runner thread asks
    broker.process_events()
    assert [(f.quantity, f.price, f.fees, f.order) for f in rec.fills] == [
        (4, 100.0, 0.5, order),
        (6, 100.2, 0.7, order),
    ]


def test_fill_is_emitted_without_commission_after_a_delay() -> None:
    broker, api, rec = paper_broker(commission_wait=0.05)
    broker.submit(Order("AAPL", Side.BUY, 1, "s"))
    api.execute(api.placed[0][0], 1, 100.0, "e1", commission=None)

    def delivered() -> bool:
        broker.process_events()
        return len(rec.fills) == 1

    wait_until(delivered)
    assert rec.fills[0].fees == 0.0


def test_cancel_and_reject_statuses() -> None:
    broker, api, rec = paper_broker()
    a, b = Order("AAPL", Side.BUY, 1, "s"), Order("AAPL", Side.BUY, 1, "s")
    broker.submit(a)
    broker.submit(b)
    broker.cancel(a.order_id)
    assert api.cancelled == [api.placed[0][0]]
    broker.on_order_status(api.placed[0][0], "Cancelled", 0, 1)
    broker.on_order_status(api.placed[0][0], "Cancelled", 0, 1)  # repeated by TWS: reported once
    broker.on_order_error(api.placed[1][0], 201, "Order rejected - reason: no trading permissions")
    broker.process_events()
    assert rec.statuses == [(a.order_id, OrderStatus.CANCELLED), (b.order_id, OrderStatus.REJECTED)]


def test_reconnect_re_requests_executions_and_reconcile_reports_mismatches() -> None:
    broker, api, _ = paper_broker()
    broker.client.reconnect_delays = (0.01,)
    api.drop()
    wait_until(lambda: api.executions_requested == 1)
    api.positions = [("AAPL", 10)]
    assert broker.reconcile({"AAPL": 10}) == {}
    assert broker.reconcile({"MSFT": 5}) == {"AAPL": (10, 0), "MSFT": (0, 5)}


# --- bar feed ----------------------------------------------------------------------------------------------


def test_aggregator_builds_minute_bars_from_5_second_bars() -> None:
    agg = BarAggregator(60)
    t0 = 1_767_625_800  # a minute boundary
    out = []
    for i in range(12):
        out += agg.add(t0 + 5 * i, 100 + i, 100.5 + i, 99.5 + i, 100 + i, 10)
    (bar,) = out  # emitted with the last 5-second bar, without waiting for the next minute
    assert (bar.start, bar.open, bar.high, bar.low, bar.close, bar.volume) == (t0, 100, 111.5, 99.5, 111, 120)
    # a minute with a missing last slot is emitted when the next minute starts
    assert agg.add(t0 + 60, 1, 1, 1, 1, 1) == []
    (late,) = agg.add(t0 + 125, 2, 2, 2, 2, 1)
    assert late.start == t0 + 60


def test_feed_combines_symbols_forward_fills_and_skips_warm_up_overlap() -> None:
    client, api = connected()
    t0 = 1_767_625_800
    api.history = {s: [(t0 - 120, 50.0), (t0 - 60, 51.0), (t0, 52.0)] for s in ("AAPL", "MSFT")}
    feed = IBKRBarFeed(client, ["AAPL", "MSFT"], bar_seconds=60, heartbeat=0.01)
    history = feed.history("1 D")
    assert list(history.columns[:2]) == ["open_AAPL", "high_AAPL"] and "close" in history
    assert history.index[-1] == pd.Timestamp(t0 - 60, unit="s", tz="UTC")  # the forming bar is dropped

    bars: list[pd.Series] = []

    def consume() -> None:
        for item in feed:
            if item is not None:
                bars.append(item)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    wait_until(lambda: len(api.realtime) == 2)
    for i in range(12):  # minute t0: both symbols trade
        api.realtime_bar("AAPL", t0 + 5 * i, 100.0)
        api.realtime_bar("MSFT", t0 + 5 * i, 200.0)
    for i in range(12):  # minute t0 + 60: only AAPL trades
        api.realtime_bar("AAPL", t0 + 60 + 5 * i, 101.0)
    api.realtime_bar("AAPL", t0 + 120 + 55, 102.0)
    api.realtime_bar("MSFT", t0 + 120 + 55, 201.0)
    wait_until(lambda: len(bars) == 3)
    feed.stop()
    thread.join(2)

    assert [b.name for b in bars] == [pd.Timestamp(t0 + 60 * k, unit="s", tz="UTC") for k in range(3)]
    assert (bars[0]["close_AAPL"], bars[0]["close_MSFT"], bars[0]["close"]) == (100.0, 200.0, 100.0)
    assert (bars[1]["close_MSFT"], bars[1]["volume_MSFT"]) == (200.0, 0.0)  # forward-filled
    assert api.realtime == {}  # unsubscribed on stop


def test_feed_raises_when_tws_refuses_real_time_bars() -> None:
    client, api = connected()
    api.realtime_error = (10089, "Requested market data requires additional subscription for API")
    feed = IBKRBarFeed(client, ["AAPL"], heartbeat=0.01)
    with pytest.raises(IBKRRequestError) as info:
        next(item for item in feed if item is not None)
    assert info.value.code == 10089
    assert api.realtime_requests == 1  # not re-subscribed, nothing left to cancel


# --- end to end: runner + IBKR broker + IBKR feed ----------------------------------------------------------


class BuyFirstBar(Strategy):
    holding_bars = 1

    def __init__(self) -> None:
        super().__init__(symbol="AAPL")
        self.done = False

    def decide(self, bar: pd.Series) -> list[Order]:
        if self.done:
            return []
        self.done = True
        return [self.order(Side.BUY, 3)]


def test_runner_on_ibkr_paper_end_to_end() -> None:
    client, api = connected(read_only=False)
    t0 = 1_767_625_800
    api.history = {"AAPL": [(t0 - 60 * k, 100.0) for k in range(5, -1, -1)]}
    feed = IBKRBarFeed(client, ["AAPL"], heartbeat=0.01)
    strategy = BuyFirstBar()
    strategy.fit(feed.history())
    broker = IBKRBroker(client)
    runner = StrategyRunner([strategy], broker, RiskManager(RiskLimits(take_profit_pct=None)))
    thread = threading.Thread(target=runner.run, args=(feed,), daemon=True)
    thread.start()

    wait_until(lambda: api.realtime)
    for i in range(12):
        api.realtime_bar("AAPL", t0 + 5 * i, 100.0)
    wait_until(lambda: len(api.placed) == 1)  # entry sent after the first live bar
    entry_id = api.placed[0][0]
    api.execute(entry_id, 3, 100.1, "x1")
    wait_until(lambda: len(api.placed) == 2)  # protective stop placed from the strategy thread
    stop_id, _, stop = api.placed[1]
    assert (stop.orderType, stop.auxPrice, stop.totalQuantity) == ("STP", 98.1, 3)

    for i in range(12):
        api.realtime_bar("AAPL", t0 + 60 + 5 * i, 101.0)
    wait_until(lambda: len(api.placed) == 3)  # time exit after holding_bars = 1
    assert stop_id in api.cancelled
    exit_id, _, exit_order = api.placed[2]
    assert (exit_order.action, exit_order.orderType, exit_order.totalQuantity) == ("SELL", "MKT", 3)
    api.execute(exit_id, 3, 101.0, "x2")
    wait_until(lambda: len(runner.router.trades) == 1)
    feed.stop()
    thread.join(2)

    (trade,) = runner.router.trades
    assert (trade.exit_price, trade.reason, trade.fees) == (101.0, "time", 2.0)
    assert trade.entry_price == pytest.approx(100.1)
    assert runner.portfolio.quantity("AAPL") == 0
