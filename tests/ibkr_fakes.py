"""Fake EClient: answers requests by calling the wrapper callbacks, like TWS would. No socket, no TWS."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from ibapi.commission_report import CommissionReport
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.execution import Execution
from ibapi.order import Order as IBOrder
from ibapi.order_state import OrderState

from quantlab.config import IBKRSettings
from quantlab.execution.ibkr.client import IBKRClient, stock

PAPER = "DU1234567"


class FakeEClient:
    def __init__(self, wrapper: Any, settings: IBKRSettings) -> None:
        self.wrapper = wrapper
        self.settings = settings
        self.accounts = PAPER
        self.connected = False
        self.refuse_connections = False
        self.connect_count = 0
        self.history: dict[str, list[tuple[int, float]]] = {}
        self.history_error: tuple[int, str] | None = None
        self.answer_history = True
        self.positions: list[tuple[str, float]] = []
        self.placed: list[tuple[int, Contract, IBOrder]] = []
        self.cancelled: list[int] = []
        self.realtime: dict[int, Contract] = {}
        self.realtime_requests = 0
        self.executions_requested = 0
        self.realtime_error: tuple[int, str] | None = None  # e.g. no market data subscription
        self.past_executions: list[tuple[Contract, Execution, float]] = []  # replayed by reqExecutions
        self.working: dict[int, tuple[Contract, IBOrder]] = {}  # reported by reqOpenOrders

    # connection
    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connect_count += 1
        self.connected = not self.refuse_connections

    def run(self) -> None:
        if self.connected:
            self.wrapper.managedAccounts(self.accounts)
            self.wrapper.nextValidId(100)

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def drop(self) -> None:
        """Simulate TWS going away."""
        self.connected = False
        self.wrapper.connectionClosed()

    # requests
    def reqHistoricalData(self, req_id: int, contract: Contract, *args: Any) -> None:
        if self.history_error:
            self.wrapper.error(req_id, *self.history_error)
            return
        if not self.answer_history:
            return
        for epoch, close in self.history.get(contract.symbol, []):
            bar = BarData()
            bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume = (
                str(epoch), close, close + 1, close - 1, close, Decimal(100)
            )  # fmt: skip
            self.wrapper.historicalData(req_id, bar)
        self.wrapper.historicalDataEnd(req_id, "", "")

    def cancelHistoricalData(self, req_id: int) -> None:
        pass

    def reqPositions(self) -> None:
        for symbol, qty in self.positions:
            self.wrapper.position(PAPER, stock(symbol), Decimal(qty), 1.0)
        self.wrapper.positionEnd()

    def cancelPositions(self) -> None:
        pass

    def reqRealTimeBars(self, req_id: int, contract: Contract, *args: Any) -> None:
        self.realtime_requests += 1
        if self.realtime_error is not None:
            self.wrapper.error(req_id, *self.realtime_error)
            return
        self.realtime[req_id] = contract

    def cancelRealTimeBars(self, req_id: int) -> None:
        self.realtime.pop(req_id, None)

    def reqExecutions(self, req_id: int, execution_filter: Any) -> None:
        self.executions_requested += 1
        for contract, execution, commission in self.past_executions:
            self.wrapper.execDetails(req_id, contract, execution)
            report = CommissionReport()
            report.execId, report.commission = execution.execId, commission
            self.wrapper.commissionReport(report)
        self.wrapper.execDetailsEnd(req_id)

    def reqOpenOrders(self) -> None:
        for order_id, (contract, order) in self.working.items():
            state = OrderState()
            state.status = "Submitted"
            self.wrapper.openOrder(order_id, contract, order, state)
        self.wrapper.openOrderEnd()

    def placeOrder(self, order_id: int, contract: Contract, order: IBOrder) -> None:
        self.placed.append((order_id, contract, order))

    def cancelOrder(self, order_id: int, order_cancel: Any) -> None:
        self.cancelled.append(order_id)

    # helpers for the tests
    def realtime_bar(self, symbol: str, epoch: int, close: float, volume: float = 10) -> None:
        req_id = next(r for r, c in self.realtime.items() if c.symbol == symbol)
        self.wrapper.realtimeBar(req_id, epoch, close, close + 0.5, close - 0.5, close, Decimal(volume), Decimal(0), 1)

    def execute(
        self, ib_id: int, shares: int, price: float, exec_id: str, commission: float | None = 1.0, silent: bool = False
    ) -> None:
        """Execute a placed order; `silent` only records it (it happened while the client was down)."""
        _, contract, order = next(p for p in self.placed if p[0] == ib_id)
        execution = Execution()
        execution.execId, execution.orderId, execution.orderRef = exec_id, ib_id, order.orderRef
        execution.shares, execution.price = Decimal(shares), price
        execution.side = "BOT" if order.action == "BUY" else "SLD"
        self.past_executions.append((contract, execution, commission or 0.0))
        if silent:
            return
        self.wrapper.execDetails(-1, contract, execution)
        if commission is not None:
            report = CommissionReport()
            report.execId, report.commission = exec_id, commission
            self.wrapper.commissionReport(report)


def make_client(read_only: bool = True, allow_live: bool = False, **kwargs: Any) -> tuple[IBKRClient, FakeEClient]:
    settings = IBKRSettings(read_only=read_only, allow_live_account=allow_live)
    client = IBKRClient(settings, api_factory=FakeEClient, **kwargs)
    return client, client.api


def connected(read_only: bool = True, **kwargs: Any) -> tuple[IBKRClient, FakeEClient]:
    client, api = make_client(read_only=read_only, **kwargs)
    client.connect(timeout=2)
    return client, api


def wait_until(condition: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)
