"""
`Broker` implementation on IBKR (paper by default).

IBKR callbacks arrive on the network thread: they are turned into closures queued here and delivered
to the router on the strategy thread by `process_events` (called by the runner between bars and on
heartbeats). Executions are reported once their commission report arrives (or after
`commission_wait` seconds without one), de-duplicated by execution id, so re-requesting executions
after a reconnect is safe.

Order mapping: our `Order.order_id` travels as the IBKR `orderRef`; protective orders are OCA groups
with `ocaType=2` (a partial fill reduces the sibling) and stay at IBKR (GTC) if this process dies.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from ibapi.commission_report import CommissionReport
from ibapi.contract import Contract
from ibapi.execution import Execution
from ibapi.order import Order as IBOrder

from quantlab.execution.broker import FillHandler, StatusHandler
from quantlab.execution.ibkr.client import IBKRClient, ReadOnlyViolation, stock
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side

logger = logging.getLogger(__name__)

#: IBKR order statuses after which the order cannot fill any more
TERMINAL_STATUSES = {"Cancelled": OrderStatus.CANCELLED, "ApiCancelled": OrderStatus.CANCELLED,
                     "Inactive": OrderStatus.REJECTED}  # fmt: skip
#: IBKR sends this "no value" sentinel in commission reports it cannot price yet
UNSET_DOUBLE = 1.7976931348623157e308


def to_ib_order(order: Order, outside_rth: bool = False) -> IBOrder:
    ib = IBOrder()
    ib.action = order.side.value
    ib.totalQuantity = order.quantity
    ib.orderType = order.order_type.value
    if order.order_type is OrderType.LIMIT:
        ib.lmtPrice = order.limit_price
    elif order.order_type is OrderType.STOP:
        ib.auxPrice = order.stop_price
    if order.oca_group is not None:
        ib.ocaGroup = order.oca_group
        ib.ocaType = 2  # reduce the remaining orders on (partial) fill, no overfill block
    # Protective orders must survive the session (and this process); entries / exits are day orders
    ib.tif = "DAY" if order.order_type is OrderType.MARKET else "GTC"
    ib.outsideRth = outside_rth
    ib.orderRef = order.order_id
    for legacy_flag in ("eTradeOnly", "firmQuoteOnly"):  # rejected by recent TWS versions when True
        if hasattr(ib, legacy_flag):
            setattr(ib, legacy_flag, False)
    return ib


class IBKRBroker:
    def __init__(
        self,
        client: IBKRClient,
        exchange: str = "SMART",
        currency: str = "USD",
        outside_rth: bool = False,
        commission_wait: float = 2.0,
    ) -> None:
        if client.settings.read_only:
            raise ReadOnlyViolation(
                "IBKRBroker needs IBKR_READ_ONLY=false. To run strategies on live IBKR data without sending "
                "orders, use the IBKR bar feed with a SimulatedBroker instead."
            )
        self.client = client
        self.exchange, self.currency, self.outside_rth = exchange, currency, outside_rth
        self.commission_wait = commission_wait
        self._on_fill: FillHandler | None = None
        self._on_status: StatusHandler | None = None
        self._events: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._orders: dict[int, Order] = {}  # IBKR order id -> our order
        self._ib_ids: dict[str, int] = {}  # our order id -> IBKR order id
        self._terminal: set[int] = set()
        self._seen_execs: set[str] = set()
        self._executed: dict[int, int] = {}  # IBKR order id -> shares executed so far
        self._awaiting_commission: dict[str, tuple[Order, Execution]] = {}
        client.order_listener = self
        client.reconnect_listeners.append(self._on_reconnect)

    # --- Broker protocol (strategy thread) -------------------------------------------------------------

    def set_handlers(self, on_fill: FillHandler, on_status: StatusHandler) -> None:
        self._on_fill, self._on_status = on_fill, on_status

    def submit(self, order: Order) -> None:
        contract = stock(order.symbol, self.exchange, self.currency)
        with self._lock:  # register before the first callback can arrive
            ib_id = self.client.place_order(contract, to_ib_order(order, self.outside_rth))
            self._orders[ib_id] = order
            self._ib_ids[order.order_id] = ib_id
        logger.info("Order %s sent as IBKR #%s: %s %d %s %s", order.order_id, ib_id, order.side, order.quantity,
                    order.symbol, order.order_type.value)  # fmt: skip

    def cancel(self, order_id: str) -> None:
        ib_id = self._ib_ids.get(order_id)
        if ib_id is not None and ib_id not in self._terminal:
            self.client.cancel_order(ib_id)

    def positions(self) -> dict[str, int]:
        return {symbol: int(qty) for symbol, qty in self.client.positions().items()}

    def process_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            event()

    def on_bar(self, bar: pd.Series) -> None:
        """Nothing to do: IBKR matches the orders."""

    def reconcile(self, expected: dict[str, int]) -> dict[str, tuple[int, int]]:
        """Symbols whose IBKR position differs from `expected`: {symbol: (ibkr, expected)}."""
        actual = self.positions()
        return {
            symbol: (actual.get(symbol, 0), expected.get(symbol, 0))
            for symbol in set(actual) | set(expected)
            if actual.get(symbol, 0) != expected.get(symbol, 0)
        }

    # --- restart ---------------------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "orders": {str(ib_id): order.to_dict() for ib_id, order in self._orders.items()},
                "terminal": sorted(self._terminal),
                "seen_execs": sorted(self._seen_execs),
                "executed": {str(ib_id): qty for ib_id, qty in self._executed.items()},
            }

    def set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._orders = {int(ib_id): Order.from_dict(o) for ib_id, o in state["orders"].items()}
            self._ib_ids = {order.order_id: ib_id for ib_id, order in self._orders.items()}
            self._terminal = set(state["terminal"])
            self._seen_execs = set(state["seen_execs"])
            self._executed = {int(ib_id): int(qty) for ib_id, qty in state["executed"].items()}

    def recover(self, known_orders: Sequence[Order], timeout: float = 10.0) -> None:
        """
        After a restart (state restored): re-map working orders by their orderRef, replay the executions
        missed while down (duplicates are ignored), then report as cancelled the orders that stopped
        working without being fully executed. Call `process_events` afterwards to deliver it all.
        """
        by_ref = {order.order_id: order for order in known_orders}
        working = self.client.open_orders(timeout)
        with self._lock:
            for ib_id, ref in working.items():
                if ib_id not in self._orders and ref in by_ref:
                    self._orders[ib_id], self._ib_ids[ref] = by_ref[ref], ib_id
        self.client.request_executions(timeout)
        deadline = time.monotonic() + self.commission_wait + 1.0
        while self._awaiting_commission and time.monotonic() < deadline:  # fills before statuses
            time.sleep(0.01)
        for ib_id, order in list(self._orders.items()):
            if ib_id not in working and self._executed.get(ib_id, 0) < order.quantity:
                self._terminate(ib_id, OrderStatus.CANCELLED, "not working at IBKR after the restart")
        logger.info("IBKR recovery: %d working orders, %d executions known", len(working), len(self._seen_execs))

    # --- OrderEventListener (network thread) -----------------------------------------------------------

    def on_order_status(self, order_id: int, status: str, filled: float, remaining: float) -> None:
        mapped = TERMINAL_STATUSES.get(status)
        if mapped is not None:
            self._terminate(order_id, mapped, status)

    def on_order_error(self, order_id: int, code: int, message: str) -> None:
        logger.error("IBKR order #%s error [%s] %s", order_id, code, message)
        self._terminate(order_id, OrderStatus.REJECTED, f"[{code}] {message}")

    def on_execution(self, contract: Contract, execution: Execution) -> None:
        with self._lock:
            if execution.execId in self._seen_execs or execution.execId in self._awaiting_commission:
                return
            order = self._orders.get(execution.orderId) or self._external_order(contract, execution)
            self._awaiting_commission[execution.execId] = (order, execution)
            self._executed[execution.orderId] = self._executed.get(execution.orderId, 0) + int(float(execution.shares))
        timer = threading.Timer(self.commission_wait, self._emit_fill, (execution.execId, 0.0))
        timer.daemon = True
        timer.start()

    def on_commission(self, report: CommissionReport) -> None:
        commission = report.commission if report.commission < UNSET_DOUBLE else 0.0
        self._emit_fill(report.execId, float(commission))

    # --- internals -------------------------------------------------------------------------------------

    def _emit_fill(self, exec_id: str, fees: float) -> None:
        with self._lock:
            pending = self._awaiting_commission.pop(exec_id, None)
            if pending is None:  # already emitted (commission arrived before the timer, or vice versa)
                return
            self._seen_execs.add(exec_id)
        order, execution = pending
        fill = Fill(order, float(execution.price), int(float(execution.shares)), pd.Timestamp.now(tz="UTC"), fees,
                    exec_id)  # fmt: skip
        self._events.put(lambda: self._on_fill(fill) if self._on_fill else None)

    def _terminate(self, ib_id: int, status: OrderStatus, message: str) -> None:
        with self._lock:
            order = self._orders.get(ib_id)
            if order is None or ib_id in self._terminal:
                return
            self._terminal.add(ib_id)
        self._events.put(lambda: self._on_status(order.order_id, status, message) if self._on_status else None)

    @staticmethod
    def _external_order(contract: Contract, execution: Execution) -> Order:
        """An execution of an order this process did not send (manual TWS order, previous session)."""
        side = Side.BUY if execution.side in ("BOT", "BUY") else Side.SELL
        logger.warning("Execution of an unknown IBKR order #%s (%s): %s", execution.orderId, execution.orderRef,
                       execution.execId)  # fmt: skip
        return Order(contract.symbol, side, max(int(float(execution.shares)), 1), "external",
                     order_id=execution.orderRef or f"ib-{execution.orderId}")  # fmt: skip

    def _on_reconnect(self) -> None:
        logger.info("IBKR reconnected: re-requesting today's executions")
        self.client.request_executions()
