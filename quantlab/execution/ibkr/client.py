"""
IBKR TWS / IB Gateway client.

`IBKRClient` is the EWrapper; the socket side is an `EClient` built by `api_factory` (a fake one in the
tests, so the whole client runs without TWS). Threading model:

- the ibapi reader and the message loop (`EClient.run`) live on daemon threads: every callback below
  runs on the network thread and only stores data, sets an `Event` or forwards to a listener that
  queues it, so a slow consumer never blocks the reception;
- blocking requests (historical bars, positions) wait on a per-request `Event` with a timeout;
- request and order ids are allocated under a lock;
- when the connection drops unexpectedly, a reconnect thread retries with exponential backoff, then
  restores the real-time bar subscriptions and notifies the reconnect listeners (the broker
  re-requests executions and reconciles).

Safety: order-mutating calls are refused while `settings.read_only` is True (the default), both here
and in the `EClient` itself (`GuardedEClient`).
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

import pandas as pd
from ibapi.client import EClient
from ibapi.commission_report import CommissionReport
from ibapi.common import BarData, TickerId
from ibapi.contract import Contract
from ibapi.execution import Execution, ExecutionFilter
from ibapi.order import Order as IBOrder
from ibapi.order_cancel import OrderCancel
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper

from quantlab.config import IBKRSettings
from quantlab.data.schema import validate_ohlcv

logger = logging.getLogger(__name__)

#: farm-status notices sent through `error`
INFO_CODES = {2104, 2106, 2107, 2108, 2158, 2119}
CONNECTIVITY_LOST, CONNECTIVITY_RESTORED_DATA_LOST, CONNECTIVITY_RESTORED = 1100, 1101, 1102
NOT_CONNECTED_CODES = {502, 504}
#: order-level codes that do not mean the order is dead (e.g. "order held", price capped)
ORDER_WARNING_CODES = {399, 404, 2109, 10148}


class ReadOnlyViolation(PermissionError):
    """An order-mutating call was attempted while the IBKR connection is read-only."""


class LiveAccountViolation(ReadOnlyViolation):
    """An order was attempted on a real-money account without IBKR_ALLOW_LIVE=true."""


class IBKRRequestError(RuntimeError):
    def __init__(self, req_id: int, code: int, message: str) -> None:
        super().__init__(f"IBKR request {req_id} failed: [{code}] {message}")
        self.code = code


def _ensure_orders_allowed(settings: IBKRSettings, action: str) -> None:
    if settings.read_only:
        raise ReadOnlyViolation(
            f"{action} refused: IBKR connection is read-only (set IBKR_READ_ONLY=false to allow orders)."
        )


class GuardedEClient(EClient):  # type: ignore[misc]
    """The real socket client, with the read-only guard on every order-mutating call."""

    def __init__(self, wrapper: EWrapper, settings: IBKRSettings) -> None:
        super().__init__(wrapper)
        self.settings = settings

    def placeOrder(self, orderId: int, contract: Contract, order: IBOrder) -> None:
        _ensure_orders_allowed(self.settings, "placeOrder")
        super().placeOrder(orderId, contract, order)

    def cancelOrder(self, orderId: int, orderCancel: OrderCancel) -> None:
        _ensure_orders_allowed(self.settings, "cancelOrder")
        super().cancelOrder(orderId, orderCancel)

    def reqGlobalCancel(self, *args: Any) -> None:
        _ensure_orders_allowed(self.settings, "reqGlobalCancel")
        super().reqGlobalCancel(*args)


ApiFactory = Callable[[EWrapper, IBKRSettings], Any]


class OrderEventListener(Protocol):
    """Receives order events on the network thread (the `IBKRBroker` queues them)."""

    def on_order_status(self, order_id: int, status: str, filled: float, remaining: float) -> None: ...

    def on_execution(self, contract: Contract, execution: Execution) -> None: ...

    def on_commission(self, report: CommissionReport) -> None: ...

    def on_order_error(self, order_id: int, code: int, message: str) -> None: ...


def stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = exchange
    contract.currency = currency
    return contract


def _bar_time(raw: str) -> pd.Timestamp:
    """Bar date with formatDate=2: epoch seconds for intraday bars (maybe followed by a zone), YYYYMMDD for daily."""
    token = raw.split()[0]
    if len(token) == 8:
        return pd.Timestamp(token, tz="UTC")
    return pd.Timestamp(int(token), unit="s", tz="UTC")


@dataclass
class _Request:
    done: threading.Event = field(default_factory=threading.Event)
    rows: list[Any] = field(default_factory=list)
    error: IBKRRequestError | None = None


@dataclass
class _BarSubscription:
    contract: Contract
    callback: Callable[[dict[str, Any]], None]
    use_rth: bool
    on_error: Callable[[IBKRRequestError], None] | None = None


class IBKRClient(EWrapper):  # type: ignore[misc]
    def __init__(
        self,
        settings: IBKRSettings | None = None,
        api_factory: ApiFactory = GuardedEClient,
        auto_reconnect: bool = True,
        reconnect_delays: tuple[float, ...] = (1, 2, 5, 10, 30),
    ) -> None:
        super().__init__()
        self.settings = settings or IBKRSettings.from_env()
        self.api = api_factory(self, self.settings)
        self.auto_reconnect = auto_reconnect
        self.reconnect_delays = reconnect_delays
        self.accounts: list[str] = []
        self.order_listener: OrderEventListener | None = None
        self.reconnect_listeners: list[Callable[[], None]] = []

        self._lock = threading.Lock()
        self._ready = threading.Event()  # nextValidId received
        self._accounts_ready = threading.Event()
        self._req_ids = itertools.count(900_000_000)  # far from the order ids (same id space)
        self._next_order_id: int | None = None
        self._requests: dict[int, _Request] = {}
        self._positions: _Request | None = None
        self._open_orders: _Request | None = None
        self._subscriptions: dict[int, _BarSubscription] = {}
        self._closing = False
        self._reconnecting = False

    # --- connection ------------------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._ready.is_set() and bool(self.api.isConnected())

    def connect(self, timeout: float = 10.0) -> None:
        """Connect and wait for the first order id; raises ConnectionError on failure."""
        s = self.settings
        logger.info("Connecting to IBKR %s:%s (client id %s, read-only=%s)", s.host, s.port, s.client_id, s.read_only)
        self._closing = False
        self._ready.clear()
        self._accounts_ready.clear()
        self.api.connect(s.host, s.port, s.client_id)
        threading.Thread(target=self.api.run, name="ibkr-network", daemon=True).start()
        if not self._ready.wait(timeout):
            self.api.disconnect()
            raise ConnectionError(
                f"No answer from TWS/IB Gateway on {s.host}:{s.port} within {timeout:.0f} s: check that it runs, "
                "that Configure > API > Settings > 'Enable ActiveX and Socket Clients' is checked, and that the "
                f"socket port matches IBKR_PORT and client id {s.client_id} is free."
            )
        self._accounts_ready.wait(2.0)

    def disconnect(self) -> None:
        self._closing = True
        self._ready.clear()
        self.api.disconnect()

    @property
    def is_paper(self) -> bool:
        """Paper-trading accounts are named DU... (DF... for paper advisor accounts)."""
        return bool(self.accounts) and all(a.startswith(("DU", "DF")) for a in self.accounts)

    def ensure_account_allowed(self) -> None:
        """Orders go to paper accounts only, unless the settings explicitly allow a real-money account."""
        if not self.accounts:
            raise LiveAccountViolation("Unknown account type (no managedAccounts from TWS yet): orders refused.")
        if not self.is_paper and not self.settings.allow_live_account:
            raise LiveAccountViolation(
                "Connected to a real-money account: orders refused (paper accounts are DU...). "
                "Set IBKR_ALLOW_LIVE=true only if you really mean to trade real money."
            )

    def _new_req_id(self) -> int:
        with self._lock:
            return next(self._req_ids)

    def next_order_id(self) -> int:
        with self._lock:
            if self._next_order_id is None:
                raise ConnectionError("No order id from TWS yet: not connected.")
            order_id = self._next_order_id
            self._next_order_id += 1
            return order_id

    # --- callbacks: connection -------------------------------------------------------------------------

    def nextValidId(self, orderId: int) -> None:
        with self._lock:
            self._next_order_id = max(orderId, self._next_order_id or 0)
        logger.info("Connected to IBKR (next order id %s)", orderId)
        self._ready.set()

    def managedAccounts(self, accountsList: str) -> None:
        self.accounts = [a for a in accountsList.split(",") if a]
        self._accounts_ready.set()

    def connectionClosed(self) -> None:
        was_ready = self._ready.is_set()
        self._ready.clear()
        logger.log(logging.INFO if self._closing else logging.WARNING, "IBKR connection closed")
        self._fail_pending(IBKRRequestError(-1, 504, "connection closed"))
        if was_ready and not self._closing and self.auto_reconnect:
            self._start_reconnect()

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        if errorCode in INFO_CODES:
            logger.debug("IBKR info %s: %s", errorCode, errorString)
            return
        if errorCode == CONNECTIVITY_LOST:
            logger.error("IBKR: TWS lost its connection to IB servers (%s)", errorString)
            return
        if errorCode in (CONNECTIVITY_RESTORED, CONNECTIVITY_RESTORED_DATA_LOST):
            logger.warning("IBKR: connectivity restored (%s)", errorString)
            if errorCode == CONNECTIVITY_RESTORED_DATA_LOST:
                self._resubscribe()
            return
        request = self._requests.get(reqId)
        if request is not None:
            request.error = IBKRRequestError(reqId, errorCode, errorString)
            request.done.set()
            return
        sub = self._subscriptions.get(reqId)
        if sub is not None:
            # e.g. 10089 / 354: no market data subscription for the API. The stream will never start.
            logger.error("IBKR real-time bars %s (%s): [%s] %s", reqId, sub.contract.symbol, errorCode, errorString)
            del self._subscriptions[reqId]
            if sub.on_error is not None:
                sub.on_error(IBKRRequestError(reqId, errorCode, errorString))
            return
        if reqId > 0 and self.order_listener is not None and errorCode not in ORDER_WARNING_CODES:
            self.order_listener.on_order_error(reqId, errorCode, errorString)
            return
        if errorCode in NOT_CONNECTED_CODES:
            self._ready.clear()
        logger.error("IBKR error: reqId=%s code=%s %s %s", reqId, errorCode, errorString, advancedOrderRejectJson)

    # --- reconnection ----------------------------------------------------------------------------------

    def _start_reconnect(self) -> None:
        with self._lock:
            if self._reconnecting:
                return
            self._reconnecting = True
        threading.Thread(target=self._reconnect_loop, name="ibkr-reconnect", daemon=True).start()

    def _reconnect_loop(self) -> None:
        try:
            for attempt in itertools.count(1):
                if self._closing:
                    return
                delay = self.reconnect_delays[min(attempt - 1, len(self.reconnect_delays) - 1)]
                logger.info("IBKR reconnect attempt %d in %.0f s", attempt, delay)
                time.sleep(delay)
                try:
                    self.api.disconnect()
                    self.connect()
                except ConnectionError as exc:
                    logger.warning("IBKR reconnect attempt %d failed: %s", attempt, exc)
                    continue
                self._resubscribe()
                for listener in self.reconnect_listeners:
                    listener()
                logger.info("IBKR reconnected")
                return
        finally:
            with self._lock:
                self._reconnecting = False

    def _fail_pending(self, error: IBKRRequestError) -> None:
        for request in list(self._requests.values()):
            request.error = request.error or error
            request.done.set()

    # --- historical bars -------------------------------------------------------------------------------

    def historical_bars(
        self,
        contract: Contract,
        end: str = "",
        duration: str = "1 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        timeout: float = 60.0,
    ) -> pd.DataFrame:
        """
        Historical bars as a validated OHLCV frame (UTC index). `end` is IBKR's endDateTime
        ("" = now, or e.g. "20260105 16:00:00 US/Eastern"). Raises IBKRRequestError / TimeoutError.
        """
        req_id = self._new_req_id()
        request = self._requests[req_id] = _Request()
        try:
            self.api.reqHistoricalData(
                req_id, contract, end, duration, bar_size, what_to_show, int(use_rth), 2, False, []
            )
            if not request.done.wait(timeout):
                self.api.cancelHistoricalData(req_id)
                raise TimeoutError(f"Historical data request {req_id} for {contract.symbol} timed out.")
            if request.error is not None:
                raise request.error
        finally:
            self._requests.pop(req_id, None)
        frame = pd.DataFrame(request.rows, columns=["Datetime", "open", "high", "low", "close", "volume"])
        if not frame.empty:
            frame["Datetime"] = [_bar_time(raw) for raw in frame["Datetime"]]
        return validate_ohlcv(frame, datetime_column="Datetime" if not frame.empty else None)

    def historicalData(self, reqId: int, bar: BarData) -> None:
        request = self._requests.get(reqId)
        if request is not None:
            request.rows.append(
                (str(bar.date), float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume))
            )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        request = self._requests.get(reqId)
        if request is not None:
            request.done.set()

    # --- positions -------------------------------------------------------------------------------------

    def positions(self, timeout: float = 10.0) -> dict[str, float]:
        """Net stock position per symbol, summed over the managed accounts."""
        request = self._positions = _Request()
        self.api.reqPositions()
        try:
            if not request.done.wait(timeout):
                raise TimeoutError("Positions request timed out.")
        finally:
            self.api.cancelPositions()
            self._positions = None
        totals: dict[str, float] = {}
        for symbol, quantity in request.rows:
            totals[symbol] = totals.get(symbol, 0.0) + quantity
        return {symbol: qty for symbol, qty in totals.items() if qty}

    def position(self, account: str, contract: Contract, position: Decimal, avgCost: float) -> None:
        if self._positions is not None and contract.secType == "STK":
            self._positions.rows.append((contract.symbol, float(position)))

    def positionEnd(self) -> None:
        if self._positions is not None:
            self._positions.done.set()

    # --- real-time bars --------------------------------------------------------------------------------

    def subscribe_realtime_bars(
        self,
        contract: Contract,
        callback: Callable[[dict[str, Any]], None],
        use_rth: bool = False,
        on_error: Callable[[IBKRRequestError], None] | None = None,
    ) -> int:
        """
        5-second TRADES bars; `callback(bar)` runs on the network thread. Restored after a reconnect.
        `on_error` is called (network thread) if TWS refuses or ends the subscription.
        """
        req_id = self._new_req_id()
        self._subscriptions[req_id] = _BarSubscription(contract, callback, use_rth, on_error)
        self.api.reqRealTimeBars(req_id, contract, 5, "TRADES", use_rth, [])
        logger.info("Real-time bars for %s (reqId %s)", contract.symbol, req_id)
        return req_id

    def cancel_realtime_bars(self, req_id: int) -> None:
        if self._subscriptions.pop(req_id, None) is not None and self.is_connected:
            self.api.cancelRealTimeBars(req_id)

    def _resubscribe(self) -> None:
        for req_id, sub in list(self._subscriptions.items()):
            self.api.reqRealTimeBars(req_id, sub.contract, 5, "TRADES", sub.use_rth, [])
            logger.info("Real-time bars for %s restored (reqId %s)", sub.contract.symbol, req_id)

    def realtimeBar(
        self,
        reqId: TickerId,
        time: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: Decimal,
        wap: Decimal,
        count: int,
    ) -> None:
        sub = self._subscriptions.get(reqId)
        if sub is not None:
            sub.callback(
                {
                    "symbol": sub.contract.symbol,
                    "time": int(time),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )

    # --- orders ----------------------------------------------------------------------------------------

    def place_order(self, contract: Contract, order: IBOrder) -> int:
        """Send `order` with a fresh order id and return that id (refused while read-only)."""
        _ensure_orders_allowed(self.settings, "place_order")
        self.ensure_account_allowed()
        order_id = self.next_order_id()
        self.api.placeOrder(order_id, contract, order)
        return order_id

    def cancel_order(self, order_id: int) -> None:
        _ensure_orders_allowed(self.settings, "cancel_order")
        self.api.cancelOrder(order_id, OrderCancel())

    def request_executions(self, timeout: float | None = None) -> None:
        """
        Re-send today's executions to the order listener (after a reconnect / restart; the listener must
        ignore duplicates). With a `timeout`, wait until TWS has sent them all.
        """
        req_id = self._new_req_id()
        request = self._requests[req_id] = _Request()
        try:
            self.api.reqExecutions(req_id, ExecutionFilter())
            if timeout is not None and not request.done.wait(timeout):
                raise TimeoutError("Executions request timed out.")
        finally:
            self._requests.pop(req_id, None)

    def open_orders(self, timeout: float = 10.0) -> dict[int, str]:
        """Working orders of this client id: {IBKR order id: orderRef}."""
        request = self._open_orders = _Request()
        self.api.reqOpenOrders()
        try:
            if not request.done.wait(timeout):
                raise TimeoutError("Open orders request timed out.")
        finally:
            self._open_orders = None
        return dict(request.rows)

    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: Decimal,
        remaining: Decimal,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:
        if self.order_listener is not None:
            self.order_listener.on_order_status(orderId, status, float(filled), float(remaining))

    def openOrder(self, orderId: int, contract: Contract, order: IBOrder, orderState: OrderState) -> None:
        logger.debug("Open order %s %s %s: %s", orderId, order.action, contract.symbol, orderState.status)
        if self._open_orders is not None and orderState.status not in (
            "Filled",
            "Cancelled",
            "ApiCancelled",
            "Inactive",
        ):
            self._open_orders.rows.append((orderId, order.orderRef))

    def openOrderEnd(self) -> None:
        if self._open_orders is not None:
            self._open_orders.done.set()

    def execDetailsEnd(self, reqId: int) -> None:
        request = self._requests.get(reqId)
        if request is not None:
            request.done.set()

    def execDetails(self, reqId: int, contract: Contract, execution: Execution) -> None:
        if self.order_listener is not None:
            self.order_listener.on_execution(contract, execution)

    def commissionReport(self, commissionReport: CommissionReport) -> None:
        if self.order_listener is not None:
            self.order_listener.on_commission(commissionReport)
