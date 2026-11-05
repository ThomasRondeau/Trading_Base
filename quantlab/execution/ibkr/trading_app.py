"""
IBKR TWS / IB Gateway client (EClient + EWrapper), used as a process-wide singleton.

Will be rewritten in step 4 of the refactor plan (request queues, events, reconnection).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData, TickerId
from ibapi.contract import Contract
from ibapi.order import Order as IBOrder
from ibapi.wrapper import EWrapper

logger = logging.getLogger(__name__)

# Informational "errors" sent by TWS when a farm connection is OK / inactive.
INFO_CODES = {2104, 2106, 2158, 2119}
# Codes meaning the connection to TWS is lost.
CONNECTION_ERROR_CODES = {502, 504, 1100, 1101, 1102}

LiveCallback = Callable[[int, dict[str, Any]], None]


class TradingApp(EClient, EWrapper):  # type: ignore[misc]
    _instance: TradingApp | None = None

    @classmethod
    def get_instance(cls) -> TradingApp:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if TradingApp._instance is not None:
            raise RuntimeError("TradingApp is a singleton: use TradingApp.get_instance()")

        EClient.__init__(self, self)
        self.data: dict[int, pd.DataFrame] = {}
        self.nextOrderId: int | None = None
        self.connected = False
        # Live streaming state
        self.live_data: dict[int, dict[str, Any]] = {}
        self.live_callbacks: dict[int, Callable[..., None]] = {}

    # --- Connection ------------------------------------------------------------------------------------

    def error(self, reqId: int, errorCode: int, errorString: str, advanced: str = "") -> None:
        if errorCode in INFO_CODES:
            logger.debug("IBKR info %s: %s", errorCode, errorString)
            return
        logger.error("IBKR error: reqId=%s code=%s %s %s", reqId, errorCode, errorString, advanced)
        if errorCode in CONNECTION_ERROR_CODES:
            logger.error("IBKR connection lost")
            self.connected = False

    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        self.connected = True
        logger.info("Connected to IBKR. Next order id: %s", orderId)

    def connectionClosed(self) -> None:
        logger.info("IBKR connection closed")
        self.connected = False

    # --- Historical data -------------------------------------------------------------------------------

    def get_historical_data(self, reqId: int, contract: Contract, end_date: str = "") -> pd.DataFrame:
        """One day of 1-minute TRADES bars ending at `end_date` (blocking, 10 s timeout)."""
        if not self.connected:
            logger.error("Not connected to TWS/IB Gateway")
            return pd.DataFrame()

        self.data[reqId] = pd.DataFrame(columns=["Datetime", "high", "low", "open", "close", "volume"])
        self.data[reqId].set_index("Datetime", inplace=True)

        if not end_date:
            # Default: 4 days ago at 17:00 ET, to be sure to get a full session
            end_date = (datetime.now() - timedelta(days=4)).strftime("%Y%m%d 17:00:00 US/Eastern")

        logger.info("Requesting historical data for %s until %s...", contract.symbol, end_date)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime=end_date,
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=2,
            formatDate=2,  # epoch seconds
            keepUpToDate=False,
            chartOptions=[],
        )

        timeout = 10
        start_time = time.time()
        start_length = len(self.data[reqId])

        # Wait for the first bars...
        while len(self.data[reqId]) == 0 and (time.time() - start_time) < timeout:
            time.sleep(0.3)
        # ...then until no more bars arrive
        while len(self.data[reqId]) > start_length and (time.time() - start_time) < timeout:
            time.sleep(1)
            start_length = len(self.data[reqId])

        if len(self.data[reqId]) == 0:
            logger.warning("No historical data received before timeout")
        else:
            logger.info(
                "Received %d bars: %s to %s",
                len(self.data[reqId]),
                self.data[reqId].index[0],
                self.data[reqId].index[-1],
            )
        return self.data[reqId]

    def historicalData(self, reqId: int, bar: BarData) -> None:
        if reqId not in self.data:
            return

        df = self.data[reqId]
        try:
            if isinstance(bar.date, (int, float)):  # epoch seconds
                date_index = pd.to_datetime(bar.date, unit="s")
            else:  # string such as "20240101 15:30:00"
                date_index = pd.to_datetime(bar.date)
        except (ValueError, TypeError):
            try:
                date_index = pd.to_datetime(float(bar.date), unit="s")
            except (ValueError, TypeError):
                logger.error("Cannot parse IBKR bar date: %s", bar.date)
                return

        df.loc[date_index, ["high", "low", "open", "close", "volume"]] = [
            float(bar.high),
            float(bar.low),
            float(bar.open),
            float(bar.close),
            float(bar.volume),
        ]
        self.data[reqId] = df

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        logger.debug("End of historical data for reqId %s", reqId)

    # --- Orders ----------------------------------------------------------------------------------------

    @staticmethod
    def get_contract(symbol: str, exchange: str = "SMART") -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.primaryExchange = exchange if exchange != "SMART" else ""
        contract.exchange = exchange
        contract.currency = "USD"
        return contract

    def place_order(self, contract: Contract, action: str, order_type: str, quantity: int) -> None:
        if not self.connected or self.nextOrderId is None:
            logger.error("Not connected to TWS/IB Gateway")
            return

        order = IBOrder()
        order.action = action
        order.orderType = order_type
        order.totalQuantity = quantity

        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1
        logger.info("Order placed: %s %s %s", action, quantity, contract.symbol)

    # --- Live market data ------------------------------------------------------------------------------

    def start_live_data(self, reqId: int, contract: Contract, callback: LiveCallback | None = None) -> None:
        """Stream top-of-book ticks; `callback(reqId, snapshot)` is called on every update."""
        if not self.connected:
            logger.error("Not connected to TWS/IB Gateway")
            return

        self.live_data[reqId] = {
            "symbol": contract.symbol,
            "last_price": None,
            "bid": None,
            "ask": None,
            "volume": None,
            "last_update": None,
        }
        if callback:
            self.live_callbacks[reqId] = callback

        self.reqMktData(reqId, contract, "", False, False, [])
        logger.info("Live data started for %s (reqId %s)", contract.symbol, reqId)

    def stop_live_data(self, reqId: int) -> None:
        if reqId in self.live_data:
            self.cancelMktData(reqId)
            del self.live_data[reqId]
            self.live_callbacks.pop(reqId, None)
            logger.info("Live data stopped for reqId %s", reqId)

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib: Any) -> None:
        if reqId not in self.live_data:
            return
        if tickType == 4:  # LAST
            self.live_data[reqId]["last_price"] = price
        elif tickType == 1:  # BID
            self.live_data[reqId]["bid"] = price
        elif tickType == 2:  # ASK
            self.live_data[reqId]["ask"] = price

        self.live_data[reqId]["last_update"] = datetime.now()
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, self.live_data[reqId].copy())

    def tickSize(self, reqId: TickerId, tickType: int, size: int) -> None:
        if reqId not in self.live_data:
            return
        if tickType == 8:  # VOLUME
            self.live_data[reqId]["volume"] = size

        self.live_data[reqId]["last_update"] = datetime.now()
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, self.live_data[reqId].copy())

    def get_live_data(self, reqId: int) -> dict[str, Any] | None:
        return self.live_data.get(reqId)

    def start_live_bars(
        self, reqId: int, contract: Contract, bar_size: int = 5, callback: LiveCallback | None = None
    ) -> None:
        """Stream real-time bars of `bar_size` seconds; `callback(reqId, bar_dict)` on each bar."""
        if not self.connected:
            logger.error("Not connected to TWS/IB Gateway")
            return
        if callback:
            self.live_callbacks[reqId] = callback

        self.reqRealTimeBars(
            reqId=reqId, contract=contract, barSize=bar_size, whatToShow="TRADES", useRTH=False, realTimeBarsOptions=[]
        )
        logger.info("Live bars started for %s (reqId %s, %ss)", contract.symbol, reqId, bar_size)

    def stop_live_bars(self, reqId: int) -> None:
        self.cancelRealTimeBars(reqId)
        self.live_callbacks.pop(reqId, None)
        logger.info("Live bars stopped for reqId %s", reqId)

    def realtimeBar(
        self,
        reqId: TickerId,
        time: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        wap: float,
        count: int,
    ) -> None:
        bar_data = {
            "time": datetime.fromtimestamp(time),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "wap": wap,
            "count": count,
        }
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, bar_data)
