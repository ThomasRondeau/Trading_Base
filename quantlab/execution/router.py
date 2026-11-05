"""
Order routing shared by backtest and live: risk checks, trade lifecycle and fill attribution.

Every entry order a strategy emits opens a *lot* owned by that strategy. Once the entry is filled, the
router places the lot's protective orders at the broker (an OCA pair: stop loss + take profit, from the
`RiskLimits`), so they keep working if the process dies. After `holding_bars` bars it cancels them and
closes the lot at market. Several strategies may hold lots on the same symbol: the broker position is
their net sum, and each lot's exit moves it by exactly that lot's quantity.

Fills are attributed to lots through the order ids. A live race (a protective order filling while the
time exit is in flight) can over-close a lot: the excess becomes a new lot that is closed at once.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from quantlab.execution.broker import Broker
from quantlab.execution.orders import Fill, Order, OrderStatus, OrderType, Side
from quantlab.execution.portfolio import Portfolio
from quantlab.execution.risk import RiskManager

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    PROTECT = "protect"


@dataclass(frozen=True)
class Trade:
    """A closed (or, at the end of a backtest, marked-to-market) round trip of one lot."""

    strategy_id: str
    symbol: str
    side: Side
    quantity: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    fees: float
    bars_held: int
    reason: str

    @property
    def pnl(self) -> float:
        return self.side.sign * (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_net(self) -> float:
        return self.pnl - self.fees

    @property
    def return_pct(self) -> float:
        """Net P&L as a percentage of the entry notional."""
        return 100 * self.pnl_net / (self.entry_price * self.quantity)


@dataclass
class Lot:
    lot_id: str
    strategy_id: str
    symbol: str
    side: Side
    holding_bars: int | None
    entry_order_id: str = ""
    quantity: int = 0  # still open
    entry_quantity: int = 0
    entry_price: float = 0.0  # average fill price
    entry_time: pd.Timestamp | None = None
    entry_done: bool = False
    fees: float = 0.0
    exit_quantity: int = 0
    exit_value: float = 0.0
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""
    bars_held: int = 0
    protective_ids: set[str] = field(default_factory=set)
    exit_order_id: str | None = None


@dataclass
class _Tracked:
    order: Order
    role: Role
    lot_id: str
    reason: str = ""
    filled: int = 0

    @property
    def remaining(self) -> int:
        return self.order.quantity - self.filled


class OrderRouter:
    def __init__(
        self, broker: Broker, portfolio: Portfolio, risk: RiskManager | None = None, tick_size: float = 0.01
    ) -> None:
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk or RiskManager()
        self.tick_size = tick_size
        self.lots: dict[str, Lot] = {}
        self.trades: list[Trade] = []
        self.fill_listeners: list[Callable[[Fill], None]] = []
        self._orders: dict[str, _Tracked] = {}
        self._lot_ids = (f"lot{i}" for i in itertools.count(1))
        broker.set_handlers(self.on_fill, self.on_status)

    # --- entries ---------------------------------------------------------------------------------------

    def exposure(self, symbol: str) -> int:
        """Net position plus the unfilled part of pending entries (signed)."""
        pending = sum(
            t.order.side.sign * t.remaining
            for t in self._orders.values()
            if t.role is Role.ENTRY and t.order.symbol == symbol and t.remaining > 0
        )
        return self.portfolio.quantity(symbol) + pending

    def submit_entry(self, order: Order, holding_bars: int | None) -> Order | None:
        """Risk-check `order` and send it; returns the order actually sent (None if refused)."""
        checked = self.risk.check_entry(order, self.exposure(order.symbol))
        if checked is None:
            return None
        lot = Lot(next(self._lot_ids), order.strategy_id, order.symbol, order.side, holding_bars, checked.order_id)
        self.lots[lot.lot_id] = lot
        self._send(checked, Role.ENTRY, lot.lot_id)
        return checked

    # --- broker events ---------------------------------------------------------------------------------

    def on_fill(self, fill: Fill) -> None:
        self.portfolio.apply_fill(fill.order.symbol, fill.signed_quantity, fill.price, fill.fees)
        tracked = self._orders.get(fill.order.order_id)
        if tracked is None:
            logger.warning("Fill of an order not sent by this router (applied to the portfolio only): %s", fill)
            return
        tracked.filled += fill.quantity
        lot = self.lots.get(tracked.lot_id)
        if lot is None:
            logger.error("Fill for the already closed %s: %s", tracked.lot_id, fill)
            self._open_correction(fill.order.symbol, fill.order.side, fill.quantity, fill)
        elif tracked.role is Role.ENTRY:
            total = lot.entry_quantity + fill.quantity
            lot.entry_price = (lot.entry_price * lot.entry_quantity + fill.price * fill.quantity) / total
            lot.entry_quantity, lot.quantity = total, lot.quantity + fill.quantity
            lot.entry_time = lot.entry_time or fill.timestamp
            lot.fees += fill.fees
            if tracked.remaining <= 0:
                lot.entry_done = True
                self._protect(lot)
        else:
            self._reduce(lot, fill, tracked.reason)
        for listener in self.fill_listeners:
            listener(fill)

    def on_status(self, order_id: str, status: OrderStatus, message: str = "") -> None:
        tracked = self._orders.get(order_id)
        if tracked is None:
            return
        logger.info("Order %s %s %s", order_id, status.value, message)
        lot = self.lots.get(tracked.lot_id)
        if lot is None:
            return
        if tracked.role is Role.ENTRY and not lot.entry_done:
            tracked.filled = tracked.order.quantity  # nothing more will fill
            lot.entry_done = True
            if lot.quantity > 0:  # partially filled: protect what we have
                self._protect(lot)
            elif lot.exit_quantity > 0:  # already closed by an exit
                self._close(lot)
            else:
                del self.lots[lot.lot_id]
        elif tracked.role is Role.PROTECT:
            lot.protective_ids.discard(order_id)
        elif tracked.role is Role.EXIT and lot.exit_order_id == order_id:
            lot.exit_order_id = None  # retried at the next bar close

    # --- bar close: time exits and kill switch ---------------------------------------------------------

    def on_bar_close(self, timestamp: pd.Timestamp) -> None:
        for lot in list(self.lots.values()):
            if lot.quantity <= 0:
                continue
            lot.bars_held += 1
            if lot.holding_bars is not None and lot.bars_held >= lot.holding_bars and lot.exit_order_id is None:
                self._exit(lot, "time")
        if self.risk.on_equity(timestamp, self.portfolio.equity) and self.risk.limits.flatten_on_halt:
            self.flatten("kill switch")

    def flatten(self, reason: str) -> None:
        """Cancel pending entries and close every open lot at market."""
        for lot in list(self.lots.values()):
            if not lot.entry_done:
                self.broker.cancel(lot.entry_order_id)
            if lot.quantity > 0 and lot.exit_order_id is None:
                self._exit(lot, reason)

    def open_trades(self, prices: dict[str, float], timestamp: pd.Timestamp) -> list[Trade]:
        """Open lots as trades marked at `prices` (end of a backtest); nothing is sent."""
        trades = []
        for lot in self.lots.values():
            if lot.quantity > 0 and lot.entry_time is not None and lot.symbol in prices:
                trades.append(self._trade(lot, lot.quantity, prices[lot.symbol], timestamp, "open"))
        return trades

    # --- internals -------------------------------------------------------------------------------------

    def _send(self, order: Order, role: Role, lot_id: str, reason: str = "") -> None:
        self._orders[order.order_id] = _Tracked(order, role, lot_id, reason)
        self.broker.submit(order)

    def _round(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _protect(self, lot: Lot) -> None:
        if lot.quantity <= 0 or lot.exit_order_id is not None:
            return
        limits, sign, exit_side = self.risk.limits, lot.side.sign, lot.side.opposite
        oca = f"oca-{lot.entry_order_id}"
        protective: list[tuple[Order, str]] = []
        if limits.stop_loss_pct is not None:
            stop = self._round(lot.entry_price * (1 - sign * limits.stop_loss_pct))
            protective.append(
                (Order(lot.symbol, exit_side, lot.quantity, lot.strategy_id, OrderType.STOP, stop_price=stop,
                       oca_group=oca), "stop_loss")
            )  # fmt: skip
        if limits.take_profit_pct is not None:
            target = self._round(lot.entry_price * (1 + sign * limits.take_profit_pct))
            protective.append(
                (Order(lot.symbol, exit_side, lot.quantity, lot.strategy_id, OrderType.LIMIT, limit_price=target,
                       oca_group=oca), "take_profit")
            )  # fmt: skip
        for order, reason in protective:
            lot.protective_ids.add(order.order_id)
            self._send(order, Role.PROTECT, lot.lot_id, reason)

    def _exit(self, lot: Lot, reason: str) -> None:
        for order_id in list(lot.protective_ids):
            self.broker.cancel(order_id)
        lot.protective_ids.clear()
        order = Order(lot.symbol, lot.side.opposite, lot.quantity, lot.strategy_id)
        lot.exit_order_id = order.order_id
        self._send(order, Role.EXIT, lot.lot_id, reason)

    def _reduce(self, lot: Lot, fill: Fill, reason: str) -> None:
        used = min(fill.quantity, lot.quantity)
        if used:
            lot.quantity -= used
            lot.exit_quantity += used
            lot.exit_value += used * fill.price
            lot.fees += fill.fees
            lot.exit_time, lot.exit_reason = fill.timestamp, reason
        if fill.quantity > used:
            logger.error("Fill over-closes %s by %d shares: opening a correction", lot.lot_id, fill.quantity - used)
            self._open_correction(lot.symbol, fill.order.side, fill.quantity - used, fill)
        if lot.quantity == 0 and lot.entry_done:
            self._close(lot)

    def _close(self, lot: Lot) -> None:
        for order_id in lot.protective_ids:
            self.broker.cancel(order_id)  # the OCA sibling, if the broker has not cancelled it yet
        exit_order = self._orders.get(lot.exit_order_id or "")
        if exit_order is not None and exit_order.remaining > 0:
            self.broker.cancel(exit_order.order.order_id)
        del self.lots[lot.lot_id]
        assert lot.exit_time is not None
        self.trades.append(
            self._trade(lot, lot.exit_quantity, lot.exit_value / lot.exit_quantity, lot.exit_time, lot.exit_reason)
        )

    @staticmethod
    def _trade(lot: Lot, quantity: int, exit_price: float, exit_time: pd.Timestamp, reason: str) -> Trade:
        assert lot.entry_time is not None
        return Trade(
            strategy_id=lot.strategy_id,
            symbol=lot.symbol,
            side=lot.side,
            quantity=quantity,
            entry_time=lot.entry_time,
            exit_time=exit_time,
            entry_price=lot.entry_price,
            exit_price=exit_price,
            fees=lot.fees,
            bars_held=lot.bars_held,
            reason=reason,
        )

    def _open_correction(self, symbol: str, side: Side, quantity: int, fill: Fill) -> None:
        """Unintended shares (over-fill): track them as a lot and close it immediately."""
        lot = Lot(next(self._lot_ids), fill.order.strategy_id, symbol, side, holding_bars=0, entry_done=True)
        lot.quantity = lot.entry_quantity = quantity
        lot.entry_price, lot.entry_time = fill.price, fill.timestamp
        self.lots[lot.lot_id] = lot
        self._exit(lot, "correction")
