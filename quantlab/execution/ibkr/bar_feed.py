"""
Live IBKR bars for the `StrategyRunner`.

IBKR streams 5-second TRADES bars; `BarAggregator` builds strategy bars of `bar_seconds` aligned on
multiples of the period (UTC epoch), stamped with their start time like IBKR historical bars. A bar is
emitted as soon as its last 5-second bar arrives (or when a later one shows it is over), so the
strategy reacts within a few milliseconds of the period end.

Multi-symbol feeds emit one bar per period with `<field>_<SYMBOL>` columns plus the plain columns of
the first symbol (the layout of the backtest pair data). A symbol without trades in a period is
forward-filled from its last close with zero volume.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantlab.data.schema import OHLCV_COLUMNS
from quantlab.execution.ibkr.client import IBKRClient, IBKRRequestError, stock

logger = logging.getLogger(__name__)

REALTIME_BAR_SECONDS = 5


@dataclass
class _Bar:
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def values(self) -> dict[str, float]:
        return {"open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume}


class BarAggregator:
    def __init__(self, bar_seconds: int = 60) -> None:
        if bar_seconds % REALTIME_BAR_SECONDS:
            raise ValueError(f"bar_seconds must be a multiple of {REALTIME_BAR_SECONDS} (got {bar_seconds}).")
        self.bar_seconds = bar_seconds
        self._current: _Bar | None = None

    def add(self, time: int, open_: float, high: float, low: float, close: float, volume: float) -> list[_Bar]:
        """Add a 5-second bar starting at `time` (epoch s); returns the bars it completes."""
        completed: list[_Bar] = []
        start = time - time % self.bar_seconds
        current = self._current
        if current is not None and start != current.start:
            completed.append(current)
            current = None
        if current is None:
            current = _Bar(start, open_, high, low, close, volume)
        else:
            current.high, current.low = max(current.high, high), min(current.low, low)
            current.close, current.volume = close, current.volume + volume
        if time + REALTIME_BAR_SECONDS >= start + self.bar_seconds:  # last 5-s slot of the period
            completed.append(current)
            current = None
        self._current = current
        return completed


class IBKRBarFeed:
    def __init__(
        self,
        client: IBKRClient,
        symbols: Sequence[str],
        bar_seconds: int = 60,
        use_rth: bool = False,
        heartbeat: float = 1.0,
    ) -> None:
        if not symbols:
            raise ValueError("At least one symbol is required.")
        self.client = client
        self.symbols = list(symbols)
        self.bar_seconds = bar_seconds
        self.use_rth = use_rth
        self.heartbeat = heartbeat
        self._queue: queue.SimpleQueue[dict[str, Any] | IBKRRequestError] = queue.SimpleQueue()
        self._aggregators = {symbol: BarAggregator(bar_seconds) for symbol in self.symbols}
        self._pending: dict[int, dict[str, _Bar]] = {}  # period start -> symbol -> bar
        self._last: dict[str, _Bar] = {}
        self._last_emitted: pd.Timestamp | None = None
        self._stop = threading.Event()

    # --- warm-up ---------------------------------------------------------------------------------------

    def history(self, duration: str = "2 D") -> pd.DataFrame:
        """Recent historical bars in the live layout, to `fit` / warm the strategies before `run`."""
        minutes, seconds = divmod(self.bar_seconds, 60)
        bar_size = f"{self.bar_seconds} secs" if seconds else ("1 min" if minutes == 1 else f"{minutes} mins")
        frames = {
            symbol: self.client.historical_bars(
                stock(symbol), duration=duration, bar_size=bar_size, use_rth=self.use_rth
            )
            for symbol in self.symbols
        }
        if len(self.symbols) == 1:
            data = frames[self.symbols[0]]
        else:
            data = pd.concat({s: f[list(OHLCV_COLUMNS)] for s, f in frames.items()}, axis=1, join="inner")
            data.columns = [f"{column[1]}_{column[0]}" for column in data.columns]
            for field in OHLCV_COLUMNS:
                data[field] = data[f"{field}_{self.symbols[0]}"]
        if not data.empty:
            # The newest historical bar may still be forming: drop it, the live feed will deliver it complete
            data = data.iloc[:-1]
            self._last_emitted = data.index[-1]
        return data

    # --- live ------------------------------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def __iter__(self) -> Iterator[pd.Series | None]:
        self._stop.clear()
        req_ids = [
            self.client.subscribe_realtime_bars(
                stock(symbol), self._queue.put, use_rth=self.use_rth, on_error=self._queue.put
            )
            for symbol in self.symbols
        ]
        try:
            while not self._stop.is_set():
                try:
                    raw = self._queue.get(timeout=self.heartbeat)
                except queue.Empty:
                    yield None
                    continue
                if isinstance(raw, IBKRRequestError):
                    raise raw  # no live data: never wait silently
                yield from self._on_realtime_bar(raw)
        finally:
            for req_id in req_ids:
                self.client.cancel_realtime_bars(req_id)

    def _on_realtime_bar(self, raw: dict[str, Any]) -> Iterator[pd.Series]:
        symbol = raw["symbol"]
        for bar in self._aggregators[symbol].add(
            raw["time"], raw["open"], raw["high"], raw["low"], raw["close"], raw["volume"]
        ):
            self._pending.setdefault(bar.start, {})[symbol] = bar
        yield from self._emit_ready()

    def _emit_ready(self) -> Iterator[pd.Series]:
        for start in sorted(self._pending):
            bars = self._pending[start]
            later = any(s > start for s in self._pending)
            if len(bars) < len(self.symbols) and not later:
                return  # wait for the other symbols
            del self._pending[start]
            for symbol in self.symbols:
                if symbol not in bars and symbol in self._last:  # no trade in the period: forward-fill
                    close = self._last[symbol].close
                    bars[symbol] = _Bar(start, close, close, close, close, 0.0)
            if len(bars) < len(self.symbols):
                logger.warning("Dropping the %s bar: no price yet for %s", start, set(self.symbols) - set(bars))
                continue
            self._last.update(bars)
            timestamp = pd.Timestamp(start, unit="s", tz="UTC")
            if self._last_emitted is not None and timestamp <= self._last_emitted:
                continue  # already covered by the warm-up history
            self._last_emitted = timestamp
            yield self._to_series(timestamp, bars)

    def _to_series(self, timestamp: pd.Timestamp, bars: dict[str, _Bar]) -> pd.Series:
        if len(self.symbols) == 1:
            return pd.Series(bars[self.symbols[0]].values(), name=timestamp)
        values = {f"{field}_{symbol}": v for symbol in self.symbols for field, v in bars[symbol].values().items()}
        values.update(bars[self.symbols[0]].values())
        return pd.Series(values, name=timestamp)
