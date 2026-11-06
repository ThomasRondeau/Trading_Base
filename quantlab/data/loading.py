"""Bar loading and train / test split shared by the CLI and the UI."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.pairs import combine_symbols
from quantlab.data.schema import validate_ohlcv

SOURCES = ("yfinance", "massive", "ibkr", "file")


@contextmanager
def ibkr_client(auto_reconnect: bool = False, account: str | None = None) -> Iterator[Any]:
    """A connected `IBKRClient` (settings from the environment, `account` = paper / live), disconnected on exit."""
    from quantlab.config import IBKRSettings
    from quantlab.execution.ibkr.client import IBKRClient

    client = IBKRClient(IBKRSettings.from_env(account), auto_reconnect=auto_reconnect)
    client.connect()
    try:
        yield client
    finally:
        client.disconnect()


def read_bars_file(path: str | Path, symbols: Sequence[str] = ()) -> pd.DataFrame:
    """Bars from a parquet / csv file; multi-symbol files (`close_<SYMBOL>` columns) are returned as is."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if symbols and any(str(c).endswith(f"_{symbols[-1]}") for c in raw.columns):
        return raw
    return validate_ohlcv(raw)


def load_bars(
    symbols: Sequence[str], source: str, days: int, file: str | Path | None = None, ibkr_account: str | None = None
) -> pd.DataFrame:
    """Bars of `symbols` over the last `days` calendar days (several symbols: the multi-symbol layout)."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; available: {', '.join(SOURCES)}.")
    if source == "file":
        if not file:
            raise ValueError("The file source needs a file path.")
        return read_bars_file(file, symbols)

    from quantlab.data.data_manager import DataManager

    manager = DataManager()
    frames: dict[str, pd.DataFrame] = {}
    if source == "ibkr":
        with ibkr_client(account=ibkr_account) as client:
            for symbol in symbols:
                frames[symbol] = manager.collect_data_ibkr(client, symbol, days, 0)
    else:
        collect = manager.collect_data_yfinance if source == "yfinance" else manager.collect_data_massive
        for symbol in symbols:
            frames[symbol] = collect(symbol, days, 0)
    for symbol, frame in frames.items():
        if frame.empty:
            raise ValueError(f"No {source} data for {symbol}.")
    return combine_symbols(frames)


def split(data: pd.DataFrame, test_days: float, test_fraction: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training / test blocks: the last `test_fraction` of the bars, or the last `test_days` calendar days."""
    if test_fraction is not None:
        cut = int(len(data) * (1 - test_fraction))
        return data.iloc[:cut], data.iloc[cut:]
    boundary = data.index[-1] - pd.Timedelta(days=test_days)
    return data[data.index <= boundary], data[data.index > boundary]
