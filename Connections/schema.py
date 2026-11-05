"""Single OHLCV contract shared by every data source (yfinance, Massive, IBKR, CSV cache)."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
INDEX_NAME = "Datetime"


def validate_ohlcv(df: pd.DataFrame, *, datetime_column: str | None = None) -> pd.DataFrame:
    """
    Normalize raw bars to the project contract and return a new DataFrame:

    - tz-aware UTC ``DatetimeIndex`` named ``Datetime`` (naive timestamps are assumed to be UTC),
      sorted, without duplicates;
    - lowercase ``open, high, low, close, volume`` columns as float (extra columns are kept);
    - rows without a close price are dropped.

    :param datetime_column: column holding the timestamps when they are not already the index.
    """
    if df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS), index=pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME))

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):  # yfinance returns (field, ticker) columns
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).lower() for c in out.columns]

    if datetime_column is not None:
        out = out.set_index(datetime_column.lower())
    elif not isinstance(out.index, pd.DatetimeIndex):
        for candidate in ("datetime", "date", "timestamp"):
            if candidate in out.columns:
                out = out.set_index(candidate)
                break

    index = pd.DatetimeIndex(pd.to_datetime(out.index))
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out.index = index.rename(INDEX_NAME)

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns {missing}; got {list(out.columns)}")
    out[list(OHLCV_COLUMNS)] = out[list(OHLCV_COLUMNS)].astype(float)

    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.dropna(subset=["close"])
