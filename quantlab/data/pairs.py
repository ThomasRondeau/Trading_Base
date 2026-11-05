"""Multi-symbol bar layout shared by the backtests and the live IBKR feed."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantlab.data.schema import OHLCV_COLUMNS


def combine_symbols(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Bars of several symbols aligned on common timestamps: `<field>_<SYMBOL>` columns for every symbol,
    plus the plain OHLCV columns of the first one (read by single-symbol code). One symbol: unchanged.
    """
    symbols = list(frames)
    if len(symbols) == 1:
        return frames[symbols[0]]
    data = pd.concat({s: frames[s][list(OHLCV_COLUMNS)] for s in symbols}, axis=1, join="inner")
    data.columns = [f"{column[1]}_{column[0]}" for column in data.columns]
    for field in OHLCV_COLUMNS:
        data[field] = data[f"{field}_{symbols[0]}"]
    return data
