"""Base technical indicators: SMA, RSI, ATR, MACD, Bollinger Bands, volume, VWAP, z-score, ADX."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.features.base import FeatureInterface, ResolvedDependencies, require_columns


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


class SMA(FeatureInterface):
    """Simple moving average of the close."""

    def __init__(self, data: pd.DataFrame | None = None, window: int = 14) -> None:
        self.window = window
        self.data = data if data is not None else _empty()
        self.sma: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["close"], "SMA")
        self.sma = self.data["close"].rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["close"], "SMA")
        self.data = new_data

    def clear_cache(self) -> None:
        self.sma = None
        self.data = _empty()


class RSI(FeatureInterface):
    """Relative Strength Index with simple-moving-average smoothing."""

    def __init__(self, data: pd.DataFrame | None = None, window: int = 14) -> None:
        self.data = data if data is not None else _empty()
        self.window = window
        self.rsi: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["close"], "RSI")
        delta = self.data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["close"], "RSI")
        self.data = new_data

    def clear_cache(self) -> None:
        self.rsi = None
        self.data = _empty()


class ATR(FeatureInterface):
    """Average True Range (simple moving average of the true range)."""

    def __init__(self, data: pd.DataFrame | None = None, window: int = 14) -> None:
        self.data = data if data is not None else _empty()
        self.window = window
        self.true_range: pd.Series | None = None
        self.atr: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["high", "low", "close"], "ATR")
        high_low = self.data["high"] - self.data["low"]
        high_close_prev = (self.data["high"] - self.data["close"].shift(1)).abs()
        low_close_prev = (self.data["low"] - self.data["close"].shift(1)).abs()
        self.true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.atr = self.true_range.rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["high", "low", "close"], "ATR")
        self.data = new_data

    def clear_cache(self) -> None:
        self.true_range = None
        self.atr = None
        self.data = _empty()


class MACD(FeatureInterface):
    """MACD line, signal line and histogram (EMA-based)."""

    def __init__(
        self, data: pd.DataFrame | None = None, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ) -> None:
        if short_window >= long_window:
            raise ValueError(f"MACD: short_window ({short_window}) must be < long_window ({long_window}).")
        self.data = data if data is not None else _empty()
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.macd: pd.Series | None = None
        self.signal_line: pd.Series | None = None
        self.macd_histogram: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["close"], "MACD")
        short_ema = self.data["close"].ewm(span=self.short_window, adjust=False).mean()
        long_ema = self.data["close"].ewm(span=self.long_window, adjust=False).mean()
        self.macd = short_ema - long_ema
        self.signal_line = self.macd.ewm(span=self.signal_window, adjust=False).mean()
        self.macd_histogram = self.macd - self.signal_line

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["close"], "MACD")
        self.data = new_data

    def clear_cache(self) -> None:
        self.macd = None
        self.signal_line = None
        self.macd_histogram = None
        self.data = _empty()


class BollingerBands(FeatureInterface):
    def __init__(self, data: pd.DataFrame | None = None, window: int = 20, num_std_dev: float = 2) -> None:
        self.data = data if data is not None else _empty()
        self.window = window
        self.num_std_dev = num_std_dev
        self.sma: pd.Series | None = None
        self.std: pd.Series | None = None
        self.upper_band: pd.Series | None = None
        self.lower_band: pd.Series | None = None
        self.width: pd.Series | None = None
        self.position: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        close = self.data["close"]
        self.sma = close.rolling(window=self.window).mean()
        self.std = close.rolling(window=self.window).std()
        self.upper_band = self.sma + (self.num_std_dev * self.std)
        self.lower_band = self.sma - (self.num_std_dev * self.std)
        self.width = self.upper_band - self.lower_band
        self.position = (close - self.lower_band) / (self.upper_band - self.lower_band)

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["close"], "BollingerBands")
        self.data = new_data

    def clear_cache(self) -> None:
        self.sma = None
        self.std = None
        self.upper_band = None
        self.lower_band = None
        self.width = None
        self.position = None
        self.data = _empty()


class Volume(FeatureInterface):
    """Raw volume, its moving average and the volume / moving-average ratio."""

    def __init__(self, data: pd.DataFrame | None = None, window: int = 14) -> None:
        self.window = window
        self.data = data if data is not None else _empty()
        self.volume: pd.Series | None = None
        self.volume_ma: pd.Series | None = None
        self.volume_ratio: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["volume"], "Volume")
        self.volume = self.data["volume"]
        self.volume_ma = self.data["volume"].rolling(window=self.window).mean()
        self.volume_ratio = self.data["volume"] / self.volume_ma

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["volume"], "Volume")
        self.data = new_data

    def clear_cache(self) -> None:
        self.volume = None
        self.volume_ma = None
        self.volume_ratio = None
        self.data = _empty()


class VWAP(FeatureInterface):
    """
    Session VWAP with volume-weighted standard-deviation bands.
    The VWAP restarts at each session (calendar date in `session_tz`, the exchange time zone).
    """

    def __init__(
        self, data: pd.DataFrame | None = None, std_multiplier: float = 3, session_tz: str = "America/New_York"
    ) -> None:
        self.data = data if data is not None else _empty()
        self.session_tz = session_tz
        self.std_multiplier = std_multiplier
        self.point_of_control: pd.Series | None = None
        self.superior_limit: pd.Series | None = None
        self.far_superior_limit: pd.Series | None = None
        self.far_inferior_limit: pd.Series | None = None
        self.inferior_limit: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        require_columns(self.data, ["close", "high", "low", "volume"], "VWAP")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("VWAP: a DatetimeIndex is required to reset the VWAP at each session.")

        index = self.data.index if self.data.index.tz is not None else self.data.index.tz_localize("UTC")
        session = pd.Series(index.tz_convert(self.session_tz).date, index=self.data.index)

        typical_price = (self.data["close"] + self.data["high"] + self.data["low"]) / 3
        volume = self.data["volume"]

        cum_volume = volume.groupby(session).cumsum()
        vwap = (typical_price * volume).groupby(session).cumsum() / cum_volume
        self.point_of_control = vwap

        # Volume-weighted standard deviation of the typical price around the VWAP
        mean_sq = (typical_price**2 * volume).groupby(session).cumsum() / cum_volume
        std = np.sqrt((mean_sq - vwap**2).clip(lower=0))

        self.far_superior_limit = vwap + (std * self.std_multiplier * 2)
        self.superior_limit = vwap + (std * self.std_multiplier)
        self.inferior_limit = vwap - (std * self.std_multiplier)
        self.far_inferior_limit = vwap - (std * self.std_multiplier * 2)

    def update(self, new_data: pd.DataFrame) -> None:
        require_columns(new_data, ["close", "high", "low", "volume"], "VWAP")
        self.data = new_data

    def clear_cache(self) -> None:
        self.point_of_control = None
        self.superior_limit = None
        self.far_superior_limit = None
        self.far_inferior_limit = None
        self.inferior_limit = None
        self.data = _empty()


class ZScore(FeatureInterface):
    """Rolling z-score of the `ratio` column (pair trading)."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.data = _empty()
        self.zscore: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        ratio = self.data["ratio"]
        mean = ratio.rolling(window=self.window).mean()
        std = ratio.rolling(window=self.window).std()
        self.zscore = (ratio - mean) / std

    def update(self, new_data: pd.DataFrame) -> None:
        self.data = new_data

    def clear_cache(self) -> None:
        self.zscore = None
        self.data = _empty()


class ADX(FeatureInterface):
    """Average Directional Index (rolling-sum smoothing)."""

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.data = _empty()
        self.adx: pd.Series | None = None

    def calculate(self, resolved_dependencies: ResolvedDependencies | None = None) -> None:
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr_smooth = tr.rolling(window=self.window).sum()
        plus_dm_smooth = plus_dm.rolling(window=self.window).sum()
        minus_dm_smooth = minus_dm.rolling(window=self.window).sum()

        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        self.adx = dx.rolling(window=self.window).mean()

    def update(self, new_data: pd.DataFrame) -> None:
        self.data = new_data

    def clear_cache(self) -> None:
        self.adx = None
        self.data = _empty()
