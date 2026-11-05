"""Historical data collection from IBKR (with CSV cache), yfinance and Massive."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from quantlab.data.massive import MassiveAPIClient
from quantlab.data.schema import validate_ohlcv
from quantlab.execution.ibkr.client import IBKRClient, stock

logger = logging.getLogger(__name__)

# Index tickers use a different naming on Massive.
MASSIVE_TICKER_ALIASES = {"^GSPC": "I:SPX", "SPX": "I:SPX", "^FTSE": "I:FTSE", "FTSE": "I:FTSE"}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


class DataManager:
    """
    All methods share the window convention `(symbol, days_back, until)`:
    from `days_back` days ago to `until` days ago.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _get_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_historical_data.csv"

    def _load_cache(self, symbol: str) -> pd.DataFrame:
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            return validate_ohlcv(pd.read_csv(cache_path), datetime_column="Datetime")
        return pd.DataFrame()

    def _save_cache(self, symbol: str, data: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._get_cache_path(symbol))  # index kept: it holds the timestamps

    def collect_data_ibkr(self, client: IBKRClient, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """1-minute regular-hours IBKR bars, one request per missing day; `client` must be connected."""
        logger.info("Collecting IBKR data for %s over %d days...", symbol, days_back)

        cached_data = self._load_cache(symbol)
        # Cache hits are checked per calendar date
        cached_dates = set(cached_data.index.date.tolist()) if not cached_data.empty else set()

        contract = stock(symbol)
        all_data = [cached_data] if not cached_data.empty else []
        new_data_fetched = False

        for day in range(until + 1, days_back + 1):
            target_date = datetime.now() - timedelta(days=day)
            target_date_only = target_date.date()
            try:
                if target_date_only in cached_dates:
                    logger.debug("Cache hit for %s", target_date_only)
                    continue

                date_str = target_date.strftime("%Y%m%d 20:00:00 US/Eastern")
                daily_data = client.historical_bars(contract, end=date_str, duration="1 D", use_rth=True)

                if not daily_data.empty:
                    all_data.append(daily_data)
                    new_data_fetched = True
                    logger.info("Data received for %s: %d bars", target_date_only, len(daily_data))
                else:
                    logger.warning("No data for %s", target_date_only)

            except Exception:
                logger.exception("Failed to fetch data for %s", target_date_only)
                continue

        if not all_data:
            raise ValueError("No data could be retrieved")

        consolidated_data = validate_ohlcv(pd.concat([validate_ohlcv(d) for d in all_data]))

        if new_data_fetched:
            self._save_cache(symbol, consolidated_data)
            logger.info("Cache updated: %s", self._get_cache_path(symbol))

        # Keep the requested window only
        min_date = (datetime.now() - timedelta(days=days_back)).date()
        max_date = (datetime.now() - timedelta(days=until)).date()
        consolidated_data = consolidated_data[
            (consolidated_data.index.date >= min_date) & (consolidated_data.index.date <= max_date)
        ]

        logger.info("Consolidated data: %d bars over %d days", len(consolidated_data), days_back - until)
        return consolidated_data

    def collect_data_yfinance(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """2-minute bars (yfinance serves ~60 days of intraday history)."""
        logger.info("Collecting yfinance data for %s from D-%d to D-%d...", symbol, days_back, until)
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=until)
        df = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="2m",
            auto_adjust=False,
            progress=False,
        )
        df = validate_ohlcv(df)
        logger.info("yfinance data received: %d bars", len(df))
        return df

    def collect_data_massive(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """1-minute bars from Massive API."""
        ticker = MASSIVE_TICKER_ALIASES.get(symbol, symbol)
        logger.info("Collecting Massive data for %s from D-%d to D-%d...", ticker, days_back, until)
        df = MassiveAPIClient().fetch_data(ticker, days_back=days_back, until=until)
        if df.empty:
            logger.warning("No Massive data received for %s", ticker)
        return df
