"""Massive API client (https://massive.com/docs/rest/): intraday OHLCV history."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from quantlab.data.schema import validate_ohlcv

try:
    from dotenv import load_dotenv

    load_dotenv()  # Load an optional .env file from the working directory
except ImportError:  # python-dotenv is optional
    pass

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "MASSIVE_API_KEY"

# Massive aggregate fields -> project column names
_COLUMN_MAPPING = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}


class MassiveAPIClient:
    """
    Fetches historical 1-minute bars.

    The API key is read from the MASSIVE_API_KEY environment variable (or a .env file)
    unless it is passed explicitly.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved_key:
            raise RuntimeError(
                f"Massive API key not found: set the {API_KEY_ENV_VAR} environment variable "
                f"(or add it to a .env file, see .env.example)."
            )
        self.api_key: str = resolved_key
        self.base_url = "https://api.massive.com"

    def fetch_data(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """
        1-minute history for `symbol`, from `days_back` days ago to `until` days ago (0 = today).

        :return: OHLCV DataFrame in the `validate_ohlcv` format (UTC Datetime index); empty on failure.
        """
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=until)
        from_date_str = start_date.strftime("%Y-%m-%d")
        to_date_str = end_date.strftime("%Y-%m-%d")

        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{from_date_str}/{to_date_str}"
        params: dict[str, Any] = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}

        logger.info("Downloading %s from Massive API: %s to %s", symbol, from_date_str, to_date_str)
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # Never log the exception itself: its message contains the URL, hence the API key.
            status = e.response.status_code if e.response is not None else None
            details = e.response.text[:200] if e.response is not None else ""
            logger.error("Massive API request failed (%s, HTTP %s) %s", type(e).__name__, status, details)
            return pd.DataFrame()

        data = response.json()
        if not data.get("results"):
            logger.warning("No Massive data for %s over this period.", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df = df.rename(columns={k: v for k, v in _COLUMN_MAPPING.items() if k in df.columns})
        if "timestamp" not in df.columns:
            raise ValueError(f"Unexpected Massive response: columns {list(df.columns)}")

        # Massive timestamps are epoch milliseconds, UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = validate_ohlcv(df[["timestamp", "open", "high", "low", "close", "volume"]], datetime_column="timestamp")

        logger.info("Massive API: %d 1-minute bars received.", len(df))
        return df
