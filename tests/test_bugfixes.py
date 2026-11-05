"""Regression tests for the audit bugs A1-A12 (see the refactor plan). One block per bug."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import quantlab.data.data_manager as data_manager_module
import quantlab.data.massive as massive_module
import quantlab.models.arima as arima_module
import quantlab.models.base as model_base_module
from quantlab.data.data_manager import DataManager
from quantlab.data.massive import MassiveAPIClient
from quantlab.data.schema import validate_ohlcv
from quantlab.features.feature_id import FeatureID
from quantlab.features.technical import ADX, MACD, VWAP
from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.trend_following import TrendFollowingStrategy
from quantlab.targets.forward import TargetOpportunity

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def aapl() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "aapl_1m.parquet")


def ohlcv(n: int, start: str = "2026-01-05 14:30", freq: str = "1min", tz: str | None = "UTC") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq=freq, tz=tz)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        },
        index=index,
    )


# --- A4: one OHLCV contract -------------------------------------------------------------------------------


def test_validate_ohlcv_normalizes_yfinance_shape() -> None:
    raw = ohlcv(5, tz="America/New_York").rename(columns=str.title)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
    raw = raw.iloc[[0, 2, 1, 3, 4, 4]]  # unsorted + duplicate

    out = validate_ohlcv(raw)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "Datetime"
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert (out.dtypes == np.float64).all()


def test_validate_ohlcv_naive_is_utc_and_reads_datetime_column() -> None:
    raw = ohlcv(3, tz=None).reset_index(names="Datetime")
    out = validate_ohlcv(raw, datetime_column="Datetime")
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2026-01-05 14:30", tz="UTC")


def test_validate_ohlcv_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="volume"):
        validate_ohlcv(ohlcv(3).drop(columns="volume"))


def test_yfinance_keeps_datetime_index_and_date_convention(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_download(symbol: str, **kwargs: Any) -> pd.DataFrame:
        calls.update(kwargs)
        return ohlcv(10, tz="America/New_York").rename(columns=str.title)

    monkeypatch.setattr(data_manager_module.yf, "download", fake_download)
    df = DataManager().collect_data_yfinance("AAPL", days_back=50, until=5)

    assert isinstance(df.index, pd.DatetimeIndex)  # used to be a RangeIndex -> Sharpe on 1970 dates
    today = datetime.now()
    assert calls["start"] == (today - timedelta(days=50)).strftime("%Y-%m-%d")
    assert calls["end"] == (today - timedelta(days=5)).strftime("%Y-%m-%d")


# --- A5: collect_data_massive ------------------------------------------------------------------------------


def test_collect_data_massive_uses_client_and_does_not_touch_ibkr_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    class FakeClient:
        def fetch_data(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
            calls.update(symbol=symbol, days_back=days_back, until=until)
            return validate_ohlcv(ohlcv(4))

    monkeypatch.setattr(data_manager_module, "MassiveAPIClient", FakeClient)
    manager = DataManager()
    manager.cache_dir = tmp_path

    df = manager.collect_data_massive("^GSPC", days_back=30, until=2)

    assert calls == {"symbol": "I:SPX", "days_back": 30, "until": 2}
    assert len(df) == 4
    assert list(tmp_path.iterdir()) == []


def test_massive_fetch_data_dates_and_utc_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            t0 = int(pd.Timestamp("2026-01-05 14:30", tz="UTC").timestamp() * 1000)
            return {"results": [{"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "t": t0 + 60_000 * i} for i in range(3)]}

    def fake_get(url: str, params: dict[str, Any]) -> FakeResponse:
        seen["url"] = url
        return FakeResponse()

    monkeypatch.setattr(massive_module.requests, "get", fake_get)
    df = MassiveAPIClient(api_key="test").fetch_data("AAPL", days_back=10, until=3)

    today = datetime.now()
    expected = f"/{(today - timedelta(days=10)):%Y-%m-%d}/{(today - timedelta(days=3)):%Y-%m-%d}"
    assert seen["url"].endswith(expected)
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2026-01-05 14:30", tz="UTC")


# --- A1: MACD windows --------------------------------------------------------------------------------------


def test_macd_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError):
        MACD(short_window=26, long_window=12)


@pytest.mark.parametrize("factory", [MLHybridStrategy, CascadeStrategy])
def test_strategies_register_macd_12_26(factory: Any, aapl: pd.DataFrame) -> None:
    strategy = factory()
    strategy.train(aapl.iloc[:600])
    macd = strategy.feature_manager._indicators[FeatureID.MACD]
    assert (macd.short_window, macd.long_window) == (12, 26)


# --- A2: ADX alignment -------------------------------------------------------------------------------------


def test_adx_keeps_data_index() -> None:
    data = ohlcv(100)
    adx = ADX(window=14)
    adx.update(data)
    adx.calculate({})
    assert adx.adx.index.equals(data.index)
    assert adx.adx.notna().sum() > 0
    assert adx.adx.dropna().between(0, 100).all()


# --- A9: VWAP resets per session ---------------------------------------------------------------------------


def test_vwap_resets_each_session() -> None:
    day1 = ohlcv(30, start="2026-01-05 14:30")
    day2 = ohlcv(30, start="2026-01-06 14:30")
    data = pd.concat([day1, day2])
    vwap = VWAP(data)
    vwap.calculate()

    typical = (data["close"] + data["high"] + data["low"]) / 3
    first_bar_day2 = day2.index[0]
    assert vwap.point_of_control[first_bar_day2] == pytest.approx(typical[first_bar_day2])
    assert vwap.superior_limit[first_bar_day2] == pytest.approx(typical[first_bar_day2])  # zero dispersion


# --- A3: ARIMA ---------------------------------------------------------------------------------------------


class FakeArima:
    orders: ClassVar[list[tuple[int, int, int]]] = []
    forecast_delta = 0.0

    def __init__(self, series: np.ndarray, order: tuple[int, int, int]) -> None:
        FakeArima.orders.append(order)
        self.last = float(series[-1])

    def fit(self) -> FakeArima:
        return self

    def forecast(self) -> np.ndarray:
        return np.array([self.last + FakeArima.forecast_delta])


def _arima_orders(monkeypatch: pytest.MonkeyPatch, delta: float, aapl: pd.DataFrame) -> list[Any]:
    FakeArima.orders = []
    FakeArima.forecast_delta = delta
    monkeypatch.setattr(arima_module, "ARIMA", FakeArima)
    strategy = ArimaStrategy(order=(3, 1, 1))
    strategy.train(aapl.iloc[:200])
    strategy.update_status(True)
    orders = [o for _, bar in aapl.iloc[200:210].iterrows() for o in strategy.on_bar(bar)]
    assert set(FakeArima.orders) == {(3, 1, 1)}  # order used to be ignored
    return orders


def test_arima_does_not_trade_without_edge(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    assert _arima_orders(monkeypatch, delta=1e-6, aapl=aapl) == []  # used to trade every bar


def test_arima_trades_when_edge_exceeds_costs(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    orders = _arima_orders(monkeypatch, delta=1.0, aapl=aapl)
    assert len(orders) == 10
    assert {o.side for o in orders} == {"BUY"}


# --- A7: cross-validation uses the configured estimator ----------------------------------------------------


def test_cross_validate_clones_configured_model(monkeypatch: pytest.MonkeyPatch, aapl: pd.DataFrame) -> None:
    depths: list[int] = []
    real_clone = model_base_module.clone

    def spy_clone(estimator: Any) -> Any:
        depths.append(estimator.get_params()["max_depth"])
        return real_clone(estimator)

    monkeypatch.setattr(model_base_module, "clone", spy_clone)
    strategy = MLHybridStrategy(max_depth=7)
    strategy.train(aapl.iloc[:600])
    strategy.direction_model.cross_validate(n_splits=2)
    assert depths == [7, 7]  # used to be a hard-coded max_depth=4


# --- A8: opportunity threshold -----------------------------------------------------------------------------


def test_opportunity_threshold_is_used() -> None:
    data = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.1, 100.0, 100.0, 100.0]})
    loose, strict = TargetOpportunity(threshold_pct=0.05, horizon=3), TargetOpportunity(threshold_pct=0.5, horizon=3)
    for target in (loose, strict):
        target.update(data, training_data_shift=0)
        target.calculate()
    assert loose.target.iloc[0] == 1  # +0.1 % move > 0.05 %
    assert strict.target.iloc[0] == 0  # +0.1 % move < 0.5 %


# --- A10: trend-following feature IDs ----------------------------------------------------------------------


def test_trend_following_ids_match_windows(aapl: pd.DataFrame) -> None:
    strategy = TrendFollowingStrategy(sma_fast=10, sma_slow=50)
    strategy.train(aapl.iloc[:300])
    indicators = strategy.feature_manager._indicators
    assert indicators[FeatureID.SMA_FAST].window == 10
    assert indicators[FeatureID.SMA_SLOW].window == 50
    assert FeatureID.SMA_5 not in indicators


# --- A11: bounded pair-trading history ---------------------------------------------------------------------


def test_pair_trading_history_is_bounded() -> None:
    pair = pd.read_parquet(FIXTURES / "aapl_msft_2m.parquet")
    strategy = PairTradingStrategy()
    strategy.train(pair.iloc[:200])
    strategy.update_status(True)
    for _, bar in pair.iloc[200:600].iterrows():
        strategy.on_bar(bar)
    assert len(strategy.current_data) <= strategy.max_history


# --- A12: TradingApp singleton -----------------------------------------------------------------------------


def test_no_direct_trading_app_construction() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.py")
        if "tests" not in path.parts
        and ".git" not in path.parts
        and re.search(r"\bTradingApp\(\)", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
