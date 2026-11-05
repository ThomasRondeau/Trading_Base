"""
Characterization tests: freeze the current backtest metrics of every strategy on fixed data.

They guard the refactoring steps: a pure restructuring must reproduce these numbers exactly.
When a change is *meant* to alter results (bug fix, engine change), regenerate the baseline
deliberately and review the diff:

    UPDATE_BASELINE=1 py -3.10 -m pytest tests/test_characterization.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from Engines.BacktestEngine import BacktestEngine
from Strategies.ArimaStrategy import ArimaStrategy
from Strategies.CascadeStrategy import CascadeStrategy
from Strategies.MeanReversionStrategy import MeanReversionStrategy
from Strategies.MLHybridStrategy import MLHybridStrategy
from Strategies.PairTradingStrategy import PairTradingStrategy
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Strategies.TrendFollowingStrategy import TrendFollowingStrategy

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE_PATH = FIXTURES / "baseline_metrics.json"
UPDATE = os.environ.get("UPDATE_BASELINE") == "1"


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / name)


# name -> (fixture, factory, train_bars, backtest_bars)
CASES: dict[str, tuple[str, Callable[[], TradingStrategyInterface], int, int]] = {
    "trend_following": ("aapl_1m.parquet", TrendFollowingStrategy, 4000, 1000),
    "mean_reversion": ("aapl_1m.parquet", MeanReversionStrategy, 4000, 1000),
    "ml_hybrid": ("aapl_1m.parquet", MLHybridStrategy, 4000, 1000),
    "cascade": ("aapl_1m.parquet", CascadeStrategy, 4000, 1000),
    "arima": ("aapl_1m.parquet", ArimaStrategy, 4000, 200),  # ARIMA refits every bar: keep it short
    "pair_trading": ("aapl_msft_2m.parquet", PairTradingStrategy, 1500, 450),
}


def _run(case: str) -> dict[str, Any]:
    fixture, factory, n_train, n_test = CASES[case]
    data = _load(fixture)
    engine = BacktestEngine()
    engine.set_training_data(data.iloc[:n_train])
    engine.set_backtest_data(data.iloc[n_train : n_train + n_test])
    engine.add_strategy(factory())
    with contextlib.redirect_stdout(io.StringIO()):
        engine.train_strategies()
        metrics = engine.run()
    return {k: float(v) for k, v in metrics.items()}


def _read_baseline() -> dict[str, dict[str, float]]:
    if BASELINE_PATH.exists():
        result: dict[str, dict[str, float]] = json.loads(BASELINE_PATH.read_text())
        return result
    return {}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_metrics_match_baseline(case: str) -> None:
    metrics = _run(case)
    baseline = _read_baseline()

    if UPDATE:
        baseline[case] = metrics
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        return

    assert case in baseline, f"No baseline for '{case}'; run with UPDATE_BASELINE=1 first."
    expected = baseline[case]
    assert set(metrics) == set(expected)
    for key, value in expected.items():
        actual = metrics[key]
        if math.isnan(value):
            assert math.isnan(actual), key
        else:
            assert actual == pytest.approx(value, rel=1e-9, abs=1e-12), key
