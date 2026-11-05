# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ML trading-strategy research tool: technical-indicator features → ML/statistical models → strategies → bar-by-bar backtest, with IBKR (TWS / IB Gateway) for data and, as a first-class goal, **paper/live trading**. The same strategy, feature and risk code must work in both backtest and live mode; don't introduce backtest-only shortcuts. All code, docstrings and logs are in English and use `logging` (no `print` in the library). Strategy write-ups live in `docs/`.

A staged refactor is in progress (bug list, target architecture, steps 0–6): `~/.claude/plans/parallel-watching-crown.md`. Steps 0–2 are done: safety-net tests, bug fixes, move into the `quantlab/` package.

## Running

Python 3.10 (on this machine `python` is 3.8 — use `py -3.10`). Dependencies are in `pyproject.toml`: `py -3.10 -m pip install -e .[dev]`. Run everything from the repo root.

- Tests: `py -3.10 -m pytest` (single test: `py -3.10 -m pytest tests/test_walk_forward.py::test_expand_grid`; skip the ~80 s end-to-end backtests with `-m "not slow"`).
- Characterization baseline: `tests/test_characterization.py` backtests every strategy on frozen parquet fixtures (`tests/fixtures/`) and compares all metrics to `baseline_metrics.json`. A pure refactor must keep it green. Only regenerate (`UPDATE_BASELINE=1 py -3.10 -m pytest tests/test_characterization.py`) when a change is meant to alter results, and review the JSON diff.
- Lint/format/types: `py -3.10 -m ruff check .`, `py -3.10 -m ruff format .`, `py -3.10 -m mypy quantlab` (strict; not clean yet).
- Secrets: `MASSIVE_API_KEY` from the environment or `.env` (template `.env.example`; `.env` is gitignored).

Entry scripts (thin wrappers, to be replaced by a CLI in step 6):

- `Main_Yfinance.py` (cascade), `Main_<Arima|Hybrid|MeanReversion|Trend|PairTrading>.py` — single-strategy backtests on yfinance 2-minute bars (yfinance serves ~60 days of intraday data).
- `Main_AllStrategies.py` — every strategy on ~6 months of 1-minute Massive data, then as a portfolio.
- `Main_WalkForward.py` — walk-forward hyperparameter search example.
- `Main_IBKR.py`, `UI_App.py` — need TWS/IB Gateway with the API enabled on 127.0.0.1:7497 (paper port).

## Architecture (`quantlab/`)

Flow: `DataManager` → `strategy.train(training_data)` (registers features/targets, fits models) → `BacktestEngine.run()` feeds each bar to `strategy.on_bar(bar)` and simulates the resulting orders.

Package `__init__.py` files are intentionally empty: always import from the defining module (`from quantlab.features.technical import SMA`).

- **data/** — `DataManager` (`collect_data_yfinance`, `collect_data_massive`, `collect_training_data` for IBKR with a CSV cache in `data_cache/`). Window convention for all sources: `(symbol, days_back, until)` = from `days_back` days ago to `until` days ago. Every source returns `schema.validate_ohlcv` output (tz-aware UTC `DatetimeIndex` named `Datetime`, float lowercase OHLCV, sorted, unique); metrics and VWAP sessions rely on it.
- **features/** and **targets/** — parallel designs: instances implement `FeatureInterface` / `TargetInterface` (`update`, `calculate`, `clear_cache`, `dependencies`) and are registered under an enum key (`FeatureID` / `TargetID`) in a `FeatureManager` / `TargetManager`, which resolve dependencies recursively and cache per update; `manager.update(data)` clears every cache. `FeatureInterface.dependencies()` is a method, `TargetInterface.dependencies` a property. `features/composite.py` bundles base indicators (`technical.py`) into one input set per model. `features/levels.py` (support/resistance) is **not causal**: charting only.
- **Look-ahead protection** — ML strategies feed the `FeatureManager` with `training_data.iloc[:-training_data_shift]` while `TargetManager.update(training_data, training_data_shift=...)` gets the full frame and trims back (`CloseBasedTarget._trim`). Preserve this split when touching training code.
- **models/** — `TradingModelInterface`; models never get data directly: strategies inject managers (`set_feature_manager` / `set_target_manager`), then call `train()`, `cross_validate()`, `predict()` (no args, last row). `SklearnFeatureModel` holds the shared dataset/CV logic; a model only declares `feature_id`, `target_id`, `feature_columns`, `score` and its estimator. CV clones the configured estimator.
- **strategies/** — `TradingStrategyInterface`. Strategies keep a rolling `current_data` window, re-run `feature_manager.update` every bar and return `list[Order]`. `on_bar` returns `[]` when not running, except `CascadeStrategy`, which raises. Orders are `quantlab.execution.orders.Order(strategy_id, symbol, side='BUY'|'SELL', quantity)`.
- **backtest/engine.py** — simulation inline in `run()`: fills at the signal bar's close, exits after 5 bars or on SL/TP from `RiskManager` (checked on closes), fees = 1 bp × qty × (entry + exit). The metrics (`print_summary` dict) sum per-trade percentages — not an equity curve; step 4 replaces this. Multi-symbol bars carry `close_<SYMBOL>` columns.
- **backtest/walk_forward.py** — the single place for hyperparameter search. Folds are `[train][embargo][validation][embargo][test]`: chosen on validation, refit on train+validation, scored once on test. Takes `strategy_factory(**params)` because a strategy instance cannot be retrained.
- **execution/ibkr/trading_app.py** — `TradingApp` (EClient/EWrapper) singleton: always `TradingApp.get_instance()`, the constructor raises on a second instance. To be rewritten in step 4.

## Adding components

- **Feature** — subclass `FeatureInterface`; add a `FeatureID`; declare dependencies; fetch them in `calculate` and fail if missing; `clear_cache()` resets every computed attribute and the data. Register it in the strategy's `train()`.
- **Target** — subclass `CloseBasedTarget` and pass the full-length label through `self._trim(...)`; add a `TargetID`.
- **Model** — subclass `SklearnFeatureModel` (or `TradingModelInterface`); `cross_validate` must stay time-ordered (`TimeSeriesSplit`); `predict()` uses only the last row and returns a scalar.
- **Strategy** — subclass `TradingStrategyInterface`; must be causal (only data up to the current bar) so it can run live.

The `quant-finance-expert` skill (`.claude/skills/`) holds domain guidance for strategy/ML validation (overfitting diagnosis, purged CV, labeling, microstructure).
