# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Modular Python quant-trading research system: technical-indicator features → ML/statistical models → strategies → bar-by-bar backtest. Code comments, docstrings and console output are in French; keep that convention. Strategy write-ups live in `docs/` (`STRATEGY_*.md`, `ARCHITECTURE_PAIR_TRADING.md`, `README_EVOLUTIONS.md`).

## Running

Python 3.10 (on this machine `python` is 3.8 — use `py -3.10`). Dependencies are declared in `pyproject.toml`: `py -3.10 -m pip install -e .[dev]`.

- Tests: `py -3.10 -m pytest` (single test: `py -3.10 -m pytest tests/test_walk_forward.py::test_expand_grid`; skip the ~80 s end-to-end backtests with `-m "not slow"`). `pythonpath = ["."]` is set in `pyproject.toml`, so run from the repo root.
- Characterization baseline: `tests/test_characterization.py` backtests every strategy on frozen parquet fixtures (`tests/fixtures/`) and compares to `baseline_metrics.json`. A pure refactor must keep it green. Only regenerate (`UPDATE_BASELINE=1 py -3.10 -m pytest tests/test_characterization.py`) when a change is meant to alter results, and review the JSON diff.
- Lint/types: `py -3.10 -m ruff check .`, `py -3.10 -m mypy <path>` (strict; legacy root packages don't pass yet).
- Ongoing refactor plan (bug list, target `quantlab/` architecture, steps): `~/.claude/plans/parallel-watching-crown.md`.
- Secrets: `MASSIVE_API_KEY` is read from the environment or a `.env` file (template: `.env.example`; `.env` is gitignored).

All scripts must be run from the repo root (imports are absolute from the root packages):

- `python Main_Yfinance.py` — Cascade strategy backtest on AAPL using yfinance 2-minute bars (yfinance only serves ~60 days of intraday data).
- `python Main_<Arima|Hybrid|MeanReversion|PairTrading|Trend>.py` — single-strategy backtests.
- `python Main_Project_Submission.py` — runs every strategy on ~6 months of 1-min data from Massive API (`Connections/MassiveAPI.py`).
- `python Main_IBKR.py` / `python UI_App.py` — require TWS/IB Gateway running locally with the API enabled (127.0.0.1:7497).
- `python test_massive.py` — smoke test of the Massive API connection (not a unit test).
- `python Main_WalkForward.py` — example walk-forward hyperparameter search (TrendFollowing on yfinance data).

## Architecture

Pipeline: `DataManager` → `Strategy.train()` (registers features/targets, trains models) → `Strategy.test()` → `BacktestEngine.run()` calls `strategy.on_bar(bar)` per row and simulates fills.

- **Connections/** — `DataManager` exposes `collect_data_yfinance`, `collect_data_massive`, and `collect_training_data` (IBKR, with per-symbol CSV cache `Connections/<SYMBOL>_historical_data.csv`). Signature convention: `(symbol, days_back, until)` = window from `days_back` days ago to `until` days ago, so training uses e.g. `(50, 5)` and backtest `(5, 0)`. `TradingApp` is an IBKR `EClient/EWrapper` singleton — use `TradingApp.get_instance()`, never the constructor (it raises on a second instance). `BacktestEngine.__init__` also grabs this singleton.
- **Features/** and **Targets/** — parallel designs. Each indicator/target implements `FeatureInterface` / `TargetInterface` (`calculate`, `update`, `clear_cache`, `dependencies`) and is keyed by an enum (`FeatureID` / `TargetID`). `FeatureManager` / `TargetManager` hold registered instances, resolve dependencies recursively, and cache results; `update(new_data)` clears caches and pushes data into every registered instance. New components must be added to the enum and exported in the package `__init__.py` (packages are consumed via `from Features import *` / `from Targets import *`). Note: `FeatureInterface.dependencies()` is a method, but `TargetManager` reads `target.dependencies` as an attribute.
- **Features/Models/** (`AmplitudeFeatures`, `DirectionFeatures`, `OpportunityFeatures`) are composite features that bundle base indicators into the input matrix for the matching model.
- **Models/** — implement `TradingModelInterface`. Models do not receive data directly: the strategy injects managers via `setFeatureManager` / `setTargetManager`, then calls `train()`, `test()`, `cross_validate()`, and `predict()` (no args — reads the current state of the feature manager).
- **Look-ahead protection** — strategies feed `FeatureManager` with `training_data.iloc[:-training_data_shift]` while `TargetManager.update(training_data, training_data_shift=...)` receives the full frame; targets are forward-shifted (`Target_Plus_N`). Preserve this split when touching training code.
- **Strategies/** — implement `TradingStrategyInterface`. `on_bar` raises unless `update_status(True)` was called (the engine does this in `run()`). Strategies keep a rolling `current_data` window (e.g. last 200 bars), re-run `featureManager.update` each bar, and return a list of `Types.Order(strategy_ID, symbol, side='BUY'|'SELL', quantity)`. `CascadeStrategy` chains Opportunity → Direction → Amplitude models and only trades when they agree.
- **Engines/BacktestEngine** — the actual simulation lives inline in `run()`: orders fill at the current bar's close, positions exit after 5 bars or on stop-loss / take-profit from `RiskManager`, fees = 1 bp × quantity × (entry + exit). `print_summary` returns a metrics dict (win rate, net PnL %, annualized Sharpe from daily-grouped returns, max drawdown). `BacktestOrderManager`/`OrderManager` are instantiated but not used for fills.
- **Engines/WalkForwardOptimizer** — the single place for hyperparameter search (don't write per-strategy sweep scripts). Folds are `[train][embargo][validation][embargo][test]`: params are chosen on validation, refit on train+validation, scored once on test. It takes a `strategy_factory(**params)` because strategies register features in `train()` and cannot be retrained — every backtest needs a fresh instance. The objective is any key of the `print_summary` dict.
- **Multi-symbol data** — for pair trading, bars carry `close_<SYMBOL>` columns; the engine looks up `close_{order.symbol}` and falls back to `close`.

## Adding components

Use strict type hints and vectorized pandas/numpy throughout.

- **Feature** — subclass `FeatureInterface`; return dependency `FeatureID`s from `dependencies()`; in `calculate(resolved_dependencies)` fetch them with `cast(DepClass, resolved_dependencies.get(FeatureID.X))` and check they exist; `update(new_data)` stores the frame and `clear_cache()` resets every computed attribute to `None`. Add the ID to `FeatureID.py`, export it in `Features/__init__.py`, and register it in the strategy's `train()`.
- **Target** — subclass `TargetInterface`; compute `self.target` (a Series) in `calculate()` and always apply `self.training_data_shift` when aligning future prices (e.g. `self.data['close'].iloc[5 : 5 - self.training_data_shift]`); `update(new_data, training_data_shift)` stores both; `clear_cache()` sets `self.target = None`. Add to `TargetID.py`, export, register.
- **Model** — subclass `TradingModelInterface`; never instantiate managers inside the model (they arrive via setters). Build data in a private `_build_dataset()` (engineered feature + target → DataFrame → `.dropna()` → `X`, `y` arrays). `cross_validate()` must use `sklearn.model_selection.TimeSeriesSplit`. `predict()` uses only `.iloc[-1]` of the features, reshaped to 2D, and returns a single scalar (`float`/`bool`/`str`).
- **Strategy** — subclass `TradingStrategyInterface`; register features/targets and wire managers into models in `train()`; in `on_bar` append the bar to internal history, `featureManager.update(...)`, call model `predict()`s, return `list[Order]` (empty list for no trade); manage `IsRunning` via `update_status()`/`get_status()`.

The `quant-finance-expert` skill (`.claude/skills/`) holds domain guidance for strategy/ML validation (overfitting diagnosis, purged CV, labeling, microstructure).
