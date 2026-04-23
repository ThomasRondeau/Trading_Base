# Project Context: Quantitative Trading System

This project is a modular, machine-learning-based quantitative trading system written in Python. It is designed to backtest and execute trading strategies by leveraging technical indicators as features and predicting market movements.

## Architecture

The system follows a strict, interface-driven architecture to maintain modularity:
- **Connections:** Data retrieval via YFinance or IBKR API (`DataManager`, `TradingApp`). Handles CSV caching.
- **Features (`Features/`):** Technical indicators and feature engineering. Managed by `FeatureManager` with automatic recursive dependency resolution. All features must implement `FeatureInterface`.
- **Targets (`Targets/`):** Label generation for ML models (e.g., predicting direction or opportunity 5 bars ahead). Managed by `TargetManager`. All targets must implement `TargetInterface`.
- **Models (`Models/`):** ML wrappers (RandomForest, XGBoost) implementing `TradingModelInterface`. They consume features and targets to train and predict.
- **Strategies (`Strategies/`):** The trading logic (e.g., `CascadeStrategy`). Implements `TradingStrategyInterface`. Consumes model predictions to generate `Order` objects.
- **Engines (`Engines/`):** The `BacktestEngine` runs the simulation loop, injecting data bar by bar, triggering strategy updates, and handling orders via `BacktestOrderManager` and `RiskManager`.

## Development Guidelines

- **Type Hinting:** Strict use of Python type hints (`typing` module, e.g., `Dict`, `List`, `Set`, `Optional`, `cast`).
- **Pandas/Numpy:** Use vectorized operations with `pandas` and `numpy` wherever possible for performance.
- **No Look-Ahead Bias:** When calculating features or targets, be extremely careful not to use future data. The `TargetManager` uses `training_data_shift` to prevent this during training.
- **Caching:** The `FeatureManager` and `TargetManager` use caching. Any new `FeatureInterface` or `TargetInterface` implementation MUST properly clear its internal state in the `clear_cache()` method.
- **Interfaces:** When adding a new component, you MUST inherit from the corresponding base class (e.g., `FeatureInterface`, `TradingModelInterface`) and implement all abstract methods.