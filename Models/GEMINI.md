# Models Directory Guidelines

This directory contains the machine learning models (e.g., RandomForest, XGBoost) that predict market conditions based on features.

## Adding a New Model

1. **Inheritance:** Your class must inherit from `TradingModelInterface`.
2. **Managers:** The model receives `FeatureManager` and `TargetManager` via setter methods. Do not instantiate these managers inside the model.
3. **Dataset Building:** Create a private `_build_dataset()` method. Extract the specific engineered feature from the `FeatureManager` and the specific target from the `TargetManager`. Combine them into a pandas DataFrame, apply `.dropna()`, and return a cleaned `X` and `y` numpy array.
4. **Training & Validation:** Implement `train()` on the full dataset. Implement `cross_validate()` using `TimeSeriesSplit` from `sklearn.model_selection` to strictly respect temporal ordering (no look-ahead bias).
5. **Prediction:** The `predict()` method should extract ONLY the last row (`.iloc[-1]`) of the required features from the `FeatureManager`, format it into a 2D numpy array, make the prediction, and return a single typed value (e.g., `float`, `bool`, or `str`).