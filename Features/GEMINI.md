# Features Directory Guidelines

This directory contains technical indicators and engineered features used by the Machine Learning models.

## Adding a New Feature

1. **Inheritance:** Your class must inherit from `FeatureInterface`.
2. **Dependencies:** If your feature relies on other features (e.g., an engineered feature relying on SMA and ATR), declare them in the `dependencies()` method by returning a set of their `FeatureID`s.
3. **Calculation:** Implement the `calculate(self, resolved_dependencies)` method. Extract your dependencies safely using `cast(DependencyClass, resolved_dependencies.get(FeatureID.DEPENDENCY_NAME))`. Verify they exist before proceeding.
4. **Vectorization:** Use `pandas` and `numpy` vectorized operations in `calculate()`.
5. **State Management:** Implement `update(self, new_data: pd.DataFrame)` to set the new DataFrame, and `clear_cache(self)` to reset all internal calculated variables to `None`.
6. **Registration:** After creation, ensure the feature gets an ID in `FeatureID.py` and is registered in the Strategy's `train()` method using `FeatureManager.register_indicator()`.