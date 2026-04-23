# Strategies Directory Guidelines

This directory contains the actual trading logic (`CascadeStrategy`, etc.) that coordinates models and managers to generate trading orders.

## Adding a New Strategy

1. **Inheritance:** Your class must inherit from `TradingStrategyInterface`.
2. **Managers:** A strategy typically holds instances of `FeatureManager` and `TargetManager`, as well as instances of the models it will use (e.g., `AmplitudeModel`).
3. **Training Phase (`train()`):**
   - Register all required base indicators and feature models in the `FeatureManager`.
   - Register all required targets in the `TargetManager`.
   - Pass the managers into the required trading models.
   - Trigger the `train()` method on all instantiated models.
4. **Execution Phase (`on_bar(self, bar)`):**
   - This method is called by the `BacktestEngine` or Live Engine for each new data point (bar).
   - Concatenate the new bar to the strategy's internal historical data.
   - Update the `FeatureManager` with this new data.
   - Call `predict()` on your models to get the current signals.
   - Return a list of `Order` objects (e.g., `[Order(symbol='AAPL', side='BUY', quantity=10)]`) based on your custom logic. If no trade is to be taken, return an empty list `[]`.
5. **State:** Implement `update_status()` and `get_status()` to manage the `IsRunning` boolean flag correctly.