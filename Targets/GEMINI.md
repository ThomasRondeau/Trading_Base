# Targets Directory Guidelines

This directory defines the ground truth labels (targets) that the Machine Learning models aim to predict.

## Adding a New Target

1. **Inheritance:** Your class must inherit from `TargetInterface`.
2. **Dependencies:** If a target depends on other targets (rare, but possible), declare them by returning a list of `TargetID`s in the `dependencies` property.
3. **Calculation:** Implement the `calculate(self, resolved_dependencies)` method to compute the `self.target` property (a pandas Series of labels). 
4. **Look-Ahead Shift:** This is critical. In `calculate()`, you MUST use the `self.training_data_shift` attribute to align future data with current data to prevent look-ahead bias during training. (e.g., `shifted = self.data['close'].iloc[5 : 5 - self.training_data_shift]`). 
5. **State Management:** Implement `update(self, new_data: pd.DataFrame, training_data_shift: int)` to set the new DataFrame and shift, and `clear_cache(self)` to set `self.target` to `None`.
6. **Registration:** Ensure the target gets an ID in `TargetID.py` and is registered in the Strategy's `train()` method using `TargetManager.register_target()`.