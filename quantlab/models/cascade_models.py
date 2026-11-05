"""The three models of the cascade strategy: opportunity -> direction -> amplitude."""

from __future__ import annotations

from typing import Any, ClassVar

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score

from quantlab.features.feature_id import FeatureID
from quantlab.models.base import SklearnFeatureModel
from quantlab.targets.target_id import TargetID


class AmplitudeModel(SklearnFeatureModel):
    """RandomForestRegressor estimating the signed 5-bar move, in percent."""

    feature_id = FeatureID.AMPLITUDE
    target_id = TargetID.AMPLITUDE
    feature_columns: ClassVar[dict[str, str]] = {
        "volatility": "volatility",
        "dist_sma": "dist_sma",
        "bb_width": "bb_width",
        "rsi_extreme": "rsi_extreme",
        "macd_hist_abs": "macd_hist_abs",
        "momentum_5": "momentum_5",
        "volume_ratio": "volume_ratio",
    }
    score = staticmethod(r2_score)

    def __init__(self, model: Any = None) -> None:
        super().__init__(
            model
            if model is not None
            else RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
        )

    def predict(self) -> float:
        return float(self.model.predict(self._last_row())[0])


class DirectionModel(SklearnFeatureModel):
    """XGBClassifier predicting the 5-bar direction ('UP' / 'DOWN')."""

    feature_id = FeatureID.DIRECTION
    target_id = TargetID.DIRECTION
    feature_columns: ClassVar[dict[str, str]] = {
        "rsi": "rsi",
        "macd_hist": "macd_hist_norm",
        "dist_sma": "price_dist_sma",
        "sma_slope": "sma_slope_norm",
        "bb_pos": "bb_position",
        "vol_trend": "volume_trend",
        "volatility": "volatility",
        "mom_5": "mom_5",
        "mom_10": "mom_10",
        "mom_20": "mom_20",
    }
    score = staticmethod(accuracy_score)

    def __init__(self, model: Any = None, max_depth: int = 4) -> None:
        super().__init__(
            model
            if model is not None
            else xgb.XGBClassifier(
                n_estimators=100,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
        )

    def predict(self) -> str:
        prediction = self.model.predict(self._last_row())[0]
        return "UP" if prediction == 1 else "DOWN"


class OpportunityModel(SklearnFeatureModel):
    """RandomForestClassifier detecting whether a tradable move is coming."""

    feature_id = FeatureID.OPPORTUNITY
    target_id = TargetID.OPPORTUNITY
    feature_columns: ClassVar[dict[str, str]] = {
        "bb_width": "bb_width_norm",
        "vol_ratio": "volatility_ratio",
        "vol_change": "volatility_change",
        "volume_ratio": "volume_ratio",
        "volume_trend": "volume_trend",
        "momentum_align": "momentum_alignment",
        "dist_lower": "dist_lower_bb",
        "dist_upper": "dist_upper_bb",
        "dist_sma": "dist_sma",
        "price_accel": "price_accel",
    }
    score = staticmethod(balanced_accuracy_score)

    def __init__(self, model: Any = None) -> None:
        super().__init__(
            model
            if model is not None
            else RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=15,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        )

    def predict(self) -> bool:
        return bool(self.model.predict(self._last_row())[0] == 1)
