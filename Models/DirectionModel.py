from typing import Any, Dict, Set, cast
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from Features.TargetID import TargetID
from Features.FeatureID import FeatureID
from Features.FeatureManager import FeatureManager
from Features.TargetManager import TargetManager
from Indicators import *
from Targets import *
from Models.TradingModelInterface import TradingModelInterface
import numpy as np
import pandas as pd
import xgboost as xgb


class DirectionModel(TradingModelInterface):
    """
    Prédicteur de direction - équivalent à un XGBClassifier
    pour déterminer la direction du mouvement.
    """
    
    def __init__(self, model=None):
        self.feature_manager = None
        self.target_manager = None

        if model is not None:
            self.model = model
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            
    def setFeatureManager(self, feature_manager : FeatureManager) -> None:
        self.feature_manager = feature_manager

    def setTargetManager(self, target_manager: TargetManager) -> None:
        self.target_manager = target_manager

    def _build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit X et y à partir du FeatureManager et TargetManager.
        """
        Indicator = cast(DirectionIndicators, self.feature_manager.get_feature(FeatureID.DIRECTION))
        Target = cast(Target_Plus_5, self.target_manager.get_target(TargetID.SHIFTED_PLUS_5))

        df = pd.DataFrame({
            "rsi_momentum": Indicator.rsi_momentum,
            "macd_momentum": Indicator.macd_momentum,
            "price_vs_sma": Indicator.price_vs_sma,
            "distance_bb": Indicator.distance_to_bb,
            "rsi_extreme": Indicator.rsi_extreme,
            "macd_intensity": Indicator.macd_intensity,
            "momentum_strength": Indicator.momentum_strength,
            "volume_intensity": Indicator.volume_intensity,
            "target": Target.target
        }).dropna()

        X = df.drop(columns="target").to_numpy()
        y = df["target"].to_numpy()

        return X, y
    
    def get_required_features(self) -> Set[FeatureID]:
        return {FeatureID.DIRECTION}
    
    def train(self) -> None:
        """
        Entraîne le modèle sur toutes les données.
        """
        X, y = self._build_dataset()
        self.model.fit(X, y)

    def test(self) -> float:
        """
        Test rapide sur toutes les données (⚠️ fuite de données possible).
        Retourne une accuracy.
        """
        X, y = self._build_dataset()
        return accuracy_score(y, self.model.predict(X))

    def cross_validate(self, n_splits: int = 5) -> tuple[float, list[float]]:
        """
        Validation croisée temporelle simple (TimeSeriesSplit).
        Retourne la moyenne et les scores d'accuracy par split.
        """
        X, y = self._build_dataset()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

        return float(np.mean(scores)), scores

    def predict(self) -> str:
        """
        Prédit la direction ('UP' ou 'DOWN') sur la dernière observation.
        """
        Indicator = cast(DirectionIndicators, self.feature_manager.get_feature(FeatureID.DIRECTION))
        
        feature_array = np.array([[
            Indicator.rsi_momentum.iloc[-1],
            Indicator.macd_momentum.iloc[-1],
            Indicator.price_vs_sma.iloc[-1],
            Indicator.distance_to_bb.iloc[-1],
            Indicator.rsi_extreme.iloc[-1],
            Indicator.macd_intensity.iloc[-1],
            Indicator.momentum_strength.iloc[-1],
            Indicator.volume_intensity.iloc[-1]
        ]])
        
        prediction = self.model.predict(feature_array)[0]
        return "UP" if prediction == 1 else "DOWN"
