from typing import Any, Dict, Set, cast
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from Features.FeatureID import FeatureID
from Features.FeatureManager import FeatureManager
from Features.TargetID import TargetID
from Features.TargetManager import TargetManager
from Indicators import *
from Targets import *
from Models.TradingModelInterface import TradingModelInterface
import numpy as np
import pandas as pd

class AmplitudeModel(TradingModelInterface):
    """
    Estimateur d'amplitude - équivalent à un RandomForestRegressor
    pour estimer l'amplitude du mouvement.
    """
    
    def __init__(self, model=None):
        self.feature_manager = None
        self.target_manager = None

        if model is not None:
            self.model = model
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
    def setFeatureManager(self, feature_manager : FeatureManager) -> None:
        self.feature_manager = feature_manager

    def setTargetManager(self, target_manager: TargetManager) -> None:
        self.target_manager = target_manager

    def _build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit X et Y à partir du FeatureManager et TargetManager.
        """
        Indicator = cast(AmplitudeIndicators, self.feature_manager.get_feature(FeatureID.AMPLITUDE))
        Target = cast(Target_Amplitude, self.target_manager.get_target(TargetID.AMPLITUDE))

        df = pd.DataFrame({
            "volatility": Indicator.current_volatility,
            "percentile": Indicator.volatility_percentile,
            "distance_sma": Indicator.distance_to_sma,
            "distance_bb": Indicator.distance_to_bb,
            "rsi": Indicator.rsi_extreme,
            "macd": Indicator.macd_intensity,
            "momentum": Indicator.momentum_strength,
            "volume": Indicator.volume_intensity,
            "target": Target.target
        })

        df = df.dropna().reset_index(drop=True)
        
        X = df.drop(columns="target").to_numpy()
        y = df["target"].to_numpy()

        return X, y

    def train(self) -> None:
        """
        Entraîne le modèle sur toutes les données.
        """
        X, y = self._build_dataset()
        self.model.fit(X, y)

    def test(self) -> float:
        """
        Test rapide sur toutes les données (⚠️ fuite de données possible).
        """
        X, y = self._build_dataset()
        return r2_score(y, self.model.predict(X))

    def cross_validate(self, n_splits: int = 5) -> tuple[float, list[float]]:
        """
        Validation croisée temporelle simple (TimeSeriesSplit).
        Retourne la moyenne et les scores R² par split.
        """
        X, y = self._build_dataset()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(r2_score(y_test, y_pred))

        return float(np.mean(scores)), scores

    def get_required_features(self) -> Set[FeatureID]:
        return {FeatureID.AMPLITUDE}
    
    def predict(self) -> float:
        """
        Prédit l'amplitude du mouvement sur la dernière observation.
        """
        Indicator = cast(AmplitudeIndicators, self.feature_manager.get_feature(FeatureID.AMPLITUDE))
        
        feature_array = np.array([[
            Indicator.current_volatility.iloc[-1],
            Indicator.volatility_percentile.iloc[-1],
            Indicator.distance_to_sma.iloc[-1],
            Indicator.distance_to_bb.iloc[-1],
            Indicator.rsi_extreme.iloc[-1],
            Indicator.macd_intensity.iloc[-1],
            Indicator.momentum_strength.iloc[-1],
            Indicator.volume_intensity.iloc[-1]
        ]])
        
        return float(self.model.predict(feature_array)[0])
