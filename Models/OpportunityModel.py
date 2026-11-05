from typing import Any, Dict, Set, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from Features import *
from Targets import *
from Models.TradingModelInterface import TradingModelInterface
import numpy as np
import pandas as pd


class OpportunityModel(TradingModelInterface):
    """
    Détecteur d'opportunité - équivalent à un RandomForestClassifier
    pour identifier s'il y a une opportunité de trade.
    """
    
    def __init__(self, model=None):
        self.feature_manager = None
        self.target_manager = None

        if model is not None:
            self.model = model
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=15,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

    def setFeatureManager(self, feature_manager : FeatureManager) -> None:
        self.feature_manager = feature_manager

    def setTargetManager(self, target_manager: TargetManager) -> None:
        self.target_manager = target_manager

    def _build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit X et y à partir du FeatureManager et TargetManager.
        """
        Indicator = cast(OpportunityFeatures, self.feature_manager.get_feature(FeatureID.OPPORTUNITY))
        Target = cast(Target_Opportunity, self.target_manager.get_target(TargetID.OPPORTUNITY))

        df = pd.DataFrame({
            "bb_width": Indicator.bb_width_norm,
            "vol_ratio": Indicator.volatility_ratio,
            "vol_change": Indicator.volatility_change,
            "volume_ratio": Indicator.volume_ratio,
            "volume_trend": Indicator.volume_trend,
            "momentum_align": Indicator.momentum_alignment,
            "dist_lower": Indicator.dist_lower_bb,
            "dist_upper": Indicator.dist_upper_bb,
            "dist_sma": Indicator.dist_sma,
            "price_accel": Indicator.price_accel,
            "target": Target.target
        })

        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        X = df.drop(columns="target").to_numpy()
        y = df["target"].to_numpy()

        return X, y
    
    def get_required_features(self) -> Set[FeatureID]:
        return {FeatureID.OPPORTUNITY}
    
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
        return balanced_accuracy_score(y, self.model.predict(X))

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

            model = clone(self.model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, y_pred))

        return float(np.mean(scores)), scores

    def predict(self) -> bool:
        """
        Prédit s'il y a une opportunité de trade sur la dernière observation.
        """
        Indicator = cast(OpportunityFeatures, self.feature_manager.get_feature(FeatureID.OPPORTUNITY))
        
        feature_array = np.array([[
            Indicator.bb_width_norm.iloc[-1],
            Indicator.volatility_ratio.iloc[-1],
            Indicator.volatility_change.iloc[-1],
            Indicator.volume_ratio.iloc[-1],
            Indicator.volume_trend.iloc[-1],
            Indicator.momentum_alignment.iloc[-1],
            Indicator.dist_lower_bb.iloc[-1],
            Indicator.dist_upper_bb.iloc[-1],
            Indicator.dist_sma.iloc[-1],
            Indicator.price_accel.iloc[-1]
        ]])
        
        prediction = self.model.predict(feature_array)[0]
        
        return True if prediction == 1 else False
