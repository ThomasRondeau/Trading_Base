import pandas as pd
from typing import Dict, Set, Type, Any
from Features.FeatureID import FeatureID
from Indicators.IndicatorInterface import IndicatorInterface


class FeatureManager:
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):
        self.data = data
        self._cache: Dict[FeatureID, IndicatorInterface] = {}
        self._indicators: Dict[FeatureID, IndicatorInterface] = {}


    def register_indicator(self, feature_id: FeatureID, indicator: IndicatorInterface):
        """
        Enregistre directement une instance d'indicateur.
        """
        if feature_id in self._indicators:
            raise ValueError(f"Indicator {feature_id} already registered.")
        self._indicators[feature_id] = indicator


    def get_feature(self, feature_id: FeatureID) -> IndicatorInterface:
        """
        Retourne la valeur de la feature depuis le cache ou en recalculant si nécessaire.
        """
        if feature_id in self._cache:
            return self._cache[feature_id]

        if feature_id not in self._indicators:
            raise ValueError(f"Indicator {feature_id} not registered.")

        indicator = self._indicators[feature_id]

        # Résoudre récursivement les dépendances
        resolved_dependencies = {}
        for dep_id in indicator.dependencies():
            resolved_dependencies[dep_id] = self.get_feature(dep_id)

        indicator.calculate(resolved_dependencies)
        self._cache[feature_id] = indicator
        return indicator
    

    def get_features(self, feature_ids: Set[FeatureID]) -> Dict[FeatureID, IndicatorInterface]:
        """
        Retourne un dictionnaire de valeurs pour les features demandées.
        """
        return {fid: self.get_feature(fid) for fid in feature_ids if fid in self._indicators}


    def update(self, new_data: pd.DataFrame):
        """
        Met à jour toutes les instances d'indicateurs et vide le cache.
        """
        self.data = new_data
        self.clear_cache()
        for indicator in self._indicators.values():
            indicator.update(new_data)


    def clear_cache(self):
        """
        Vide le cache des features calculées.
        """
        self._cache.clear()
        
        for indicator in self._indicators.values():
            indicator.clear_cache()