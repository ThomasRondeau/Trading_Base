from FeatureRegistry import FeatureRegistry
from FeatureID import FeatureID
from typing import Dict, Set, Any, Optional, List
import pandas as pd


class FeatureDispatcher:
    """
    Dispatcher optimisé pour le calcul de features.
    
    Cette classe est le cœur du système : elle prend une liste de features requises,
    résout leurs dépendances, et les calcule dans l'ordre optimal. Elle remplace
    efficacement la logique de ton FeatureEngineering en ajoutant l'optimisation
    et la réutilisation des calculs.
    """
    
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self._cache: Dict[FeatureID, Any] = {}
        self._last_data_hash: Optional[int] = None
    
    def compute_features(self, data: pd.DataFrame, required_features: Set[FeatureID]) -> Dict[FeatureID, Any]:
        """
        Calcule les features demandées et toutes leurs dépendances.
        
        Cette méthode est l'équivalent optimisé de tes méthodes create_opportunity_features,
        create_direction_features, et create_amplitude_features, mais en une seule fois
        et sans recalculer les features partagées.
        
        Args:
            data: DataFrame avec les données OHLCV
            required_features: Set des features à calculer
            
        Returns:
            Dict contenant les valeurs calculées pour chaque feature
        """

        # TODO pas bon, on doit pouvoir réutiliser le cache et updater uniquement les features avec le nouveau tick de données

        # Vérifier si on peut réutiliser le cache (même données)
        current_hash = hash(str(data.tail(50).values.tobytes()))  # Hash des 50 dernières lignes
        if current_hash != self._last_data_hash:
            self._cache.clear()
            self._last_data_hash = current_hash
        
        # Résoudre toutes les dépendances nécessaires
        all_features = self.registry.get_all_dependencies(required_features)
        
        # Calculer l'ordre optimal (tri topologique)
        calculation_order = self._compute_calculation_order(all_features)
        
        # Calculer chaque feature dans l'ordre
        for feature_id in calculation_order:
            if feature_id not in self._cache:
                config = self.registry.get_config(feature_id)
                
                # Préparer les dépendances déjà calculées
                deps_values = {dep: self._cache[dep] for dep in config.dependencies}
                
                try:
                    # Calculer la feature
                    self._cache[feature_id] = config.calculator(data, deps_values)
                except Exception as e:
                    raise ValueError(f"Erreur lors du calcul de {feature_id.name}: {e}")
        
        # Retourner seulement les features demandées (pas toutes les dépendances)
        return {fid: self._cache[fid] for fid in required_features}
    
    def _compute_calculation_order(self, features: Set[FeatureID]) -> List[FeatureID]:
        """
        Calcule l'ordre optimal de calcul en utilisant un tri topologique.
        Cette méthode s'assure qu'une feature est toujours calculée après ses dépendances.
        """
        ordered = []
        remaining = features.copy()
        
        while remaining:
            # Trouver les features prêtes (toutes dépendances satisfaites)
            ready = set()
            for feature in remaining:
                config = self.registry.get_config(feature)
                if config.dependencies.issubset(set(ordered)):
                    ready.add(feature)
            
            if not ready:
                remaining_names = [f.name for f in remaining]
                raise ValueError(f"Dépendance circulaire détectée dans: {remaining_names}")
            
            # Ajouter par ordre alphabétique pour la cohérence
            ready_list = sorted(ready, key=lambda x: x.name)
            ordered.extend(ready_list)
            remaining -= ready
        
        return ordered