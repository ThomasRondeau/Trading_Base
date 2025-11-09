from Features.FeatureRegistry import FeatureRegistry
from Features.FeatureDispatcher import FeatureDispatcher
from Features.FeatureID import FeatureID
from Models.TradingModelInterface import TradingModelInterface
from typing import List, Dict, Any

import pandas as pd

class TradingFeatureSystem:
    """
    Système complet qui combine le dispatcher et facilite l'utilisation.
    Cette classe remplace ton FeatureEngineering en ajoutant l'optimisation
    et la flexibilité du nouveau système.
    """
    
    def __init__(self):
        self.registry = FeatureRegistry()
        self.dispatcher = FeatureDispatcher(self.registry)
    
    def compute_features_for_models(self, data: pd.DataFrame, models: List[TradingModelInterface]) -> Dict[str, Dict[FeatureID, Any]]:
        """
        Calcule les features pour plusieurs modèles en une seule fois.
        Cette méthode optimise en mutualisant les features partagées.
        
        Args:
            data: DataFrame avec les données OHLCV
            models: Liste des modèles nécessitant des features
            
        Returns:
            Dict avec les features pour chaque modèle
        """
        # Collecter toutes les features requises par tous les modèles
        all_required = set()
        for model in models:
            all_required.update(model.get_required_features())
        
        # Calculer toutes les features en une seule fois (optimisation clé!)
        all_features = self.dispatcher.compute_features(data, all_required)
        
        # Distribuer les features à chaque modèle
        result = {}
        for i, model in enumerate(models):
            model_name = f"{type(model).__name__}_{i}"
            model_features = {fid: all_features[fid] for fid in model.get_required_features()}
            result[model_name] = model_features
        
        return result
    
    def get_trading_pipeline_predictions(self, data: pd.DataFrame, 
                                       opportunity_model: TradingModelInterface,
                                       direction_model: TradingModelInterface, 
                                       amplitude_model: TradingModelInterface) -> Dict[str, Any]:
        """
        Exécute ton pipeline de trading complet : opportunité -> direction -> amplitude.
        Cette méthode reproduit ta logique cascade avec l'optimisation des features.
        """
        models = [opportunity_model, direction_model, amplitude_model]
        
        # Calculer toutes les features nécessaires en une fois
        features_by_model = self.compute_features_for_models(data, models)
        
        # Étape 1: Détecter l'opportunité
        opportunity = opportunity_model.predict(features_by_model[f"{type(opportunity_model).__name__}_0"])
        
        result = {
            'opportunity': opportunity,
            'direction': None,
            'amplitude': None,
            'trade_decision': 'NO_TRADE'
        }
        
        if opportunity:
            # Étape 2: Prédire la direction
            direction = direction_model.predict(features_by_model[f"{type(direction_model).__name__}_1"])
            result['direction'] = direction
            
            # Étape 3: Estimer l'amplitude
            amplitude = amplitude_model.predict(features_by_model[f"{type(amplitude_model).__name__}_2"])
            result['amplitude'] = amplitude
            
            result['trade_decision'] = f"TRADE_{direction}_size_{amplitude:.2f}"
        
        return result