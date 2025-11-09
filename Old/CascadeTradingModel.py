import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import pickle
import warnings
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import xgboost as xgb

from Connections.Trading_App import TradingApp
from Old.FeatureEngineering import FeatureEngineering

class CascadeTradingModel:
    """
    Modèle de trading utilisant l'architecture cascade.
    
    Cette classe implémente les trois modèles de notre cascade :
    1. Détecteur d'opportunité (Random Forest)
    2. Prédicteur de direction (XGBoost)
    3. Estimateur d'amplitude (Random Forest)
    """
    
    def __init__(self):
        self.feature_engine = FeatureEngineering()
        
        # Modèles de la cascade
        self.opportunity_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.direction_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.amplitude_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # État des modèles
        self.models_trained = False
        self.training_stats = {}
    
    def collect_training_data(self, app: TradingApp, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Collecte les données d'entraînement pour une période donnée.
        
        Cette fonction utilise ton infrastructure existante pour récupérer
        les données historiques nécessaires à l'entraînement.
        
        Args:
            app: Instance de TradingApp connectée
            symbol: Symbole à trader (ex: "SPY")
            days_back: Nombre de jours de données à récupérer
            
        Returns:
            DataFrame consolidé avec toutes les données
        """
        print(f"Collecte des données d'entraînement pour {symbol} sur {days_back} jours...")
        
        all_data = []
        contract = TradingApp.get_contract(symbol)
        
        # Récupération des données jour par jour
        for day in range(1, days_back + 1):
            try:
                # Calcul de la date pour ce jour
                target_date = datetime.now() - timedelta(days=day)
                date_str = target_date.strftime("%Y%m%d 20:00:00 US/Eastern")
                
                # Récupération des données pour ce jour
                daily_data = app.get_historical_data(99, contract, end_date=date_str)
                
                if not daily_data.empty:
                    daily_data['date'] = target_date.date()
                    all_data.append(daily_data)
                    print(f"Données récupérées pour {target_date.date()}: {len(daily_data)} points")
                else:
                    print(f"Pas de données pour {target_date.date()}")
                    
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour {target_date.date()}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Aucune donnée n'a pu être récupérée")
        
        # Consolidation des données
        consolidated_data = pd.concat(all_data, ignore_index=True)
        consolidated_data = consolidated_data.sort_values(by='date').reset_index(drop=True)
        
        print(f"Données consolidées: {len(consolidated_data)} points sur {len(all_data)} jours")
        
        return consolidated_data
    
    def train_models(self, data: pd.DataFrame):
        """
        Entraîne les trois modèles de la cascade.
        
        Cette fonction suit la logique d'entraînement séquentiel :
        1. Entraîne d'abord le détecteur d'opportunité
        2. Utilise ses prédictions pour filtrer les données du prédicteur de direction
        3. Entraîne l'estimateur d'amplitude sur toutes les données
        
        Args:
            data: DataFrame avec les données d'entraînement
        """
        print("Début de l'entraînement des modèles...")
        
        # Calcul des indicateurs techniques
        data_with_indicators = self.feature_engine.calculate_technical_indicators(data)
        
        # Création des features pour chaque modèle
        opportunity_features = self.feature_engine.create_opportunity_features(data_with_indicators)
        direction_features = self.feature_engine.create_direction_features(data_with_indicators)
        amplitude_features = self.feature_engine.create_amplitude_features(data_with_indicators)
        
        # Création des labels
        data_with_labels = self.feature_engine.create_labels(data_with_indicators)
        
        # Fusion de toutes les features
        all_features = pd.concat([
            opportunity_features,
            direction_features,
            amplitude_features,
            data_with_labels[['opportunity_label', 'direction_label', 'amplitude_label']]
        ], axis=1)

        # Suppression des lignes avec des valeurs manquantes
        all_features = all_features.dropna() #TODO Truc de mongole parce que l'on met des NaN dans les features donc on saute des ligne qu'on aurait pu utiliser
        
        print(f"Données d'entraînement après nettoyage: {len(all_features)} échantillons")
        
        # 1. Entraînement du détecteur d'opportunité
        print("Entraînement du détecteur d'opportunité...")
        
        opportunity_X = self.feature_engine.prepare_features_for_model(all_features, 'opportunity')
        opportunity_y = all_features['opportunity_label']
        
        # Division des données avec préservation de l'ordre temporel
        split_idx = int(len(opportunity_X) * 0.8)
        
        opp_X_train, opp_X_test = opportunity_X[:split_idx], opportunity_X[split_idx:]
        opp_y_train, opp_y_test = opportunity_y[:split_idx], opportunity_y[split_idx:]
        
        # Entraînement
        self.opportunity_model.fit(opp_X_train, opp_y_train) 
        
        # Évaluation
        opp_train_pred = self.opportunity_model.predict(opp_X_train)
        opp_test_pred = self.opportunity_model.predict(opp_X_test)
        
        self.training_stats['opportunity'] = {
            'train_accuracy': accuracy_score(opp_y_train, opp_train_pred),
            'test_accuracy': accuracy_score(opp_y_test, opp_test_pred),
            'train_samples': len(opp_X_train),
            'test_samples': len(opp_X_test),
            'positive_samples': sum(opportunity_y),
            'feature_importance': dict(zip(opportunity_X.columns, self.opportunity_model.feature_importances_))
        }
        
        print(f"Détecteur d'opportunité - Précision train: {self.training_stats['opportunity']['train_accuracy']:.3f}")
        print(f"Détecteur d'opportunité - Précision test: {self.training_stats['opportunity']['test_accuracy']:.3f}")
        
        # 2. Entraînement du prédicteur de direction (seulement sur les opportunités)
        print("Entraînement du prédicteur de direction...")
        
        # Filtrage des données où il y a une opportunité
        direction_mask = all_features['opportunity_label'] == 1
        direction_data = all_features[direction_mask]
        
        print(direction_data.columns)

        if len(direction_data) == 0:
            raise ValueError("Aucune opportunité détectée dans les données d'entraînement")
        
        direction_X = self.feature_engine.prepare_features_for_model(direction_data, 'direction')
        direction_y = direction_data['direction_label'].dropna()
        
        # Alignement des indices
        common_idx = direction_X.index.intersection(direction_y.index)
        direction_X = direction_X.loc[common_idx]
        direction_y = direction_y.loc[common_idx]
        
        if len(direction_X) == 0:
            raise ValueError("Aucune donnée valide pour l'entraînement du prédicteur de direction")
        
        # Division des données
        split_idx = int(len(direction_X) * 0.8)
        
        dir_X_train, dir_X_test = direction_X[:split_idx], direction_X[split_idx:]
        dir_y_train, dir_y_test = direction_y[:split_idx], direction_y[split_idx:]

        dir_X_train = dir_X_train.values
        dir_y_train = dir_y_train.values
        dir_X_test = dir_X_test.values
        dir_y_test = dir_y_test.values

        dir_X_train = np.array(dir_X_train, dtype=np.float32)
        dir_y_train = np.array(dir_y_train, dtype=np.int32)
        dir_X_test = np.array(dir_X_test, dtype=np.float32)
        dir_y_test = np.array(dir_y_test, dtype=np.int32)


        # Entraînement
        print("Dir x train")
        print(dir_X_train)
        print("Dir y train")
        print(dir_y_train)
        self.direction_model.fit(dir_X_train, dir_y_train)
        
        # Évaluation
        dir_train_pred = self.direction_model.predict(dir_X_train)
        dir_test_pred = self.direction_model.predict(dir_X_test)
        
        self.training_stats['direction'] = {
            'train_accuracy': accuracy_score(dir_y_train, dir_train_pred),
            'test_accuracy': accuracy_score(dir_y_test, dir_test_pred),
            'train_samples': len(dir_X_train),
            'test_samples': len(dir_X_test),
            'bullish_samples': sum(direction_y),
            'feature_importance': dict(zip(direction_X.columns, self.direction_model.feature_importances_))
        }
        
        print(f"Prédicteur de direction - Précision train: {self.training_stats['direction']['train_accuracy']:.3f}")
        print(f"Prédicteur de direction - Précision test: {self.training_stats['direction']['test_accuracy']:.3f}")
        
        # 3. Entraînement de l'estimateur d'amplitude
        print("Entraînement de l'estimateur d'amplitude...")
        
        amplitude_X = self.feature_engine.prepare_features_for_model(all_features, 'amplitude')
        amplitude_y = all_features['amplitude_label']
        
        # Division des données
        split_idx = int(len(amplitude_X) * 0.8)  
        
        amp_X_train, amp_X_test = amplitude_X[:split_idx], amplitude_X[split_idx:]
        amp_y_train, amp_y_test = amplitude_y[:split_idx], amplitude_y[split_idx:]
        
        # Entraînement
        self.amplitude_model.fit(amp_X_train, amp_y_train)
        
        # Évaluation
        amp_train_pred = self.amplitude_model.predict(amp_X_train)
        amp_test_pred = self.amplitude_model.predict(amp_X_test)
        
        self.training_stats['amplitude'] = {
            'train_mae': mean_absolute_error(amp_y_train, amp_train_pred),
            'test_mae': mean_absolute_error(amp_y_test, amp_test_pred),
            'train_samples': len(amp_X_train),
            'test_samples': len(amp_X_test),
            'feature_importance': dict(zip(amplitude_X.columns, self.amplitude_model.feature_importances_))
        }
        
        print(f"Estimateur d'amplitude - MAE train: {self.training_stats['amplitude']['train_mae']:.5f}")
        print(f"Estimateur d'amplitude - MAE test: {self.training_stats['amplitude']['test_mae']:.5f}")
        
        self.models_trained = True
        print("Entraînement terminé avec succès!")
    
    def predict(self, current_data: pd.DataFrame) -> Dict:
        """
        Fait une prédiction en utilisant la cascade de modèles.
        
        Cette fonction applique les trois modèles dans l'ordre :
        1. Détecte s'il y a une opportunité
        2. Si oui, prédit la direction
        3. Estime l'amplitude du mouvement
        
        Args:
            current_data: DataFrame avec les données actuelles
            
        Returns:
            Dict avec les prédictions de chaque modèle
        """
        if not self.models_trained:
            raise ValueError("Les modèles doivent être entraînés avant de faire des prédictions")
        
        # Calcul des indicateurs techniques
        data_with_indicators = self.feature_engine.calculate_technical_indicators(current_data)
        
        # Création des features
        opportunity_features = self.feature_engine.create_opportunity_features(data_with_indicators)
        direction_features = self.feature_engine.create_direction_features(data_with_indicators)
        amplitude_features = self.feature_engine.create_amplitude_features(data_with_indicators)
        
        # Fusion des features
        all_features = pd.concat([
            opportunity_features,
            direction_features,
            amplitude_features
        ], axis=1)
        
        # Préparation des features pour chaque modèle
        opp_X = self.feature_engine.prepare_features_for_model(all_features, 'opportunity')
        dir_X = self.feature_engine.prepare_features_for_model(all_features, 'direction')
        amp_X = self.feature_engine.prepare_features_for_model(all_features, 'amplitude')
        
        # Prise de la dernière observation (la plus récente)
        if len(opp_X) == 0:
            return {'opportunity': 0, 'direction': None, 'amplitude': 0, 'confidence': 0}
        
        latest_opp = opp_X.iloc[-1:].fillna(0)
        latest_dir = dir_X.iloc[-1:].fillna(0)
        latest_amp = amp_X.iloc[-1:].fillna(0)
        
        # 1. Prédiction d'opportunité
        opp_pred = self.opportunity_model.predict(latest_opp)[0]
        opp_prob = self.opportunity_model.predict_proba(latest_opp)[0]
        
        result = {
            'opportunity': opp_pred,
            'opportunity_confidence': max(opp_prob),
            'direction': None,
            'direction_confidence': None,
            'amplitude': None,
            'timestamp': datetime.now()
        }
        
        # 2. Si opportunité détectée, prédire la direction
        if opp_pred == 1:
            dir_pred = self.direction_model.predict(latest_dir)[0]
            dir_prob = self.direction_model.predict_proba(latest_dir)[0]
            
            result['direction'] = dir_pred  # 1 pour bullish, 0 pour bearish
            result['direction_confidence'] = max(dir_prob)
            
            # 3. Estimation de l'amplitude
            amp_pred = self.amplitude_model.predict(latest_amp)[0]
            result['amplitude'] = amp_pred
        
        return result
    
    def save_models(self, filepath: str):
        """Sauvegarde les modèles entraînés."""
        model_data = {
            'opportunity_model': self.opportunity_model,
            'direction_model': self.direction_model,
            'amplitude_model': self.amplitude_model,
            'feature_engine': self.feature_engine,
            'training_stats': self.training_stats,
            'models_trained': self.models_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modèles sauvegardés dans {filepath}")
    
    
    def load_models(self, filepath: str):
        """
        Charge les modèles précédemment sauvegardés.
        
        Cette méthode restaure l'état complet du système de trading,
        incluant les modèles entraînés et leurs statistiques.
        
        Args:
            filepath: Chemin vers le fichier de sauvegarde
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restauration des modèles
            self.opportunity_model = model_data['opportunity_model']
            self.direction_model = model_data['direction_model']
            self.amplitude_model = model_data['amplitude_model']
            self.feature_engine = model_data['feature_engine']
            self.training_stats = model_data['training_stats']
            self.models_trained = model_data['models_trained']
            
            print(f"Modèles chargés avec succès depuis {filepath}")
            print(f"Statut d'entraînement: {self.models_trained}")
            
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement des modèles: {e}")
    
    def get_model_summary(self) -> Dict:
        """
        Retourne un résumé des performances des modèles.
        
        Cette méthode fournit une vue d'ensemble des statistiques
        d'entraînement et peut être utilisée pour le monitoring.
        
        Returns:
            Dict avec les statistiques de performance
        """
        if not self.models_trained:
            return {"error": "Modèles non entraînés"}
        
        summary = {
            "models_trained": self.models_trained,
            "training_timestamp": datetime.now().isoformat(),
            "performance_summary": {}
        }
        
        # Résumé pour chaque modèle
        for model_name, stats in self.training_stats.items():
            if model_name == 'opportunity':
                summary["performance_summary"][model_name] = {
                    "type": "classification",
                    "train_accuracy": stats['train_accuracy'],
                    "test_accuracy": stats['test_accuracy'],
                    "total_samples": stats['train_samples'] + stats['test_samples'],
                    "positive_rate": stats['positive_samples'] / (stats['train_samples'] + stats['test_samples']),
                    "top_features": sorted(stats['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                }
            elif model_name == 'direction':
                summary["performance_summary"][model_name] = {
                    "type": "classification",
                    "train_accuracy": stats['train_accuracy'],
                    "test_accuracy": stats['test_accuracy'],
                    "total_samples": stats['train_samples'] + stats['test_samples'],
                    "bullish_rate": stats['bullish_samples'] / (stats['train_samples'] + stats['test_samples']),
                    "top_features": sorted(stats['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                }
            elif model_name == 'amplitude':
                summary["performance_summary"][model_name] = {
                    "type": "regression",
                    "train_mae": stats['train_mae'],
                    "test_mae": stats['test_mae'],
                    "total_samples": stats['train_samples'] + stats['test_samples'],
                    "top_features": sorted(stats['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                }
        
        return summary