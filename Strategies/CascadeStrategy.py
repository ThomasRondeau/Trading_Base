from datetime import datetime, time, timedelta
import pandas as pd
from Connections.TradingApp import TradingApp
from Models import AmplitudeModel, DirectionModel, OpportunityModel
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from typing import Dict, Any
from Features import *
from Targets import *
from Types import *

class CascadeStrategy(TradingStrategyInterface):
    """
    Implémente une stratégie de trading en cascade.
    Cette stratégie exécute des actions de trading en fonction des signaux fournis par les modèles.
    """

    def __init__(self) -> None:
        """
        Initialise la stratégie avec les modèles nécessaires.

        :param models: Dictionnaire contenant les modèles de trading.
        """
        self.IsRunning = False

        self.featureManager = FeatureManager()
        self.targetManager = TargetManager()

        # Initialisation des modèles de trading
        self.AmplitudeModel = AmplitudeModel()
        self.DirectionModel = DirectionModel()
        self.OpportunityModel = OpportunityModel()

        self.training_data_shift = 30


    def train(self, training_data: pd.DataFrame) -> None:
        """
        Entraîne les modèles de la stratégie avec les données d'entraînement.

        :param training_data: Dictionnaire contenant les données d'entraînement.
        """
        ## Enregistrement des indicateurs dans le FeatureManager

        #Base Indicators
        self.featureManager.register_indicator(FeatureID.SMA_5, SMA(window=5))
        self.featureManager.register_indicator(FeatureID.SMA_10, SMA(window=10))
        self.featureManager.register_indicator(FeatureID.SMA_20, SMA(window=20))
        self.featureManager.register_indicator(FeatureID.RSI, RSI(window=14))
        self.featureManager.register_indicator(FeatureID.MACD, MACD(long_window=12, short_window=26, signal_window=9))
        self.featureManager.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        self.featureManager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.featureManager.register_indicator(FeatureID.VOLUME, VOLUME(window=14))

        #Models Indicators
        self.featureManager.register_indicator(FeatureID.AMPLITUDE, AmplitudeFeatures())
        self.featureManager.register_indicator(FeatureID.DIRECTION, DirectionFeatures())
        self.featureManager.register_indicator(FeatureID.OPPORTUNITY, OpportunityFeatures())
        
        self.featureManager.clear_cache()

        ## Enregistrement des indicateurs dans le Target Manager

        # Target Base
        self.targetManager.register_target(TargetID.SHIFTED_PLUS_1, Target_Plus_1())
        self.targetManager.register_target(TargetID.SHIFTED_PLUS_5, Target_Plus_5())
        self.targetManager.register_target(TargetID.SHIFTED_PLUS_10, Target_Plus_10())
        self.targetManager.register_target(TargetID.SHIFTED_PLUS_20, Target_Plus_20())
        self.targetManager.register_target(TargetID.DIRECTION, Target_Direction())
        self.targetManager.register_target(TargetID.OPPORTUNITY, Target_Opportunity())
        self.targetManager.register_target(TargetID.AMPLITUDE, Target_Amplitude())

        self.targetManager.clear_cache()

        training_data_features = training_data.copy().iloc[:-self.training_data_shift]
        self.featureManager.update(training_data_features)
        self.targetManager.update(training_data, training_data_shift=self.training_data_shift)
        
        self.AmplitudeModel.setFeatureManager(self.featureManager)
        self.AmplitudeModel.setTargetManager(self.targetManager)
        self.DirectionModel.setFeatureManager(self.featureManager)
        self.DirectionModel.setTargetManager(self.targetManager)
        self.OpportunityModel.setFeatureManager(self.featureManager)
        self.OpportunityModel.setTargetManager(self.targetManager)
        
        self.AmplitudeModel.train()
        self.DirectionModel.train()
        self.OpportunityModel.train()

    def test(self) -> None:
        """
        Teste les modèles de la stratégie avec les données de test.

        :param test_data: Dictionnaire contenant les données de test.
        """
        pass


    def on_bar(self, bar) -> list[Order]:
        """
        Exécute la stratégie de trading.
        """
        if self.IsRunning:
            raise RuntimeError("La stratégie est déjà en cours d'exécution.")
        
        self.IsRunning = True

        # Implémentation de la logique de la stratégie en cascade

        # 1. Y a-t-il une opportunité ?
        has_opportunity = self.OpportunityModel.predict()
        
        if not has_opportunity:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'expected_return': 0.0,
                'reasoning': "Pas d'opportunité détectée (mouvement insuffisant attendu)"
            }
        
        # 2. Quelle direction ?
        direction = self.DirectionModel.predict()  # 1 pour UP ou 0 pour DOWN
        
        # 3. Quelle amplitude ?
        amplitude = self.AmplitudeModel.predict()  # en %
        
        # 4. Vérifier si l'amplitude justifie le trade
        if abs(amplitude) < self.min_amplitude_threshold:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'expected_return': amplitude,
                'reasoning': f"Opportunité détectée mais amplitude trop faible ({amplitude:.2f}%)"
            }
        
        # 5. Vérifier la cohérence direction/amplitude
        direction_matches_amplitude = (
            (direction == 'UP' and amplitude > 0) or
            (direction == 'DOWN' and amplitude < 0)
        )
        
        if not direction_matches_amplitude:
            # Conflit entre les modèles
            return {
                'action': 'HOLD',
                'confidence': 0.3,
                'expected_return': amplitude,
                'reasoning': f"Conflit: Direction={direction} mais Amplitude={amplitude:.2f}%"
            }
        
        # 6. Tout est cohérent → Trade !
        action = 'BUY' if direction == 'UP' else 'SELL'
        
        # Confiance basée sur l'amplitude (plus c'est grand, plus on est confiant)
        confidence = min(abs(amplitude) / 5.0, 1.0)  # Plafond à 5%

        
        pass


    def stop(self):
        """
        Arrête la stratégie de trading.
        """
        self.IsRunning = False
    
    
    def get_status(self) -> str:
        """
        Retourne l'état actuel de la stratégie de trading.

        :return: Dictionnaire contenant l'état de la stratégie.
        """
        return "CascadeStrategy is running"
    