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

    def __init__(self, min_amplitude_threshold=0.05) -> None:
        """
        Initialise la stratégie avec les modèles nécessaires.

        :param min_amplitude_threshold: amplitude prédite minimale pour trader, en % (même unité que
                                        Target_Amplitude et Target_Opportunity.threshold_pct).
        """
        super().__init__()
        self.IsRunning = False
        self.strategy_id = 1
        self.symbol = "AAPL"
        self.min_amplitude_threshold = min_amplitude_threshold

        self.featureManager = FeatureManager()
        self.targetManager = TargetManager()

        # Initialisation des modèles de trading
        self.AmplitudeModel = AmplitudeModel()
        self.DirectionModel = DirectionModel()
        self.OpportunityModel = OpportunityModel()

        self.training_data_shift = 30
        self.current_data = pd.DataFrame()


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
        self.featureManager.register_indicator(FeatureID.MACD, MACD(short_window=12, long_window=26, signal_window=9))
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

        self.current_data = training_data.copy().iloc[:-self.training_data_shift]
        self.featureManager.update(self.current_data)
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
        amplitude_accuracy = self.AmplitudeModel.test()
        direction_accuracy = self.DirectionModel.test()
        opportunity_accuracy = self.OpportunityModel.test()

        amplitude_cv, _ = self.AmplitudeModel.cross_validate()
        direction_cv, _ = self.DirectionModel.cross_validate()
        opportunity_cv, _ = self.OpportunityModel.cross_validate()

        print(f"\n--- Résultats de l'évaluation ---")
        print(f"Amplitude Model   | Train R2: {amplitude_accuracy:.2f} | CV R2: {amplitude_cv:.2f}")
        print(f"Direction Model   | Train Acc: {direction_accuracy:.2f} | CV Acc: {direction_cv:.2f}")
        print(f"Opportunity Model | Train Acc: {opportunity_accuracy:.2f} | CV Acc: {opportunity_cv:.2f}")
        print(f"Note: Un score sur le train proche de 1.0 indique que le modèle a mémorisé les données par cœur. Fiez-vous aux scores de validation croisée (CV).\n")


    def on_bar(self, bar) -> list[Order]:
        """
        Exécute la stratégie de trading.
        """
        if not self.IsRunning:
            raise RuntimeError("La stratégie doit être lancée.")

        # 1. Mise à jour de l'historique de données (fenêtre glissante pour la performance)
        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-200:]
        
        # 2. Mise à jour des indicateurs pour pouvoir prédire sur cette nouvelle bougie
        self.featureManager.update(self.current_data)

        # Implémentation de la logique de la stratégie en cascade
        # 1. Y a-t-il une opportunité ?
        has_opportunity = self.OpportunityModel.predict()

        if not has_opportunity:
            return []
        
        # 2. Quelle direction ?
        direction = self.DirectionModel.predict()  # 1 pour UP ou 0 pour DOWN
        
        # 3. Quelle amplitude ?
        amplitude = self.AmplitudeModel.predict()  # en %
        
        # 4. Vérifier si l'amplitude justifie le trade
        if abs(amplitude) < self.min_amplitude_threshold:
            return []
        
        # 5. Vérifier la cohérence direction/amplitude
        direction_matches_amplitude = (
            (direction == 'UP' and amplitude > 0) or
            (direction == 'DOWN' and amplitude < 0)
        )
        
        if not direction_matches_amplitude:
            # Conflit entre les modèles
            return []
        
        # 6. Tout est cohérent → Trade !
        side = 'BUY' if direction == 'UP' else 'SELL'
        
        # Confiance basée sur l'amplitude (plus c'est grand, plus on est confiant)
        confidence = min(abs(amplitude) / 5.0, 1.0)  # Plafond à 5%

        # Quantité calculée à partir de la confiance (arbitraire ici, au minimum 1)
        quantity = max(1, int(confidence * 10))
        
        return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side=side, quantity=quantity)]


    def stop(self):
        """
        Arrête la stratégie de trading.
        """
        self.IsRunning = False
    

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running


    def get_status(self) -> str:
        """
        Retourne l'état actuel de la stratégie de trading.

        :return: Dictionnaire contenant l'état de la stratégie.
        """
        return self.IsRunning
    