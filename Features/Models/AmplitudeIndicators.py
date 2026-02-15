from typing import cast
import pandas as pd
import numpy as np
from Features.FeatureID import FeatureID
from Features import *

class AmplitudeIndicators(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):

        self.data = data
        self.current_volatility = None
        self.volatility_percentile = None
        self.distance_to_sma = None
        self.distance_to_bb = None
        self.rsi_extreme = None
        self.macd_intensity = None
        self.momentum_strength = None
        self.volume_intensity = None

    def dependencies(self):
        """
        Retourne les dépendances nécessaires pour le calcul de l'amplitude.
        """
        return {
            FeatureID.ATR,
            FeatureID.SMA_20,
            FeatureID.BOLLINGER_BANDS,
            FeatureID.RSI,
            FeatureID.MACD,
            FeatureID.VOLUME
        }

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface]):

        Atr = cast(ATR, resolved_dependencies.get(FeatureID.ATR))
        Sma_20 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_20))
        Bollinger_Bands = cast(BollingerBands, resolved_dependencies.get(FeatureID.BOLLINGER_BANDS))
        Rsi = cast(RSI, resolved_dependencies.get(FeatureID.RSI))
        Macd = cast(MACD, resolved_dependencies.get(FeatureID.MACD))
        Volume = cast(VOLUME, resolved_dependencies.get(FeatureID.VOLUME))

        if not Atr or not Sma_20 or not Bollinger_Bands or not Rsi or not Macd or not Volume:
            raise ValueError("Erreur dans AmplitudeIndicators : Les dépendances nécessaires ne sont pas résolues.")


        self.current_volatility = Atr.atr / self.data['close']
        self.volatility_percentile = Atr.atr.rolling(window=50).rank(pct=True)
        self.distance_to_sma = abs(self.data['close'] - Sma_20.sma) / self.data['close']
        self.distance_to_bb = np.minimum(
            abs(self.data['close'] - Bollinger_Bands.upper_band),
            abs(self.data['close'] - Bollinger_Bands.lower_band)
        ) / self.data['close']
        self.rsi_extreme = np.maximum(Rsi.rsi - 70, 30 - Rsi.rsi).clip(0)
        self.macd_intensity = abs(Macd.macd_histogram) / Atr.atr
        self.momentum_strength = abs(self.data['close'].pct_change(5))
        self.volume_intensity = Volume.Volume_ratio - self.current_volatility

        #Il va falloir que je fasse un create labels pour que le modèle apprenne les features en fonction des targets 

    def update(self, new_data: pd.DataFrame):
        """
        Met à jour l'ATR avec de nouvelles données.

        :param new_data: DataFrame contenant les nouvelles données à ajouter.
        """
        if not all(col in new_data for col in ['high', 'low', 'close']):
            raise ValueError("Erreur dans ATR : Les colonnes 'high', 'low' et 'close' doivent être présentes dans les nouvelles données.")

        self.data = new_data

    def clear_cache(self):
        """
        Vide le cache des indicateurs.
        """
        self.current_volatility = None
        self.volatility_percentile = None
        self.distance_to_sma = None 
        self.distance_to_bb = None
        self.rsi_extreme = None
        self.macd_intensity = None
        self.momentum_strength = None
        self.volume_intensity = None

        self.data = pd.DataFrame()