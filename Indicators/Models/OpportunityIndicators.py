from typing import cast
import pandas as pd
from Features.FeatureID import FeatureID
from Indicators import *

class OpportunityIndicators(IndicatorInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):

        self.data = data
        self.volatility_compression = None
        self.volatility_expansion = None
        self.volatility_change = None
        self.volume_spike = None
        self.volume_trend = None
        self.momentum_alignment = None
        self.near_support = None
        self.near_resistance = None
        self.close_to_sma = None
        self.price_acceleration = None

    def dependencies(self):
        """
        Retourne les dépendances nécessaires pour le calcul de l'amplitude.
        """
        return {
            FeatureID.ATR,
            FeatureID.SMA_5,
            FeatureID.SMA_20,
            FeatureID.BOLLINGER_BANDS,
            FeatureID.RSI,
            FeatureID.MACD,
            FeatureID.VOLUME
        }

    def calculate(self, resolved_dependencies: dict[FeatureID, IndicatorInterface]):

        Atr = cast(ATR, resolved_dependencies.get(FeatureID.ATR))
        Sma_5 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_5))
        Sma_20 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_20))
        Bollinger_Bands = cast(BollingerBands, resolved_dependencies.get(FeatureID.BOLLINGER_BANDS))
        Rsi = cast(RSI, resolved_dependencies.get(FeatureID.RSI))
        Macd= cast(MACD, resolved_dependencies.get(FeatureID.MACD))
        Volume = cast(VOLUME, resolved_dependencies.get(FeatureID.VOLUME))

        if not Atr or not Sma_5 or not Sma_20 or not Bollinger_Bands or not Rsi or not Macd or not Volume:
            raise ValueError("Erreur dans AmplitudeIndicators : Les dépendances nécessaires ne sont pas résolues.")

        self.volatility_compression = (Bollinger_Bands.width < Bollinger_Bands.width.rolling(window=10).mean()).astype(int)
        self.volatility_expansion = (Atr.atr > Atr.atr.shift(1) * 1.2).astype(int)
        self.volatility_change = Atr.atr.pct_change()
        self.volume_spike = (Volume.Volume_ratio > 1.5).astype(int) # Revoir car pas top de mettre un seuil fixe
        self.volume_trend = Volume.Volume.rolling(window=5).mean() / Volume.Volume.rolling(window=20).mean()
        self.momentum_alignment = ((Macd.macd > Macd.signal_line) & 
                                   (Rsi.rsi > 50) & 
                                   (self.data['close'] > Sma_5.sma)).astype(int)
        self.near_support = (self.data['close'] < Bollinger_Bands.lower_band * 1.02).astype(int)
        self.near_resistance = (self.data['close'] > Bollinger_Bands.upper_band * 0.98).astype(int)
        self.close_to_sma = abs(self.data['close'] - Sma_20.sma) / Atr.atr
        self.price_acceleration = self.data['close'].pct_change(2)



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
        self.volatility_compression = None
        self.volatility_expansion = None
        self.volatility_change = None
        self.volume_spike = None
        self.volume_trend = None
        self.momentum_alignment = None
        self.near_support = None
        self.near_resistance = None
        self.close_to_sma = None
        self.price_acceleration = None
        
        self.data = pd.DataFrame()