from typing import cast
import numpy as np
import pandas as pd
from Features.FeatureID import FeatureID
from Features import *

class DirectionFeatures(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):

        self.data = data
        self.rsi_momentum = None
        self.macd_momentum = None
        self.price_vs_sma = None
        self.sma_slope = None
        self.price_position = None
        self.upper_breakout = None
        self.lower_breakout = None
        self.bullish_confluence = None
        self.bearish_confluence = None
        self.volume_price_trend = None


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

    def calculate(self, resolved_dependencies: dict[FeatureID, FeatureInterface]):
        Atr = cast(ATR, resolved_dependencies.get(FeatureID.ATR))
        Sma_20 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_20))
        Bollinger_Bands = cast(BollingerBands, resolved_dependencies.get(FeatureID.BOLLINGER_BANDS))
        Rsi = cast(RSI, resolved_dependencies.get(FeatureID.RSI))
        Macd= cast(MACD, resolved_dependencies.get(FeatureID.MACD))
        Volume = cast(VOLUME, resolved_dependencies.get(FeatureID.VOLUME))

        if not Atr or not Sma_20 or not Bollinger_Bands or not Rsi or not Macd or not Volume:
            raise ValueError("Erreur dans DirectionFeatures : Les dépendances nécessaires ne sont pas résolues.")


        self.rsi_momentum = (Rsi.rsi > 50 ).astype(int)
        self.macd_momentum = (Macd.macd > Macd.signal_line).astype(int)
        self.price_vs_sma = (self.data['close'] > Sma_20.sma).astype(int)
        self.sma_slope = Sma_20.sma.diff(5)
        self.price_position = (self.data['close'] - Sma_20.sma) / Atr.atr
        self.upper_breakout = (self.data['close'] > Bollinger_Bands.upper_band).astype(int)
        self.lower_breakout = (self.data['close'] < Bollinger_Bands.lower_band).astype(int)
        self.bullish_confluence = (self.rsi_momentum & self.macd_momentum & self.price_vs_sma).astype(int)
        self.bearish_confluence = ((1 - self.rsi_momentum) & (1 - self.macd_momentum) & (1 - self.price_vs_sma)).astype(int)
        self.volume_price_trend = Volume.Volume_ratio * np.sign(self.data['close'].pct_change())


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
        self.rsi_momentum = None
        self.macd_momentum = None
        self.price_vs_sma = None
        self.distance_to_bb = None
        self.rsi_extreme = None
        self.macd_intensity = None
        self.momentum_strength = None
        self.volume_intensity = None
        
        self.data = pd.DataFrame()