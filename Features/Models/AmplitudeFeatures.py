from typing import cast
import pandas as pd
import numpy as np
from Features import *

class AmplitudeFeatures(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):
        self.data = data
        self.volatility = None
        self.dist_sma = None
        self.bb_width = None
        self.rsi_extreme = None
        self.macd_hist_abs = None
        self.momentum_5 = None
        self.volume_ratio = None

    def dependencies(self):
        return {
            FeatureID.ATR,
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
        Macd = cast(MACD, resolved_dependencies.get(FeatureID.MACD))
        Volume = cast(VOLUME, resolved_dependencies.get(FeatureID.VOLUME))

        if not Atr or not Sma_20 or not Bollinger_Bands or not Rsi or not Macd or not Volume:
            raise ValueError("Erreur dans AmplitudeFeatures : Les dépendances nécessaires ne sont pas résolues.")

        close_prices = pd.to_numeric(self.data['close'], errors='coerce')

        self.volatility = Atr.atr / close_prices
        self.dist_sma = abs(close_prices - Sma_20.sma) / close_prices
        self.bb_width = (Bollinger_Bands.upper_band - Bollinger_Bands.lower_band) / close_prices
        self.rsi_extreme = abs(Rsi.rsi - 50)
        self.macd_hist_abs = abs(Macd.macd_histogram) / close_prices
        self.momentum_5 = abs(close_prices.pct_change(5))
        self.volume_ratio = Volume.Volume_ratio


    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.volatility = None
        self.dist_sma = None 
        self.bb_width = None
        self.rsi_extreme = None
        self.macd_hist_abs = None
        self.momentum_5 = None
        self.volume_ratio = None
        self.data = pd.DataFrame()
