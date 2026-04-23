from typing import cast
import pandas as pd
import numpy as np
from Features import *

class OpportunityFeatures(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):
        self.data = data
        self.bb_width_norm = None
        self.volatility_ratio = None
        self.volatility_change = None
        self.volume_ratio = None
        self.volume_trend = None
        self.momentum_alignment = None
        self.dist_lower_bb = None
        self.dist_upper_bb = None
        self.dist_sma = None
        self.price_accel = None

    def dependencies(self):
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
        Sma_5 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_5))
        Sma_20 = cast(SMA, resolved_dependencies.get(FeatureID.SMA_20))
        Bollinger_Bands = cast(BollingerBands, resolved_dependencies.get(FeatureID.BOLLINGER_BANDS))
        Rsi = cast(RSI, resolved_dependencies.get(FeatureID.RSI))
        Macd= cast(MACD, resolved_dependencies.get(FeatureID.MACD))
        Volume = cast(VOLUME, resolved_dependencies.get(FeatureID.VOLUME))

        if not Atr or not Sma_5 or not Sma_20 or not Bollinger_Bands or not Rsi or not Macd or not Volume:
            raise ValueError("Erreur dans OpportunityFeatures : Les dépendances nécessaires ne sont pas résolues.")

        close_prices = pd.to_numeric(self.data['close'], errors='coerce')

        # Continuous indicators
        self.bb_width_norm = (Bollinger_Bands.upper_band - Bollinger_Bands.lower_band) / close_prices
        self.volatility_ratio = Atr.atr / Atr.atr.rolling(window=20).mean()
        self.volatility_change = Atr.atr.pct_change(3)
        self.volume_ratio = Volume.Volume_ratio
        self.volume_trend = Volume.Volume.rolling(window=5).mean() / Volume.Volume.rolling(window=20).mean()
        
        # Alignment: 0 to 3 scale (stationarized)
        self.momentum_alignment = (
            (Macd.macd > Macd.signal_line).astype(int) + 
            (Rsi.rsi > 50).astype(int) + 
            (close_prices > Sma_5.sma).astype(int)
        )
        
        self.dist_lower_bb = (close_prices - Bollinger_Bands.lower_band) / Atr.atr
        self.dist_upper_bb = (Bollinger_Bands.upper_band - close_prices) / Atr.atr
        self.dist_sma = (close_prices - Sma_20.sma) / Atr.atr
        self.price_accel = close_prices.pct_change(3)


    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.bb_width_norm = None
        self.volatility_ratio = None
        self.volatility_change = None
        self.volume_ratio = None
        self.volume_trend = None
        self.momentum_alignment = None
        self.dist_lower_bb = None
        self.dist_upper_bb = None
        self.dist_sma = None
        self.price_accel = None
        self.data = pd.DataFrame()
