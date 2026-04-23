from typing import cast
import numpy as np
import pandas as pd
from Features.FeatureID import FeatureID
from Features import *

class DirectionFeatures(FeatureInterface):
    def __init__(self, data: pd.DataFrame = pd.DataFrame()):
        self.data = data
        self.rsi = None
        self.macd_hist_norm = None
        self.price_dist_sma = None
        self.sma_slope_norm = None
        self.bb_position = None
        self.volume_trend = None
        self.volatility = None


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
            raise ValueError("Erreur dans DirectionFeatures : Les dépendances nécessaires ne sont pas résolues.")

        close_prices = pd.to_numeric(self.data['close'], errors='coerce')

        # 1. RSI (déjà stationnaire entre 0 et 100)
        self.rsi = Rsi.rsi
        
        # 2. MACD Histogramme normalisé par l'ATR pour la stationnarité
        self.macd_hist_norm = Macd.macd_histogram / Atr.atr
        
        # 3. Distance à la SMA 20 normalisée par l'ATR
        self.price_dist_sma = (close_prices - Sma_20.sma) / Atr.atr
        
        # 4. Pente de la SMA lissée et normalisée par l'ATR
        self.sma_slope_norm = Sma_20.sma.diff(3) / Atr.atr
        
        # 5. Position dans les Bandes de Bollinger (0 = lower, 1 = upper)
        self.bb_position = (close_prices - Bollinger_Bands.lower_band) / (Bollinger_Bands.upper_band - Bollinger_Bands.lower_band)
        
        # 6. Trend de volume
        self.volume_trend = Volume.Volume_ratio * np.sign(close_prices.diff())
        
        # 7. Volatilité relative
        self.volatility = Atr.atr / close_prices
        
        # 8. Momentum sur plusieurs horizons
        self.mom_5 = close_prices.pct_change(5)
        self.mom_10 = close_prices.pct_change(10)
        self.mom_20 = close_prices.pct_change(20)


    def update(self, new_data: pd.DataFrame):
        self.data = new_data

    def clear_cache(self):
        self.rsi = None
        self.macd_hist_norm = None
        self.price_dist_sma = None
        self.sma_slope_norm = None
        self.bb_position = None
        self.volume_trend = None
        self.volatility = None
        self.mom_5 = None
        self.mom_10 = None
        self.mom_20 = None
        self.data = pd.DataFrame()
