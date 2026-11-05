import pandas as pd
import numpy as np
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Features.FeatureManager import FeatureManager
from Types.Order import Order
from Features.FeatureID import FeatureID
from Features.Base.ZScore import ZScoreFeature

class PairTradingStrategy(TradingStrategyInterface):
    """
    Stratégie de Pair Trading (AAPL vs MSFT).
    Vend le sur-performant et achète le sous-performant quand le ratio s'écarte.
    """
    def __init__(self, zscore_threshold=2.2) -> None:
        super().__init__()
        self.strategy_id = 3
        self.symbol_a = "AAPL"
        self.symbol_b = "MSFT"
        self.IsRunning = False
        self.featureManager = FeatureManager()
        self.current_data = pd.DataFrame()
        self.zscore_threshold = zscore_threshold # Plus sélectif

    def train(self, training_data: pd.DataFrame) -> None:
        # training_data est supposé contenir 'close_AAPL' et 'close_MSFT'
        self.current_data = training_data.copy()
        if 'ratio' not in self.current_data.columns:
            self.current_data['ratio'] = self.current_data[f'close_{self.symbol_a}'] / self.current_data[f'close_{self.symbol_b}']
        
        self.featureManager.register_indicator(FeatureID.ZSCORE, ZScoreFeature(window=30))
        self.featureManager.update(self.current_data)

    def test(self) -> None:
        print("PairTrading: Smoothed Mean Reversion on Ratio Z-Score.")

    def on_bar(self, bar) -> list[Order]:
        if not self.IsRunning:
            return []

        # Mise à jour synchronisée
        new_row = bar.to_frame().T
        if 'ratio' not in new_row.columns:
            new_row['ratio'] = new_row[f'close_{self.symbol_a}'] / new_row[f'close_{self.symbol_b}']
            
        self.current_data = pd.concat([self.current_data, new_row])
        self.featureManager.update(self.current_data)
        
        zscore_feat = self.featureManager.get_feature(FeatureID.ZSCORE)
        if zscore_feat.zscore is None or len(zscore_feat.zscore) < 3:
            return []
            
        # Lissage du Z-Score sur 3 bars pour éviter les spikes
        curr_z = zscore_feat.zscore.rolling(window=3).mean().iloc[-1]
        
        # Logique de Pair Trading
        if curr_z >= self.zscore_threshold:
            return [
                Order(strategy_ID=self.strategy_id, symbol=self.symbol_a, side='SELL', quantity=10),
                Order(strategy_ID=self.strategy_id, symbol=self.symbol_b, side='BUY', quantity=10)
            ]
        
        if curr_z <= -self.zscore_threshold:
            return [
                Order(strategy_ID=self.strategy_id, symbol=self.symbol_a, side='BUY', quantity=10),
                Order(strategy_ID=self.strategy_id, symbol=self.symbol_b, side='SELL', quantity=10)
            ]
            
        return []

    def stop(self):
        self.IsRunning = False

    def get_status(self) -> bool:
        return self.IsRunning

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running
