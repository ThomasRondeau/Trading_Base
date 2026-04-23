import pandas as pd
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Features.FeatureManager import FeatureManager
from Types.Order import Order
from Features.FeatureID import FeatureID
from Features.Base.BollingerBands import BollingerBands
from Features.Base.RSI import RSI
from Features.Base.VOLUME import VOLUME
from Features.Base.ATR import ATR

class MeanReversionStrategy(TradingStrategyInterface):
    """
    Stratégie de Mean Reversion basée sur les Bandes de Bollinger et le RSI.
    """
    def __init__(self) -> None:
        super().__init__()
        self.strategy_id = 5
        self.symbol = "AAPL"
        self.IsRunning = False
        self.featureManager = FeatureManager()
        self.current_data = pd.DataFrame()

    def train(self, training_data: pd.DataFrame) -> None:
        self.featureManager.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        self.featureManager.register_indicator(FeatureID.RSI, RSI(window=14))
        self.featureManager.register_indicator(FeatureID.VOLUME, VOLUME(window=20))
        self.featureManager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.current_data = training_data.copy()
        self.featureManager.update(self.current_data)

    def test(self) -> None:
        print("MeanReversion: BB + RSI + Volume Spike + ATR Filter.")

    def on_bar(self, bar) -> list[Order]:
        if not self.IsRunning:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T])
        self.featureManager.update(self.current_data)
        
        bb = self.featureManager.get_feature(FeatureID.BOLLINGER_BANDS)
        rsi = self.featureManager.get_feature(FeatureID.RSI).rsi
        vol = self.featureManager.get_feature(FeatureID.VOLUME).Volume_ratio
        atr = self.featureManager.get_feature(FeatureID.ATR).atr
        
        if len(rsi) < 1 or bb.upper_band is None or len(vol) < 1:
            return []
            
        curr_price = bar['close']
        curr_rsi = rsi.iloc[-1]
        curr_vol = vol.iloc[-1]
        curr_atr = atr.iloc[-1]
        upper = bb.upper_band.iloc[-1]
        lower = bb.lower_band.iloc[-1]
        
        # Filtre de volume : pic de volume pour confirmer le retournement potentiel
        vol_spike = curr_vol > 1.2
        
        # Filtre d'amplitude : le prix doit être significativement hors des bandes
        oversold = curr_price < (lower - 0.2 * curr_atr)
        overbought = curr_price > (upper + 0.2 * curr_atr)
        
        if oversold and curr_rsi < 30 and vol_spike:
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='BUY', quantity=10)]
        
        if overbought and curr_rsi > 70 and vol_spike:
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='SELL', quantity=10)]
            
        return []

    def stop(self):
        self.IsRunning = False

    def get_status(self) -> bool:
        return self.IsRunning

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running
