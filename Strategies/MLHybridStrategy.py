import pandas as pd
from Strategies.TradingStrategyInterface import TradingStrategyInterface
from Features.FeatureManager import FeatureManager
from Targets.TargetManager import TargetManager
from Targets.TargetID import TargetID
from Models.DirectionModel import DirectionModel
from Types.Order import Order
from Features.FeatureID import FeatureID
from Features.Base.BollingerBands import BollingerBands
from Features.Base.RSI import RSI
from Features.Base.VOLUME import VOLUME
from Features.Base.ATR import ATR
from Features.Base.MACD import MACD
from Features.Base.SMA import SMA
from Features.Models.DirectionFeatures import DirectionFeatures
from Targets.Models.Target_Direction import Target_Direction

class MLHybridStrategy(TradingStrategyInterface):
    """
    Stratégie Hybride : Mean Reversion + DirectionModel XGBoost.
    Ne prend un trade de retournement que si le modèle ML confirme la direction.
    """
    def __init__(self, max_depth=4) -> None:
        super().__init__()
        self.strategy_id = 6
        self.symbol = "AAPL"
        self.IsRunning = False
        self.featureManager = FeatureManager()
        self.targetManager = TargetManager()
        self.directionModel = DirectionModel(max_depth=max_depth)
        self.current_data = pd.DataFrame()
        self.training_data_shift = 30

    def train(self, training_data: pd.DataFrame) -> None:
        # 1. Enregistrement des indicateurs techniques
        self.featureManager.register_indicator(FeatureID.BOLLINGER_BANDS, BollingerBands(window=20, num_std_dev=2))
        self.featureManager.register_indicator(FeatureID.RSI, RSI(window=14))
        self.featureManager.register_indicator(FeatureID.VOLUME, VOLUME(window=20))
        self.featureManager.register_indicator(FeatureID.ATR, ATR(window=14))
        self.featureManager.register_indicator(FeatureID.MACD, MACD(long_window=12, short_window=26, signal_window=9))
        self.featureManager.register_indicator(FeatureID.SMA_20, SMA(window=20))
        self.featureManager.register_indicator(FeatureID.DIRECTION, DirectionFeatures())
        
        # 2. Enregistrement des cibles ML
        self.targetManager.register_target(TargetID.DIRECTION, Target_Direction())
        
        # 3. Préparation des données
        self.current_data = training_data.copy().iloc[:-self.training_data_shift]
        self.featureManager.update(self.current_data)
        self.targetManager.update(training_data, training_data_shift=self.training_data_shift)
        
        # 4. Entraînement du modèle
        self.directionModel.setFeatureManager(self.featureManager)
        self.directionModel.setTargetManager(self.targetManager)
        self.directionModel.train()

    def test(self) -> None:
        acc = self.directionModel.test()
        print(f"MLHybrid: DirectionModel Train Acc: {acc:.2f}")

    def on_bar(self, bar) -> list[Order]:
        if not self.IsRunning:
            return []

        self.current_data = pd.concat([self.current_data, bar.to_frame().T]).iloc[-200:]
        self.featureManager.update(self.current_data)
        
        # Signaux techniques
        bb = self.featureManager.get_feature(FeatureID.BOLLINGER_BANDS)
        rsi = self.featureManager.get_feature(FeatureID.RSI).rsi
        vol = self.featureManager.get_feature(FeatureID.VOLUME).Volume_ratio
        
        curr_price = bar['close']
        curr_rsi = rsi.iloc[-1]
        curr_vol = vol.iloc[-1]
        upper = bb.upper_band.iloc[-1]
        lower = bb.lower_band.iloc[-1]
        
        # Filtre technique
        tech_buy = curr_price <= lower and curr_rsi < 35 and curr_vol > 1.1
        tech_sell = curr_price >= upper and curr_rsi > 65 and curr_vol > 1.1
        
        if not (tech_buy or tech_sell):
            return []
            
        # Filtre ML : prédiction de la direction à 5 bars
        ml_direction = self.directionModel.predict() # 'UP' ou 'DOWN'
        
        if tech_buy and ml_direction == 'UP':
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='BUY', quantity=10)]
        
        if tech_sell and ml_direction == 'DOWN':
            return [Order(strategy_ID=self.strategy_id, symbol=self.symbol, side='SELL', quantity=10)]
            
        return []

    def stop(self):
        self.IsRunning = False

    def get_status(self) -> bool:
        return self.IsRunning

    def update_status(self, is_running: bool) -> None:
        self.IsRunning = is_running
