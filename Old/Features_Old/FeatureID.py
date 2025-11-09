from enum import Enum, auto

class FeatureID(Enum):
    """
    Énumération de toutes les features possibles basée sur ton FeatureEngineering.
    Cette approche évite les erreurs de frappe et offre une auto-completion parfaite.
    """
    # Indicateurs techniques de base
    SMA_5 = auto()
    SMA_10 = auto() 
    SMA_20 = auto()
    RSI = auto()
    MACD = auto()
    MACD_SIGNAL = auto()
    MACD_HISTOGRAM = auto()
    
    # Bollinger Bands
    BB_MIDDLE = auto()
    BB_UPPER = auto()
    BB_LOWER = auto()
    BB_WIDTH = auto()
    BB_POSITION = auto()
    
    # Volume et volatilité
    ATR = auto()
    VOLUME_MA = auto()
    VOLUME_RATIO = auto()
    
    # Features d'opportunité
    VOLATILITY_COMPRESSION = auto()
    VOLATILITY_EXPANSION = auto()
    VOLATILITY_CHANGE = auto()
    VOLUME_SPIKE = auto()
    VOLUME_TREND = auto()
    MOMENTUM_ALIGNMENT = auto()
    NEAR_SUPPORT = auto()
    NEAR_RESISTANCE = auto()
    CLOSE_TO_SMA = auto()
    PRICE_ACCELERATION = auto()
    
    # Features de direction
    RSI_MOMENTUM = auto()
    MACD_MOMENTUM = auto()
    PRICE_VS_SMA = auto()
    SMA_SLOPE = auto()
    PRICE_POSITION = auto()
    UPPER_BREAKOUT = auto()
    LOWER_BREAKOUT = auto()
    BULLISH_CONFLUENCE = auto()
    BEARISH_CONFLUENCE = auto()
    VOLUME_PRICE_TREND = auto()
    
    # Features d'amplitude
    CURRENT_VOLATILITY = auto()
    VOLATILITY_PERCENTILE = auto()
    DISTANCE_TO_SMA = auto()
    DISTANCE_TO_BB = auto()
    RSI_EXTREME = auto()
    MACD_INTENSITY = auto()
    MOMENTUM_STRENGTH = auto()
    VOLUME_INTENSITY = auto()
