from enum import Enum, auto


class FeatureID(Enum):
    """Keys under which features are registered in a `FeatureManager`."""

    # Base technical indicators
    SMA_5 = auto()
    SMA_10 = auto()
    SMA_20 = auto()
    SMA_FAST = auto()  # configurable window (trend following)
    SMA_SLOW = auto()  # configurable window (trend following)
    RSI = auto()
    MACD = auto()
    BOLLINGER_BANDS = auto()
    ATR = auto()
    VOLUME = auto()
    ZSCORE = auto()
    ADX = auto()

    # Composite feature sets (one per model)
    AMPLITUDE = auto()
    DIRECTION = auto()
    OPPORTUNITY = auto()
