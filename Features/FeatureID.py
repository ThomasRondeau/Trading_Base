from enum import Enum, auto

#A terme, lancer cete classe pour générer automatiquement les features
# Ca evitera de faire des erreurs de casting plus tard.

class FeatureID(Enum):
    """
    Énumération de toutes les features possibles basée sur ton FeatureEngineering.
    Cette approche évite les erreurs de frappe et offre une auto-completion parfaite.
    """
    # Indicateurs techniques de base
    SMA_5 = auto()
    SMA_10 = auto()
    SMA_20 = auto()
    SMA_FAST = auto()  # fenêtre paramétrable (TrendFollowing)
    SMA_SLOW = auto()  # fenêtre paramétrable (TrendFollowing)
    RSI = auto()
    MACD = auto()
    
    # Bollinger Bands
    BOLLINGER_BANDS = auto()
    
    # Volume et volatilité
    ATR = auto()
    VOLUME = auto()

    #Features composites
    AMPLITUDE = auto()
    DIRECTION = auto()
    OPPORTUNITY = auto()
    ZSCORE = auto()
    ADX = auto()