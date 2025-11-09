from dataclasses import dataclass
from typing import Dict, Set, Callable, Any, List, Optional
import pandas as pd
from FeatureID import FeatureID 

@dataclass
class FeatureConfig:
    """
    Configuration d'une feature avec sa fonction de calcul et ses dépendances.
    Cette structure permet de définir clairement les relations entre features.
    """
    calculator: Callable[[pd.DataFrame, Dict[FeatureID, Any]], Any]
    dependencies: Set[FeatureID]
    description: str = ""



class FeatureRegistry:
    """
    Registre central qui maintient la correspondance entre les IDs de features
    et leurs fonctions de calcul. Cette approche centralise la logique métier
    des features tout en gardant une structure claire et maintenable.
    """
    
    def __init__(self):
        self._features: Dict[FeatureID, FeatureConfig] = {}
        self._setup_base_features()
    
    def _setup_base_features(self):
        """
        Configure toutes les features de base en s'inspirant de ton FeatureEngineering.
        Cette méthode traduit ta logique existante dans le nouveau système typé.
        """
        
        # Indicateurs techniques de base - ces features n'ont pas de dépendances
        # car elles sont calculées directement à partir des données OHLCV
        self.register(FeatureID.SMA_5, FeatureConfig(
            calculator=lambda data, deps: data['close'].rolling(window=5).mean().iloc[-1],
            dependencies=set(),
            description="SMA 5 périodes"
        ))
        
        self.register(FeatureID.SMA_10, FeatureConfig(
            calculator=lambda data, deps: data['close'].rolling(window=10).mean().iloc[-1],
            dependencies=set(),
            description="SMA 10 périodes"
        ))
        
        self.register(FeatureID.SMA_20, FeatureConfig(
            calculator=lambda data, deps: data['close'].rolling(window=20).mean().iloc[-1],
            dependencies=set(),
            description="SMA 20 périodes"
        ))
        
        self.register(FeatureID.RSI, FeatureConfig(
            calculator=lambda data, deps: self._calculate_rsi(data, 14),
            dependencies=set(),
            description="RSI 14 périodes"
        ))
        
        # MACD et ses composants
        self.register(FeatureID.MACD, FeatureConfig(
            calculator=lambda data, deps: (data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()).iloc[-1],
            dependencies=set(),
            description="Ligne MACD"
        ))
        
        self.register(FeatureID.MACD_SIGNAL, FeatureConfig(
            calculator=lambda data, deps: pd.Series([deps[FeatureID.MACD]]).ewm(span=9).mean().iloc[0],
            dependencies={FeatureID.MACD},
            description="Signal MACD"
        ))
        
        self.register(FeatureID.MACD_HISTOGRAM, FeatureConfig(
            calculator=lambda data, deps: deps[FeatureID.MACD] - deps[FeatureID.MACD_SIGNAL],
            dependencies={FeatureID.MACD, FeatureID.MACD_SIGNAL},
            description="Histogramme MACD"
        ))
        
        # ATR pour la volatilité
        self.register(FeatureID.ATR, FeatureConfig(
            calculator=lambda data, deps: self._calculate_atr(data, 14),
            dependencies=set(),
            description="Average True Range 14 périodes"
        ))
        
        # Volume features
        self.register(FeatureID.VOLUME_MA, FeatureConfig(
            calculator=lambda data, deps: data['volume'].rolling(window=20).mean().iloc[-1],
            dependencies=set(),
            description="Volume MA 20 périodes"
        ))
        
        self.register(FeatureID.VOLUME_RATIO, FeatureConfig(
            calculator=lambda data, deps: data['volume'].iloc[-1] / deps[FeatureID.VOLUME_MA] if deps[FeatureID.VOLUME_MA] > 0 else 1.0,
            dependencies={FeatureID.VOLUME_MA},
            description="Ratio volume courant vs MA 20"
        ))
        
        # Bollinger Bands - ces features montrent comment gérer des dépendances multiples
        self.register(FeatureID.BB_MIDDLE, FeatureConfig(
            calculator=lambda data, deps: deps[FeatureID.SMA_20],  # BB middle = SMA20
            dependencies={FeatureID.SMA_20},
            description="Bollinger Band milieu"
        ))
        
        self.register(FeatureID.BB_WIDTH, FeatureConfig(
            calculator=lambda data, deps: data['close'].rolling(20).std().iloc[-1] * 2,
            dependencies=set(),
            description="Largeur Bollinger Bands"
        ))
        
        self.register(FeatureID.BB_UPPER, FeatureConfig(
            calculator=lambda data, deps: deps[FeatureID.BB_MIDDLE] + deps[FeatureID.BB_WIDTH],
            dependencies={FeatureID.BB_MIDDLE, FeatureID.BB_WIDTH},
            description="Bollinger Band supérieure"
        ))
        
        self.register(FeatureID.BB_LOWER, FeatureConfig(
            calculator=lambda data, deps: deps[FeatureID.BB_MIDDLE] - deps[FeatureID.BB_WIDTH],
            dependencies={FeatureID.BB_MIDDLE, FeatureID.BB_WIDTH},
            description="Bollinger Band inférieure"
        ))
        
        self.register(FeatureID.BB_POSITION, FeatureConfig(
            calculator=lambda data, deps: (data['close'].iloc[-1] - deps[FeatureID.BB_LOWER]) / (deps[FeatureID.BB_UPPER] - deps[FeatureID.BB_LOWER]) if deps[FeatureID.BB_UPPER] != deps[FeatureID.BB_LOWER] else 0.5,
            dependencies={FeatureID.BB_UPPER, FeatureID.BB_LOWER},
            description="Position dans les Bollinger Bands"
        ))
        
        # Features d'opportunité - traduites de ta méthode create_opportunity_features
        self.register(FeatureID.VOLATILITY_COMPRESSION, FeatureConfig(
            calculator=lambda data, deps: int(deps[FeatureID.BB_WIDTH] < data['close'].rolling(20).std().rolling(10).mean().iloc[-1] * 0.8 * 2),
            dependencies={FeatureID.BB_WIDTH},
            description="Compression de volatilité détectée"
        ))
        
        self.register(FeatureID.VOLUME_SPIKE, FeatureConfig(
            calculator=lambda data, deps: int(deps[FeatureID.VOLUME_RATIO] > 1.5),
            dependencies={FeatureID.VOLUME_RATIO},
            description="Pic de volume détecté"
        ))
        
        self.register(FeatureID.MOMENTUM_ALIGNMENT, FeatureConfig(
            calculator=lambda data, deps: int((deps[FeatureID.MACD] > deps[FeatureID.MACD_SIGNAL]) and (deps[FeatureID.RSI] > 50) and (data['close'].iloc[-1] > deps[FeatureID.SMA_5])),
            dependencies={FeatureID.MACD, FeatureID.MACD_SIGNAL, FeatureID.RSI, FeatureID.SMA_5},
            description="Alignement de momentum multi-timeframe"
        ))
        
        self.register(FeatureID.CLOSE_TO_SMA, FeatureConfig(
            calculator=lambda data, deps: abs(data['close'].iloc[-1] - deps[FeatureID.SMA_20]) / deps[FeatureID.ATR] if deps[FeatureID.ATR] > 0 else 0,
            dependencies={FeatureID.SMA_20, FeatureID.ATR},
            description="Distance prix/SMA20 en multiples d'ATR"
        ))
        
        # Features de direction - traduites de create_direction_features
        self.register(FeatureID.RSI_MOMENTUM, FeatureConfig(
            calculator=lambda data, deps: int(deps[FeatureID.RSI] > 50),
            dependencies={FeatureID.RSI},
            description="Momentum RSI haussier"
        ))
        
        self.register(FeatureID.MACD_MOMENTUM, FeatureConfig(
            calculator=lambda data, deps: int(deps[FeatureID.MACD] > deps[FeatureID.MACD_SIGNAL]),
            dependencies={FeatureID.MACD, FeatureID.MACD_SIGNAL},
            description="Momentum MACD haussier"
        ))
        
        self.register(FeatureID.PRICE_VS_SMA, FeatureConfig(
            calculator=lambda data, deps: int(data['close'].iloc[-1] > deps[FeatureID.SMA_20]),
            dependencies={FeatureID.SMA_20},
            description="Prix au-dessus SMA20"
        ))
        
        self.register(FeatureID.BULLISH_CONFLUENCE, FeatureConfig(
            calculator=lambda data, deps: int(deps[FeatureID.RSI_MOMENTUM] and deps[FeatureID.MACD_MOMENTUM] and deps[FeatureID.PRICE_VS_SMA]),
            dependencies={FeatureID.RSI_MOMENTUM, FeatureID.MACD_MOMENTUM, FeatureID.PRICE_VS_SMA},
            description="Confluence haussière multi-indicateurs"
        ))
        
        # Features d'amplitude - traduites de create_amplitude_features
        self.register(FeatureID.CURRENT_VOLATILITY, FeatureConfig(
            calculator=lambda data, deps: deps[FeatureID.ATR] / data['close'].iloc[-1],
            dependencies={FeatureID.ATR},
            description="Volatilité courante normalisée"
        ))
        
        self.register(FeatureID.DISTANCE_TO_SMA, FeatureConfig(
            calculator=lambda data, deps: abs(data['close'].iloc[-1] - deps[FeatureID.SMA_20]) / data['close'].iloc[-1],
            dependencies={FeatureID.SMA_20},
            description="Distance au SMA20 en pourcentage"
        ))
        
        self.register(FeatureID.RSI_EXTREME, FeatureConfig(
            calculator=lambda data, deps: max(deps[FeatureID.RSI] - 70, 30 - deps[FeatureID.RSI], 0),
            dependencies={FeatureID.RSI},
            description="Niveau extrême du RSI"
        ))
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> float:
        """Calcule le RSI - réimplémentation de ta logique Id.RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """Calcule l'ATR - réimplémentation de ta logique Id.ATR"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    
    def register(self, feature_id: FeatureID, config: FeatureConfig):
        """Enregistre une nouvelle feature dans le registre."""
        self._features[feature_id] = config
    
    def get_config(self, feature_id: FeatureID) -> FeatureConfig:
        """Récupère la configuration d'une feature."""
        if feature_id not in self._features:
            raise ValueError(f"Feature {feature_id} not registered") #TODO Potententiel problème si on oublie d'enregistrer une feature et qu'on essaie de l'utiliser
        return self._features[feature_id]
    
    def get_all_dependencies(self, feature_ids: Set[FeatureID]) -> Set[FeatureID]:
        """
        Résout récursivement toutes les dépendances d'un ensemble de features.
        Cette fonction est cruciale pour s'assurer qu'on calcule tout ce qui est nécessaire
        dans le bon ordre, même pour des dépendances complexes imbriquées.
        """
        all_deps = set()
        to_process = feature_ids.copy()
        
        while to_process:
            current = to_process.pop()
            if current in all_deps:
                continue
                
            all_deps.add(current)
            config = self.get_config(current)
            
            for dep in config.dependencies:
                if dep not in all_deps:
                    to_process.add(dep)
        
        return all_deps