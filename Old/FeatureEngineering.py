import pandas as pd
import numpy as np
from Indicators import SMA, RSI, MACD, BollingerBands, ATR

# Machine Learning imports

from sklearn.preprocessing import StandardScaler



class FeatureEngineering:
    """
    Classe responsable de la création des features pour nos modèles ML.
    
    Cette classe transforme les données brutes de marché en features utiles
    pour nos trois modèles de la cascade : détection d'opportunité, 
    prédiction de direction, et estimation d'amplitude.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques de base nécessaires pour nos features.
        
        Args:
            data: DataFrame avec colonnes OHLCV
            
        Returns:
            DataFrame avec les indicateurs techniques ajoutés
        """
        df = data.copy()
        
        # Moyennes mobiles sur différentes périodes
        df['sma_5'] = SMA(df, period=5).sma
        df['sma_10'] = SMA(df, period=10).sma
        df['sma_20'] = SMA(df, period=20).sma
        
        # RSI pour mesurer le momentum
        df['rsi'] = RSI(df, period=14).rsi
        
        # MACD pour les signaux de momentum
        macd = MACD(df, short_window=12, long_window=26, signal_window=9)
        df['macd'] = macd.macd
        df['macd_signal'] = macd.signal_line
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands pour la volatilité
        bb = BollingerBands(df, window=20, num_std_dev=2)
        df['bb_middle'] = bb.sma
        df['bb_upper'] = bb.upper_band
        df['bb_lower'] = bb.lower_band
        df['bb_width'] = bb.width
        df['bb_position'] = bb.position
        
        # ATR pour la volatilité
        df['atr'] = ATR(df, period=14).atr
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def create_opportunity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features spécifiques pour le détecteur d'opportunité.
        
        Le détecteur d'opportunité cherche des signaux qui indiquent qu'un
        mouvement significatif est sur le point de se produire. Il se concentre
        sur la volatilité, le volume et les patterns de compression/expansion.
        
        Args:
            data: DataFrame avec les indicateurs techniques
            
        Returns:
            DataFrame avec les features d'opportunité
        """
        df = data.copy()
        
        # Features de volatilité - cherche les compressions suivies d'expansions
        df['volatility_compression'] = (df['bb_width'] < df['bb_width'].rolling(10).mean() * 0.8).astype(int)
        df['volatility_expansion'] = (df['atr'] > df['atr'].shift(1) * 1.2).astype(int)
        df['volatility_change'] = df['atr'].pct_change()
        
        # Features de volume - volume anormal signale souvent un mouvement
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Features de momentum - confluence sur plusieurs timeframes
        df['momentum_alignment'] = ((df['macd'] > df['macd_signal']) & 
                                   (df['rsi'] > 50) & 
                                   (df['close'] > df['sma_5'])).astype(int)
        
        # Features de position dans la range
        df['near_support'] = (df['close'] < df['bb_lower'] * 1.02).astype(int)
        df['near_resistance'] = (df['close'] > df['bb_upper'] * 0.98).astype(int)
        
        # Features de breakout potentiel
        df['close_to_sma'] = abs(df['close'] - df['sma_20']) / df['atr']
        df['price_acceleration'] = df['close'].pct_change(2)
        
        return df
    
    def create_direction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features spécifiques pour la prédiction de direction.
        
        Le prédicteur de direction cherche des signaux qui indiquent
        dans quelle direction le prix va bouger. Il se concentre sur
        le momentum, les niveaux techniques et les patterns directionnels.
        
        Args:
            data: DataFrame avec les indicateurs techniques
            
        Returns:
            DataFrame avec les features directionnelles
        """
        df = data.copy()
        
        # Features de momentum directionnel
        df['rsi_momentum'] = (df['rsi'] > 50).astype(int)
        df['macd_momentum'] = (df['macd'] > df['macd_signal']).astype(int)
        df['price_vs_sma'] = (df['close'] > df['sma_20']).astype(int)
        
        # Features de position relative
        df['sma_slope'] = df['sma_20'].diff(5)
        df['price_position'] = (df['close'] - df['sma_20']) / df['atr']
        
        # Features de breakout directionnel
        df['upper_breakout'] = (df['close'] > df['bb_upper']).astype(int)
        df['lower_breakout'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Features de confluence directionnelle
        df['bullish_confluence'] = (df['rsi_momentum'] & df['macd_momentum'] & df['price_vs_sma']).astype(int)
        df['bearish_confluence'] = ((1 - df['rsi_momentum']) & (1 - df['macd_momentum']) & (1 - df['price_vs_sma'])).astype(int)
        
        # Features de volume directionnel
        df['volume_price_trend'] = df['volume_ratio'] * np.sign(df['close'].pct_change())
        
        return df
    
    def create_amplitude_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features pour l'estimation d'amplitude.
        
        L'estimateur d'amplitude cherche des signaux qui indiquent
        quelle sera la force du mouvement. Il se concentre sur la volatilité,
        la distance aux niveaux techniques et l'intensité des signaux.
        
        Args:
            data: DataFrame avec les indicateurs techniques
            
        Returns:
            DataFrame avec les features d'amplitude
        """
        df = data.copy()
        
        # Features de volatilité pour l'amplitude
        df['current_volatility'] = df['atr'] / df['close']
        df['volatility_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        # Features de distance aux niveaux techniques
        df['distance_to_sma'] = abs(df['close'] - df['sma_20']) / df['close']
        df['distance_to_bb'] = np.minimum(abs(df['close'] - df['bb_upper']), 
                                         abs(df['close'] - df['bb_lower'])) / df['close']
        
        # Features d'intensité des signaux
        df['rsi_extreme'] = np.maximum(df['rsi'] - 70, 30 - df['rsi']).clip(0)
        df['macd_intensity'] = abs(df['macd_histogram']) / df['atr']
        
        # Features de momentum pour l'amplitude
        df['momentum_strength'] = abs(df['close'].pct_change(5))
        df['volume_intensity'] = df['volume_ratio'] * df['current_volatility']
        
        return df
    
    def create_labels(self, data: pd.DataFrame, look_ahead_periods: int = 12) -> pd.DataFrame:
        """
        Crée les labels pour l'entraînement des modèles.
        
        Cette fonction est cruciale car elle définit ce que nos modèles
        vont apprendre à prédire. Pour du trading intraday avec des positions
        d'une minute, nous regardons les 12 prochaines périodes (12 minutes).
        
        Args:
            data: DataFrame avec les données de marché
            look_ahead_periods: Nombre de périodes à regarder vers l'avant
            
        Returns:
            DataFrame avec les labels ajoutés
        """
        df = data.copy()
        
        # Calcul des returns futurs sur différents horizons
        future_returns = []
        for i in range(1, look_ahead_periods + 1):
            future_return = (df['close'].shift(-i) - df['close']) / df['close']
            future_returns.append(future_return)
        
        # Maximum return dans la période (pour l'amplitude)
        df['max_future_return'] = pd.concat(future_returns, axis=1).max(axis=1)
        df['min_future_return'] = pd.concat(future_returns, axis=1).min(axis=1)
        
        # Calcul des labels pour chaque modèle
        
        # 1. Label d'opportunité : est-ce qu'il y a un mouvement significatif ?
        # Un mouvement est significatif s'il dépasse 0.5% dans une direction
        df['opportunity_label'] = (
            (df['max_future_return'] > 0.005) | 
            (df['min_future_return'] < -0.005)
        ).astype(int)
        
        # 2. Label de direction : dans quelle direction va le prix ?
        # Seulement calculé quand il y a une opportunité
        df['direction_label'] = np.where(
            df['opportunity_label'] == 1,
            (df['max_future_return'] > abs(df['min_future_return'])).astype(int),
            np.nan
        )
        
        # 3. Label d'amplitude : quelle sera l'amplitude du mouvement ?
        # Amplitude maximale absolue du mouvement
        df['amplitude_label'] = np.maximum(
            df['max_future_return'],
            abs(df['min_future_return'])
        )
        
        return df
    
    def prepare_features_for_model(self, data: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Prépare les features finales pour un modèle spécifique.
        
        Cette fonction sélectionne les features appropriées pour chaque type
        de modèle et gère la normalisation si nécessaire.
        
        Args:
            data: DataFrame avec toutes les features
            model_type: 'opportunity', 'direction', ou 'amplitude'
            
        Returns:
            DataFrame avec les features sélectionnées et normalisées
        """
        if model_type == 'opportunity':
            feature_columns = [
                'volatility_compression', 'volatility_expansion', 'volatility_change',
                'volume_spike', 'volume_trend', 'momentum_alignment',
                'near_support', 'near_resistance', 'close_to_sma', 'price_acceleration',
                'rsi', 'bb_width', 'atr', 'volume_ratio'
            ]
        elif model_type == 'direction':
            feature_columns = [
                'rsi_momentum', 'macd_momentum', 'price_vs_sma',
                'sma_slope', 'price_position', 'upper_breakout', 'lower_breakout',
                'bullish_confluence', 'bearish_confluence', 'volume_price_trend',
                'rsi', 'macd_histogram', 'bb_position'
            ]
        elif model_type == 'amplitude':
            feature_columns = [
                'current_volatility', 'volatility_percentile',
                'distance_to_sma', 'distance_to_bb', 'rsi_extreme',
                'macd_intensity', 'momentum_strength', 'volume_intensity',
                'atr', 'volume_ratio', 'bb_width'
            ]
        else:
            raise ValueError(f"Type de modèle non reconnu: {model_type}")
        
        # Sélection des features qui existent dans le DataFrame
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            raise ValueError(f"Aucune feature disponible pour le modèle {model_type}")
        
        features_df = data[available_features].copy()

        # Gestion des valeurs manquantes
        features_df = features_df.fillna(method='ffill').fillna(0)

        return features_df