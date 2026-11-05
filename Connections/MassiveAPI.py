import requests
import pandas as pd
from datetime import datetime, timedelta

class MassiveAPIClient:
    """
    Client pour l'API Massive (https://massive.com/docs/rest/)
    Permet de récupérer des données historiques intraday (OHLCV).
    """
    
    def __init__(self, api_key: str = "cY1KibDdoJtqaZ1nb5ZcrdfUYM0vUgCP"):
        self.api_key = api_key
        self.base_url = "https://api.massive.com"

    def fetch_data(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """
        Récupère l'historique 1-minute pour le symbole donné sur la plage de jours spécifiée.
        
        :param symbol: Symbole (ex: 'AAPL')
        :param days_back: Nombre de jours en arrière pour le début
        :param until: Nombre de jours en arrière pour la fin (0 = aujourd'hui)
        :return: DataFrame avec les colonnes open, high, low, close, volume et index Datetime.
        """
        end_date = datetime.now() - timedelta(days=until)
        start_date = end_date - timedelta(days=days_back)
        
        from_date_str = start_date.strftime("%Y-%m-%d")
        to_date_str = end_date.strftime("%Y-%m-%d")
        
        # Endpoint des agrégats (1 minute)
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{from_date_str}/{to_date_str}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        print(f"Téléchargement via Massive API ({symbol}) : {from_date_str} au {to_date_str}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion à Massive API : {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Détails de l'erreur : {e.response.text}")
            return pd.DataFrame()
            
        data = response.json()
        
        if "results" not in data or not data["results"]:
            print(f"Aucune donnée trouvée pour {symbol} sur cette période.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data["results"])
        
        # Mapping des colonnes Massive API (similaire à Polygon) vers le format local
        column_mapping = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            't': 'timestamp'
        }
        
        # Ne renommer que les colonnes présentes
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)
        
        # Conversion du timestamp (millisecondes) en Datetime
        if 'timestamp' in df.columns:
            df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Datetime', inplace=True)
        
        # Garder uniquement les colonnes requises si elles existent
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in expected_cols if col in df.columns]
        df = df[available_cols]
        
        print(f"Massive API : {len(df)} points (barres 1-minute) récupérés.")
        return df
