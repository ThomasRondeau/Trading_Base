import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from Connections.schema import validate_ohlcv

try:
    from dotenv import load_dotenv

    load_dotenv()  # Charge un éventuel fichier .env à la racine du projet
except ImportError:  # python-dotenv est optionnel
    pass

API_KEY_ENV_VAR = "MASSIVE_API_KEY"


class MassiveAPIClient:
    """
    Client pour l'API Massive (https://massive.com/docs/rest/)
    Permet de récupérer des données historiques intraday (OHLCV).

    La clé API est lue dans la variable d'environnement MASSIVE_API_KEY
    (ou dans un fichier .env), sauf si elle est passée explicitement.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        resolved_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved_key:
            raise RuntimeError(
                f"Clé Massive API introuvable : définissez la variable d'environnement "
                f"{API_KEY_ENV_VAR} (ou ajoutez-la dans un fichier .env, voir .env.example)."
            )
        self.api_key: str = resolved_key
        self.base_url = "https://api.massive.com"

    def fetch_data(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """
        Récupère l'historique 1-minute pour le symbole donné sur la plage de jours spécifiée.
        
        :param symbol: Symbole (ex: 'AAPL')
        :param days_back: Nombre de jours en arrière pour le début
        :param until: Nombre de jours en arrière pour la fin (0 = aujourd'hui)
        :return: DataFrame OHLCV au format validate_ohlcv (index Datetime UTC).
        """
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() - timedelta(days=until)
        
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
        
        # Timestamps Massive = millisecondes epoch UTC
        if 'timestamp' not in df.columns:
            raise ValueError(f"Réponse Massive inattendue : colonnes {list(df.columns)}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = validate_ohlcv(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], datetime_column='timestamp')

        print(f"Massive API : {len(df)} points (barres 1-minute) récupérés.")
        return df
