import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from Connections.TradingApp import TradingApp
import yfinance as yf

class DataManager:
    def __init__(self):
        self.cache_dir = Path(__file__).parent
    
    def _get_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_historical_data.csv"

    def _load_cache(self, symbol: str) -> pd.DataFrame:
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            df = pd.read_csv(cache_path, parse_dates=['Datetime'])
            df.set_index('Datetime', inplace=True)
            return df
        return pd.DataFrame()

    def _save_cache(self, symbol: str, data: pd.DataFrame):
        cache_path = self._get_cache_path(symbol)
        data.to_csv(cache_path)  # index=True pour conserver le datetime dans le CSV

    def collect_training_data(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        print(f"Collecte des données d'entraînement pour {symbol} sur {days_back} jours...")
        
        cached_data = self._load_cache(symbol)
        # On dédoublonne par date uniquement (sans heure) pour le cache-hit check
        cached_dates = set(cached_data.index.date.tolist()) if not cached_data.empty else set()

        trading_app = TradingApp.get_instance()
        contract = TradingApp.get_contract(symbol)
        all_data = [cached_data] if not cached_data.empty else []
        new_data_fetched = False

        for day in range(until + 1, days_back + 1):
            try:
                target_date = datetime.now() - timedelta(days=day)
                target_date_only = target_date.date()

                if target_date_only in cached_dates:
                    print(f"Cache hit pour {target_date_only}")
                    continue

                date_str = target_date.strftime("%Y%m%d 20:00:00 US/Eastern")
                daily_data = trading_app.get_historical_data(99, contract, end_date=date_str)

                if not daily_data.empty:
                    all_data.append(daily_data)
                    new_data_fetched = True
                    print(f"Données récupérées pour {target_date_only}: {len(daily_data)} points")
                else:
                    print(f"Pas de données pour {target_date_only}")

            except Exception as e:
                print(f"Erreur lors de la récupération des données pour {target_date.date()}: {e}")
                continue

        if not all_data:
            raise ValueError("Aucune donnée n'a pu être récupérée")

        consolidated_data = pd.concat(all_data)
        consolidated_data = consolidated_data[~consolidated_data.index.duplicated(keep='first')]
        consolidated_data = consolidated_data.sort_index()

        if new_data_fetched:
            self._save_cache(symbol, consolidated_data)
            print(f"Cache mis à jour : {self._get_cache_path(symbol)}")

        # Filtrage sur la plage demandée
        min_date = (datetime.now() - timedelta(days=days_back)).date()
        max_date = (datetime.now() - timedelta(days=until)).date()
        consolidated_data = consolidated_data[
            (consolidated_data.index.date >= min_date) &
            (consolidated_data.index.date <= max_date)
        ]

        print(f"Données consolidées: {len(consolidated_data)} points sur {days_back - until} jours")
        return consolidated_data

    def collect_data_yfinance(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        print(f"Collecte des données via yfinance pour {symbol} sur {days_back} jours...")
        end_date = datetime.now() - timedelta(days=until)
        start_date = end_date - timedelta(days=days_back)
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval='2m')
        
        # Supprimer le niveau "Ticker" du MultiIndex retourné par les nouvelles versions de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.reset_index(inplace=True)
        
        # Renommer les colonnes en minuscules pour correspondre aux attentes des indicateurs (SMA, MACD...)
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        print(f"Données yfinance récupérées: {len(df)} points")
        return df

    def collect_data_massive(self, symbol: str, days_back: int, until: int = 0) -> pd.DataFrame:
        """
        Collecte des données via Massive API.
        Gère les tickers d'indices si nécessaire.
        """
        print(f"Collecte des données via Massive pour {symbol} sur {days_back} jours...")
        
        # Mapping symboles si besoin (ex: SPY -> SPY, ^GSPC -> I:SPX)
        ticker = symbol
        if symbol == "^GSPC" or symbol == "SPX":
            ticker = "I:SPX"
        elif symbol == "^FTSE" or symbol == "FTSE":
            ticker = "I:FTSE"

        df = self.massive_connector.get_intraday_data(ticker, days_back)
        
        if df.empty:
            print(f"Attention: Aucune donnée récupérée via Massive pour {ticker}")
            return pd.DataFrame()

        # Filtrage 'until' (on retire les derniers jours si until > 0)
        if until > 0:
            limit_date = datetime.now() - timedelta(days=until)
            df = df[df.index < limit_date]

        print(f"Données Massive récupérées pour {ticker}: {len(df)} points")
        
        # Mise à jour du cache local
        self._save_cache(symbol, df)
        
        return df
