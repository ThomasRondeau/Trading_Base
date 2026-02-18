import pandas as pd
import datetime
from Connections.TradingApp import TradingApp


class DataManager:
    def __init__(self):
        pass

    def collect_training_data(self, symbol: str, days_back: int, until : int) -> pd.DataFrame:

        print(f"Collecte des données d'entraînement pour {symbol} sur {days_back} jours...")
        trading_app = TradingApp.get_instance()
        all_data = []
        contract = TradingApp.get_contract(symbol)
        
        # Récupération des données jour par jour
        for day in range(until + 1, days_back + 1):
            try:
                # Calcul de la date pour ce jour
                target_date = datetime.now() - datetime.timedelta(days=day)
                date_str = target_date.strftime("%Y%m%d 20:00:00 US/Eastern")
                
                # Récupération des données pour ce jour
                daily_data = trading_app.get_historical_data(99, contract, end_date=date_str)
                
                if not daily_data.empty:
                    daily_data['date'] = target_date.date()
                    all_data.append(daily_data)
                    print(f"Données récupérées pour {target_date.date()}: {len(daily_data)} points")
                else:
                    print(f"Pas de données pour {target_date.date()}")
                    
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour {target_date.date()}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Aucune donnée n'a pu être récupérée")
        
        # Consolidation des données
        consolidated_data = pd.concat(all_data, ignore_index=True)
        consolidated_data = consolidated_data.sort_values(by='date').reset_index(drop=True)
        
        print(f"Données consolidées: {len(consolidated_data)} points sur {len(all_data)} jours")
        
        return consolidated_data