import time
import threading
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import mplfinance as mpf
from Connections.TradingApp import TradingApp
from Connections.schema import validate_ohlcv
from Features import *


def main():
    app = TradingApp.get_instance()
    
    print("Tentative de connexion à TWS/IB Gateway...")
    print("Assurez-vous que TWS ou IB Gateway est lancé et configuré pour accepter les connexions API")
    
    
    # Essayer de se connecter
    app.connect("127.0.0.1", 7497, clientId=5)
    
    # Lancer le thread de l'API
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    # Attendre la connexion avec timeout
    timeout = 10  # 10 secondes
    start_time = time.time()
    
    while not app.connected and (time.time() - start_time) < timeout:
        print("En attente de connexion...")
        time.sleep(1)
    
    if not app.connected:
        print("ERREUR: Impossible de se connecter à TWS/IB Gateway")
        print("Vérifiez que :")
        print("1. TWS ou IB Gateway est lancé")
        print("2. L'API est activée dans TWS (Configuration > API > Enable ActiveX and Socket Clients)")
        print("3. Le port 7497 est correct (7497 pour TWS, 4002 pour IB Gateway)")
        print("4. L'ID client 5 n'est pas déjà utilisé")
        return
    
    print("Connexion réussie !")
    
    spy = TradingApp.get_contract("SPY")
    data = app.get_historical_data(99, spy, end_date="20250711 20:00:00 US/Eastern")
    data = validate_ohlcv(data)
    Indicator_Vwap = Vwap(data)
    Indicator_Support = Support(data)
    Indicator_Resistance = Resistance(data)
    if not data.empty:
        for indicator in (Indicator_Vwap, Indicator_Support, Indicator_Resistance):
            indicator.calculate()

    try:

        if not data.empty:
            print(f"Données SPY obtenues: {len(data)} points")
            
            data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            additional_plots = []
            
            # VWAP et bandes
            additional_plots.append(mpf.make_addplot(Indicator_Vwap.point_of_control, color='orange', width=2, label='VWAP'))
            additional_plots.append(mpf.make_addplot(Indicator_Vwap.superior_limit, color='green', linestyle='--', label='Upper Band'))
            additional_plots.append(mpf.make_addplot(Indicator_Vwap.far_superior_limit, color='green', linestyle=':', label='Far Upper Band'))
            additional_plots.append(mpf.make_addplot(Indicator_Vwap.inferior_limit, color='green', linestyle='--', label='Lower Band'))
            additional_plots.append(mpf.make_addplot(Indicator_Vwap.far_inferior_limit, color='green', linestyle=':', label='Far Lower Band'))

            
            # Ajout des lignes de résistance
            for index, resistance in Indicator_Resistance.resistances.head(3).iterrows():
                
                # Créer une série avec NaN partout sauf aux points de la ligne
                resistance_series = pd.Series(index=data.index, data=np.nan)
                start_idx = resistance['idxStart']
                end_idx = resistance['idxEnd']
                resistance_points  = resistance['points']
                
                # Remplir la série avec le niveau de résistance entre start_idx et end_idx
                for idx in range(start_idx, min(end_idx + 1, len(data))):
                    if idx in resistance_points.keys():
                        resistance_series.iloc[idx] = resistance_points[idx]
                
                additional_plots.append(mpf.make_addplot(resistance_series, 
                                                    color='red', 
                                                    width=1.5, 
                                                    linestyle='-',
                                                    label=f'Resistance {index+1}'))
            

            # Ajout des lignes de support
            for index, support in Indicator_Support.supports.head(3).iterrows():
                
                # Créer une série avec NaN partout sauf aux points de la ligne
                support_series = pd.Series(index=data.index, data=np.nan)
                start_idx = support['idxStart']
                end_idx = support['idxEnd']
                support_points  = support['points']
                
                # Remplir la série avec le niveau de support entre start_idx et end_idx
                for idx in range(start_idx, min(end_idx + 1, len(data))):
                    if idx in support_points.keys():
                        support_series.iloc[idx] = support_points[idx]
                
                additional_plots.append(mpf.make_addplot(support_series, 
                                                    color='blue', 
                                                    width=1.5, 
                                                    linestyle='-',
                                                    label=f'Support {index+1}'))
            
            
            # Création du graphique en bougies
            fig, axes = mpf.plot(data,
                                type='candle',           # Type de graphique (bougies)
                                style='yahoo',           # Style du graphique
                                figsize=(12, 8),         # Taille de la figure
                                title='SPY',
                                ylabel='Prix ($)',
                                addplot=additional_plots, # Ajout des indicateurs
                                volume=True,             # Affiche le volume (si disponible)
                                show_nontrading=False,   # Masque les périodes sans trading
                                tight_layout=True,
                                returnfig=True)
            
            # Ajout d'une légende personnalisée
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc='upper left', fontsize=8)
            
            # Affichage du graphique
            plt.show()

        else:
            print("Aucune donnée obtenue pour SPY")

    except Exception as e:
        print(f"Erreur dans Main: {e}")

    finally:
        if app.connected:
            app.disconnect()
            print("Déconnexion")
    

if __name__ == "__main__":
    main()