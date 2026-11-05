import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ajout du chemin racine pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Connections.DataManager import DataManager
from Strategies.CascadeStrategy import CascadeStrategy
from Engines.BacktestEngine import BacktestEngine

def run_optimization():
    print("Optimisation CascadeStrategy: Seuil d'Amplitude...")
    
    dm = DataManager()
    # Collecte de données plus récente (les 60 derniers jours pour yfinance 2m)
    training_data = dm.collect_data_yfinance("AAPL", 30, 5) # 30 jours de train
    backtest_data = dm.collect_data_yfinance("AAPL", 5, 0)  # 5 jours de backtest
    
    params = np.linspace(0.01, 0.10, 6)
    results = []
    
    for threshold in params:
        print(f"Test avec threshold={threshold:.2f}...")
        strategy = CascadeStrategy(min_amplitude_threshold=threshold)
        engine = BacktestEngine()
        engine.set_training_data(training_data)
        engine.set_backtest_data(backtest_data)
        engine.add_strategy(strategy)
        
        engine.train_strategies()
        metrics = engine.run()
        results.append(metrics['total_pnl_net_percent'])
        
    # Génération du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(params, results, marker='s', linestyle='-', color='g')
    plt.title('Optimisation CascadeStrategy : Seuil d\'Amplitude Minimum')
    plt.xlabel('Seuil d\'Amplitude (%)')
    plt.ylabel('Total PnL Net (%)')
    plt.grid(True)
    
    output_path = os.path.join("report_assets", "opt_cascade.png")
    plt.savefig(output_path)
    print(f"Graphique sauvegardé dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists("report_assets"):
        os.makedirs("report_assets")
    run_optimization()
