import traceback
import pandas as pd
from Connections.MassiveAPI import MassiveAPIClient
from Strategies.CascadeStrategy import CascadeStrategy
from Strategies.ArimaStrategy import ArimaStrategy
from Strategies.MeanReversionStrategy import MeanReversionStrategy
from Strategies.MLHybridStrategy import MLHybridStrategy
from Strategies.TrendFollowingStrategy import TrendFollowingStrategy
from Engines.BacktestEngine import BacktestEngine

def run_backtest_for_strategy(strategy, training_data, backtest_data):
    """Lance un backtest individuel pour une stratégie donnée."""
    print(f"\n" + "-"*50)
    print(f" TEST DE LA STRATÉGIE : {strategy.__class__.__name__}")
    print("-"*50)
    
    engine = BacktestEngine()
    engine.set_training_data(training_data)
    engine.set_backtest_data(backtest_data)
    engine.add_strategy(strategy)
    
    print(f"Calibration (IS)...")
    engine.train_strategies()
    
    print(f"Validation (OOS)...")
    engine.run()
    print("-"*50 + "\n")

def main():
    """
    Script principal pour la soumission du projet Systematic Trading Module.
    Utilise Massive API pour obtenir 6 mois de données 1-minute.
    Teste toutes les stratégies disponibles.
    """
    print("="*60)
    print("      PROJECT SUBMISSION: SYSTEMATIC TRADING (MASSIVE API)")
    print("="*60)
    
    try:
        # 1. Collecte des données via Massive API
        client = MassiveAPIClient()
        symbol = "SPY" # S&P 500 ETF pour plus de liquidité intraday
        
        # 6 mois = ~180 jours
        print(f"Récupération de 180 jours de données 1-minute pour {symbol}...")
        full_data = client.fetch_data(symbol, days_back=180, until=0)
        
        if full_data.empty:
            raise ValueError("Impossible de récupérer les données via Massive API.")

        # 2. Séparation In-Sample (IS) et Out-of-Sample (OOS)
        # IS: 5 mois (~150 jours), OOS: 1 mois (~30 jours)
        # On va faire une séparation simple par index temporel ou par position
        split_idx = int(len(full_data) * 0.83) # ~5/6 pour l'entraînement
        training_data = full_data.iloc[:split_idx]
        backtest_data = full_data.iloc[split_idx:]

        print(f"Période IS (Entraînement) : {training_data.index[0]} à {training_data.index[-1]} ({len(training_data)} bars)")
        print(f"Période OOS (Validation) : {backtest_data.index[0]} à {backtest_data.index[-1]} ({len(backtest_data)} bars)")

        # 3. Liste des stratégies à tester
        strategies = [
            CascadeStrategy(),
            ArimaStrategy(),
            MeanReversionStrategy(),
            MLHybridStrategy(),
            TrendFollowingStrategy()
        ]

        # 4. Exécution des backtests individuels
        for strat in strategies:
            # On s'assure que le symbole est correct pour la stratégie
            strat.symbol = symbol 
            run_backtest_for_strategy(strat, training_data, backtest_data)

        # 5. Test en tant que Portfolio (toutes les stratégies ensemble)
        print("\n" + "="*60)
        print("      TEST DU PORTFOLIO (TOUTES LES STRATÉGIES)")
        print("="*60)
        portfolio_engine = BacktestEngine()
        portfolio_engine.set_training_data(training_data)
        portfolio_engine.set_backtest_data(backtest_data)

        # Nouvelles instances pour le portfolio
        portfolio_strategies = [
            CascadeStrategy(),
            ArimaStrategy(),
            MeanReversionStrategy(),
            MLHybridStrategy(),
            TrendFollowingStrategy()
        ]

        for strat in portfolio_strategies:
            strat.symbol = symbol
            portfolio_engine.add_strategy(strat)

        print("Exécution du portfolio...")
        portfolio_engine.train_strategies()
        portfolio_engine.run()


        print("\nSoumission terminée avec succès.")

    except Exception as e:
        print(f"Erreur : {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
