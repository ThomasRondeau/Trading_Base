import traceback
import pandas as pd
import numpy as np
import time
import warnings
import threading

# Importation des modules nécessaires
from Connections.Trading_App import TradingApp
from Old.CascadeTradingModel import CascadeTradingModel
from Old.TradingStrategy import TradingStrategy

warnings.filterwarnings('ignore')


 
def main():
    """
    Exemple d'utilisation du système de trading cascade.
    
    Cette fonction montre comment utiliser l'ensemble du système
    pour entraîner des modèles et générer des signaux de trading.
    """
    try:
        # Initialisation du modèle
        model = CascadeTradingModel()
        
        # Connexion à l'API de trading
        app = TradingApp()
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
        
        # Collecte des données d'entraînement
        print("Collecte des données d'entraînement...")
        training_data = model.collect_training_data(app, "SPY", days_back=5)
        
        # Entraînement des modèles
        print("Entraînement des modèles...")
        model.train_models(training_data)
        
        # Sauvegarde des modèles
        #model.save_models("cascade_models.pkl")
        
        # Affichage des performances
        summary = model.get_model_summary()
        print("\nRésumé des performances:")
        for model_name, perf in summary['performance_summary'].items():
            print(f"\n{model_name.upper()}:")
            if perf['type'] == 'classification':
                print(f"  Précision test: {perf['test_accuracy']:.3f}")
                print(f"  Échantillons: {perf['total_samples']}")
            else:
                print(f"  MAE test: {perf['test_mae']:.5f}")
                print(f"  Échantillons: {perf['total_samples']}")
            
            print(f"  Top features: {[f[0] for f in perf['top_features']]}")
        
        # Initialisation de la stratégie
        strategy = TradingStrategy(model)
        
        # Simulation de trading en temps réel
        print("\nDémarrage du trading en temps réel...")

        # Boucle de trading (exemple simplifié)
        """
        while True:
            try:
                # Récupération des données récentes
                current_data = app.get_historical_data(60)  # 60 dernières minutes
                current_price = current_data['close'].iloc[-1]
                
                # Traitement des données et génération de signaux
                signal = strategy.process_market_data(current_data, current_price)
                
                if signal:
                    print(f"\nSignal généré: {signal}")
                    
                    # Ici, vous intégreriez avec votre système de trading
                    # pour exécuter les ordres
                    
                # Attente avant la prochaine itération
                time.sleep(60)  # Vérification chaque minute
                
            except KeyboardInterrupt:
                print("Arrêt du trading...")
                break
            except Exception as e:
                print(f"Erreur dans la boucle de trading: {e}")
                continue
        """
                
        
        # Affichage des statistiques finales
        performance = strategy.get_performance_stats()
        print("\nStatistiques de performance:")
        print(performance)
        
    except Exception:
        traceback.print_exc()
    finally:
        # Nettoyage
        if 'app' in locals():
            app.disconnect()


if __name__ == "__main__":
    main()