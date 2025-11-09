import time
import threading
import traceback

from Connections.Trading_App import TradingApp
from Strategies.CascadeStrategy import CascadeStrategy


def main():
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=5)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    timeout = 10  # 10 secondes
    start_time = time.time()
    
    while not app.connected and (time.time() - start_time) < timeout:
        print("En attente de connexion...")
        time.sleep(1)

    if not app.connected:
        print("Not Connected to TWS/IB Gateway")
        return
    print("Connected to TWS/IB Gateway")

    try:
        strategy = CascadeStrategy(app)
        strategy.train()

    except Exception as e:
        print(f"Erreur lors de l'exécution de la stratégie: {e}")
        print(traceback.format_exc())
    


    if app.connected:
        app.disconnect()
        print("Déconnexion")


if __name__ == "__main__":
    main()