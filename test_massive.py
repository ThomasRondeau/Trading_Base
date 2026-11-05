from Connections.MassiveAPI import MassiveAPIClient
import pandas as pd

def test_massive():
    print("Test de Massive API...")
    client = MassiveAPIClient()
    
    # Test sur AAPL, 2 jours en arrière pour voir si on récupère des barres
    df = client.fetch_data("AAPL", days_back=2, until=0)
    
    if not df.empty:
        print("SUCCÈS : Données récupérées !")
        print(df.head())
        print(f"Nombre de points : {len(df)}")
    else:
        print("ÉCHEC : Aucune donnée récupérée. Vérifiez la clé API ou l'état du service.")

if __name__ == "__main__":
    test_massive()
