import time
import threading
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData


class TradingApp(EClient, EWrapper):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self.nextOrderId: Optional[int] = None
        self.connected = False

    def error(self, reqId: int, errorCode: int, errorString: str, advanced: str = "") -> None:
        print(f"Error: {reqId}, {errorCode}, {errorString}, {advanced}")
        
        # CORRECTION: Ces codes ne sont PAS des erreurs de connexion mais des messages informatifs
        # 2104: Connexion données de marché OK
        # 2106: Connexion données HMDS OK  
        # 2158: Connexion serveur Sec-def OK
        # 2119: Connexion données de marché inactive mais OK
        if errorCode in [2104, 2106, 2158, 2119]:
            #print("Message informatif: Connexion serveur OK")
            return
            
        # Vrais codes d'erreur de connexion
        if errorCode in [502, 504, 1100, 1101, 1102]:
            print("Erreur de connexion détectée")
            self.connected = False

    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        self.connected = True
        print(f"Connexion établie. Prochain ordre ID: {orderId}")

    def connectionClosed(self) -> None:
        print("Connexion fermée")
        self.connected = False

    def get_historical_data(self, reqId: int, contract: Contract, end_date: str = "") -> pd.DataFrame:
        if not self.connected:
            print("Erreur: Non connecté à TWS/IB Gateway")
            return pd.DataFrame()

        self.data[reqId] = pd.DataFrame(columns=["time", "high", "low", "open", "close", "volume"])
        self.data[reqId].set_index("time", inplace=True)
        
        # Si pas de date spécifiée, utiliser la date d'aujourd'hui à 16:00 (fermeture marché US)
        if not end_date:
            from datetime import datetime, timedelta
            # Prendre hier à 16h00 pour être sûr d'avoir une journée complète
            yesterday = datetime.now() - timedelta(days=4)
            end_date = yesterday.strftime("%Y%m%d 17:00:00 US/Eastern")
        
        print(f"Demande de données historiques pour {contract.symbol} jusqu'à {end_date}...")
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime=end_date,  # Date de fin précise
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",  # TRADES au lieu de MIDPOINT pour plus de données
            useRTH=1,  # 0 = inclut pré/post marché, 1 = heures régulières seulement
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )
        
        # Attendre que les données arrivent
        timeout = 10  # 10 secondes timeout
        start_time = time.time()
        start_length = len(self.data[reqId])

        while len(self.data[reqId]) == 0 and (time.time() - start_time) < timeout: #Permet de vérifier si des données sont déjà arrivées
            time.sleep(0.3)
        
        while len(self.data[reqId]) > start_length and (time.time() - start_time) < timeout: # Permet de vérifier si des données continuent d'arriver
            time.sleep(1)
            start_length = len(self.data[reqId])

        if len(self.data[reqId]) == 0:
            print("Aucune donnée reçue dans le délai imparti")
        else:
            print(f"Données reçues: {len(self.data[reqId])} barres")
            print(f"Période: {self.data[reqId].index[0]} à {self.data[reqId].index[-1]}")
            
        return self.data[reqId]

    def historicalData(self, reqId: int, bar: BarData) -> None:
        if reqId not in self.data:
            return
            
        df = self.data[reqId]
        
        # Gestion robuste du format de date d'IB
        try:
            # Si bar.date est un timestamp numérique (int/float)
            if isinstance(bar.date, (int, float)):
                date_index = pd.to_datetime(bar.date, unit="s")
            else:
                # Si bar.date est une chaîne (format: "20240101 15:30:00")
                date_index = pd.to_datetime(bar.date)
        except (ValueError, TypeError):
            # Fallback: essayer de convertir en float puis en datetime
            try:
                date_index = pd.to_datetime(float(bar.date), unit="s")
            except (ValueError, TypeError):
                print(f"Erreur de conversion de date: {bar.date}")
                return
        
        df.loc[date_index, ["high", "low", "open", "close", "volume"]] = [bar.high, bar.low, bar.open, bar.close, bar.volume]
        
        df = df.astype(float)
        self.data[reqId] = df

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self.data_complete = True  # Indiquer que les données sont complètes
        print(f"Fin des données historiques pour reqId {reqId}")

    @staticmethod
    def get_contract(symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def place_order(self, contract: Contract, action: str, order_type: str, quantity: int) -> None:
        if not self.connected:
            print("Erreur: Non connecté à TWS/IB Gateway")
            return

        order = Order()
        order.action = action
        order.orderType = order_type
        order.totalQuantity = quantity

        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1
        print("Ordre placé")

