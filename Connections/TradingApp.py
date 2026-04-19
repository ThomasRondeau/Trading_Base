import time
import threading
from datetime import datetime
from typing import Dict, Optional, Callable
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData, TickerId
from ibapi.ticktype import TickTypeEnum


class TradingApp(EClient, EWrapper):

    _instance : Optional['TradingApp'] = None

    
    @classmethod
    def get_instance(cls) -> 'TradingApp':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    def __init__(self) -> None:

        if TradingApp._instance is not None:
            raise Exception("Use get_instance() instead")
        
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self.nextOrderId: Optional[int] = None
        self.connected = False
        # Nouvelles variables pour le streaming live
        self.live_data: Dict[int, Dict] = {}
        self.live_callbacks: Dict[int, Callable] = {}

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

        self.data[reqId] = pd.DataFrame(columns=["Datetime", "high", "low", "open", "close", "volume"])
        self.data[reqId].set_index("Datetime", inplace=True)
        
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
            useRTH=2,  # 0 = inclut pré/post marché, 1 = heures régulières seulement
            formatDate=2,  # 1 = format lisible, 2 = timestamp
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
        
        df.loc[date_index, ["high", "low", "open", "close", "volume"]] = [
            float(bar.high), float(bar.low), float(bar.open), float(bar.close), float(bar.volume)
        ]
        
        self.data[reqId] = df

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        self.data_complete = True  # Indiquer que les données sont complètes
        print(f"Fin des données historiques pour reqId {reqId}")

    @staticmethod
    def get_contract(symbol: str, exchange: str = "SMART") -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.primaryExchange = exchange if exchange != "SMART" else ""
        contract.exchange = exchange
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


    # Pour le live trading 

    def start_live_data(self, reqId: int, contract: Contract, callback: Optional[Callable] = None) -> None:
        """
        Démarre le streaming de données en temps réel pour un contrat.
        
        Args:
            reqId: ID unique pour cette requête
            contract: Contrat à suivre
            callback: Fonction optionnelle appelée à chaque mise à jour (reçoit reqId et dict des données)
        """
        if not self.connected:
            print("Erreur: Non connecté à TWS/IB Gateway")
            return
        
        # Initialiser le stockage des données live
        self.live_data[reqId] = {
            'symbol': contract.symbol,
            'last_price': None,
            'bid': None,
            'ask': None,
            'volume': None,
            'last_update': None
        }
        
        if callback:
            self.live_callbacks[reqId] = callback
        
        # Demander les données de marché en temps réel
        self.reqMktData(reqId, contract, "", False, False, [])
        print(f"Streaming live démarré pour {contract.symbol} (reqId: {reqId})")
    
    def stop_live_data(self, reqId: int) -> None:
        """Arrête le streaming de données pour un reqId donné."""
        if reqId in self.live_data:
            self.cancelMktData(reqId)
            del self.live_data[reqId]
            if reqId in self.live_callbacks:
                del self.live_callbacks[reqId]
            print(f"Streaming live arrêté pour reqId {reqId}")
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib) -> None:
        """Callback appelé quand un prix change."""
        if reqId not in self.live_data:
            return
        
        tick_name = TickTypeEnum.to_str(tickType)
        
        # Mettre à jour les prix selon le type de tick
        if tickType == 4:  # LAST
            self.live_data[reqId]['last_price'] = price
        elif tickType == 1:  # BID
            self.live_data[reqId]['bid'] = price
        elif tickType == 2:  # ASK
            self.live_data[reqId]['ask'] = price
        
        self.live_data[reqId]['last_update'] = datetime.now()
        
        # Appeler le callback si défini
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, self.live_data[reqId].copy())
    
    def tickSize(self, reqId: TickerId, tickType: int, size: int) -> None:
        """Callback appelé quand une taille change."""
        if reqId not in self.live_data:
            return
        
        # Volume
        if tickType == 8:  # VOLUME
            self.live_data[reqId]['volume'] = size
        
        self.live_data[reqId]['last_update'] = datetime.now()
        
        # Appeler le callback si défini
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, self.live_data[reqId].copy())
    
    def get_live_data(self, reqId: int) -> Optional[Dict]:
        """Récupère les dernières données live pour un reqId."""
        return self.live_data.get(reqId, None)
    
    def start_live_bars(self, reqId: int, contract: Contract, bar_size: int = 5, 
                        callback: Optional[Callable] = None) -> None:
        """
        Démarre le streaming de barres en temps réel.
        
        Args:
            reqId: ID unique pour cette requête
            contract: Contrat à suivre
            bar_size: Taille des barres en secondes (5, 10, 15, 30, 60, etc.)
            callback: Fonction appelée à chaque nouvelle barre (reçoit reqId et BarData)
        """
        if not self.connected:
            print("Erreur: Non connecté à TWS/IB Gateway")
            return
        
        if callback:
            self.live_callbacks[reqId] = callback
        
        self.reqRealTimeBars(
            reqId=reqId,
            contract=contract,
            barSize=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            realTimeBarsOptions=[]
        )
        print(f"Streaming de barres live démarré pour {contract.symbol} (reqId: {reqId}, {bar_size}s)")
    
    def stop_live_bars(self, reqId: int) -> None:
        """Arrête le streaming de barres pour un reqId donné."""
        self.cancelRealTimeBars(reqId)
        if reqId in self.live_callbacks:
            del self.live_callbacks[reqId]
        print(f"Streaming de barres arrêté pour reqId {reqId}")
    
    def realtimeBar(self, reqId: TickerId, time: int, open_: float, high: float, 
                    low: float, close: float, volume: int, wap: float, count: int) -> None:
        """Callback appelé à chaque nouvelle barre temps réel."""
        bar_data = {
            'time': datetime.fromtimestamp(time),
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'wap': wap,
            'count': count
        }
        
        # Appeler le callback si défini
        if reqId in self.live_callbacks:
            self.live_callbacks[reqId](reqId, bar_data)

