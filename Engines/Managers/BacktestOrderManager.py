from Connections.TradingApp import TradingApp
from Types.Order import Order   
from ibapi.order import Order     

class BacktestOrderManager:
    def __init__(self):
        self.pending_orders : list[Order] = []
    
    def add_orders(self, orders: list[Order]):
        """Collecte les ordres de toutes les stratégies"""
        for order in orders:
            self.pending_orders.append(order)
        

    def aggregate_and_net(self):
        """Agrège et nette les ordres opposés"""
        final_orders = []
        
        for symbol, positions in self.pending_orders:
            net_quantity = positions['buy'] - positions['sell']
            
            if net_quantity > 0:
                final_orders.append(Order(symbol, 'BUY', abs(net_quantity)))
            elif net_quantity < 0:
                final_orders.append(Order(symbol, 'SELL', abs(net_quantity)))
            # Si net = 0, pas d'ordre (économie maximale!)
        
        self.pending_orders = final_orders  # Remplace les ordres par les ordres agrégés et netted

    def process_orders(self):
        """Processus complet : agrégation, netting, et préparation pour exécution"""

        trading_app = TradingApp.get_instance()

        for order in self.pending_orders:
            print(f"Traitement de l'ordre: {order}")

            
            pass
        
        return 
