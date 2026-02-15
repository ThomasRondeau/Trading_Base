from Strategies.TradingStrategyInterface import TradingStrategyInterface
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
from queue import Queue
import pandas as pd


class StrategyRunner:
    """
    Gère toute la plomberie : événements, threads, callbacks IBKR.
    Les stratégies n'ont qu'à implémenter la logique de trading.
    """
    
    def __init__(self, trading_app, strategy: TradingStrategyInterface):
        self.trading_app = trading_app
        self.strategy = strategy
        self.strategy._runner = self  # Injection du runner dans la stratégie
        
        # Gestion des événements
        self.event_queue = Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Stockage des données
        self.bars_data: Dict[str, List[Dict]] = {}  # {symbol: [bars]}
        self.positions: Dict[str, float] = {}  # {symbol: quantity}
        self.req_ids: Dict[str, int] = {}  # {symbol: reqId}
        self.symbols: List[str] = []
        
        # Configuration
        self.bar_size = 5  # secondes
        self.max_bars_stored = 1000  # Nombre max de barres en mémoire
    
    def configure(self, symbols: List[str], bar_size: int = 5, max_bars: int = 1000) -> None:
        """Configure le runner."""
        self.symbols = symbols
        self.bar_size = bar_size
        self.max_bars_stored = max_bars
    
    def _on_bar_callback(self, symbol: str, req_id: int, bar_data: Dict) -> None:
        """
        Callback interne appelé par IBKR.
        Met l'événement dans la queue pour traitement thread-safe.
        """
        self.event_queue.put({
            'type': 'bar',
            'symbol': symbol,
            'data': bar_data
        })
    
    def _process_events(self) -> None:
        """Thread de traitement des événements."""
        print("[StrategyRunner] Event processing thread started")
        
        while self.is_running:
            try:
                if not self.event_queue.empty():
                    event = self.event_queue.get(timeout=0.1)
                    
                    if event['type'] == 'bar':
                        self._handle_bar(event['symbol'], event['data'])
                    elif event['type'] == 'order_filled':
                        self._handle_order_filled(event)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[StrategyRunner] Error processing event: {e}")
                import traceback
                traceback.print_exc()
        
        print("[StrategyRunner] Event processing thread stopped")
    
    def _handle_bar(self, symbol: str, bar_data: Dict) -> None:
        """Gère l'arrivée d'une nouvelle barre."""
        # Stocker la barre
        if symbol not in self.bars_data:
            self.bars_data[symbol] = []
        
        self.bars_data[symbol].append(bar_data)
        
        # Limiter la taille en mémoire
        if len(self.bars_data[symbol]) > self.max_bars_stored:
            self.bars_data[symbol] = self.bars_data[symbol][-self.max_bars_stored:]
        
        # Appeler la stratégie
        try:
            self.strategy.on_bar(symbol, bar_data)
        except Exception as e:
            print(f"[StrategyRunner] Error in strategy.on_bar(): {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_order_filled(self, event: Dict) -> None:
        """Gère le remplissage d'un ordre."""
        symbol = event['symbol']
        side = event['side']
        quantity = event['quantity']
        price = event['price']
        
        # Mettre à jour la position
        if side == 'BUY':
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif side == 'SELL':
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Notifier la stratégie
        try:
            self.strategy.on_order_filled(symbol, side, quantity, price)
        except Exception as e:
            print(f"[StrategyRunner] Error in strategy.on_order_filled(): {e}")
    
    def load_historical_data(self) -> None:
        """Charge les données historiques pour tous les symboles."""
        print("[StrategyRunner] Loading historical data...")
        
        for idx, symbol in enumerate(self.symbols):
            reqId = 5000 + idx
            self.req_ids[symbol] = reqId
            
            contract = self.trading_app.get_contract(symbol)
            historical_df = self.trading_app.get_historical_data(reqId, contract)
            
            if not historical_df.empty:
                self.bars_data[symbol] = []
                for timestamp, row in historical_df.iterrows():
                    self.bars_data[symbol].append({
                        'time': timestamp,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                
                print(f"[StrategyRunner] Loaded {len(self.bars_data[symbol])} bars for {symbol}")
    
    def start_live_streaming(self) -> None:
        """Démarre le streaming de données en temps réel."""
        print("[StrategyRunner] Starting live streaming...")
        
        for idx, symbol in enumerate(self.symbols):
            contract = self.trading_app.get_contract(symbol)
            stream_req_id = 6000 + idx
            
            # Créer un callback qui capture le symbole
            def make_callback(sym):
                return lambda req_id, bar_data: self._on_bar_callback(sym, req_id, bar_data)
            
            self.trading_app.start_live_bars(
                reqId=stream_req_id,
                contract=contract,
                bar_size=self.bar_size,
                callback=make_callback(symbol)
            )
            
            print(f"[StrategyRunner] Streaming started for {symbol} ({self.bar_size}s bars)")
    
    def stop_live_streaming(self) -> None:
        """Arrête le streaming."""
        print("[StrategyRunner] Stopping live streaming...")
        
        for idx in range(len(self.symbols)):
            stream_req_id = 6000 + idx
            try:
                self.trading_app.stop_live_bars(stream_req_id)
            except Exception as e:
                print(f"[StrategyRunner] Error stopping stream: {e}")
    
    def run(self) -> None:
        """Démarre le runner et la stratégie."""
        if not self.trading_app or not self.trading_app.connected:
            print("[StrategyRunner] Error: Not connected to IBKR")
            return
        
        print("\n" + "="*70)
        print("STRATEGY RUNNER - Starting")
        print("="*70)
        
        self.is_running = True
        
        # 1. Charger l'historique
        self.load_historical_data()
        
        # 2. Notifier la stratégie qu'elle démarre
        try:
            self.strategy.on_start()
        except Exception as e:
            print(f"[StrategyRunner] Error in strategy.on_start(): {e}")
        
        # 3. Démarrer le streaming
        self.start_live_streaming()
        
        # 4. Lancer le thread de traitement
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        print("="*70)
        print("STRATEGY RUNNER - Running")
        print("="*70 + "\n")
    
    def stop(self) -> None:
        """Arrête le runner et la stratégie."""
        print("\n[StrategyRunner] Stopping...")
        
        # 1. Notifier la stratégie
        try:
            self.strategy.on_stop()
        except Exception as e:
            print(f"[StrategyRunner] Error in strategy.on_stop(): {e}")
        
        # 2. Arrêter le streaming
        self.stop_live_streaming()
        
        # 3. Arrêter le thread de traitement
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        print("[StrategyRunner] Stopped")

    # ============= API pour les stratégies =============
    
    def place_order(self, symbol: str, side: str, quantity: int) -> None:
        """Place un ordre."""
        if not self.trading_app or not self.trading_app.connected:
            print("[StrategyRunner] Cannot place order: not connected")
            return
        
        contract = self.trading_app.get_contract(symbol)
        print(f"[StrategyRunner] Placing {side} order: {quantity} {symbol}")
        
        self.trading_app.place_order(contract, side, "MKT", quantity)
        
        # Simuler le remplissage immédiat (en production, attendez orderStatus)
        # Pour une vraie implémentation, gérez cela dans le callback orderStatus
        current_price = self.bars_data[symbol][-1]['close'] if symbol in self.bars_data else 0
        self.event_queue.put({
            'type': 'order_filled',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': current_price
        })
    
    def get_position(self, symbol: str) -> float:
        """Récupère la position."""
        return self.positions.get(symbol, 0)
    
    def get_bars(self, symbol: str, count: int = None) -> List[Dict]:
        """Récupère les barres."""
        bars = self.bars_data.get(symbol, [])
        if count:
            return bars[-count:]
        return bars
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le status du runner."""
        return {
            'running': self.is_running,
            'symbols': self.symbols,
            'positions': self.positions,
            'bars_count': {sym: len(bars) for sym, bars in self.bars_data.items()}
        }
