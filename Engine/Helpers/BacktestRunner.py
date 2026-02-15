from Strategies import TradingStrategyInterface
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
from queue import Queue
import pandas as pd

from Strategies.Helpers.RiskManager import RiskManager


class BacktestRunner:
    """
    Fait tourner une stratégie sur des données historiques.
    Simule le comportement du StrategyRunner mais en mode backtest.
    """
    
    def __init__(self, strategy: TradingStrategyInterface, risk_manager: Optional[RiskManager] = None):
        self.strategy = strategy
        self.strategy._runner = self  # Injection
        self.risk_manager = risk_manager or RiskManager()
        
        # Données
        self.historical_data: Dict[str, pd.DataFrame] = {}  # {symbol: df}
        self.current_prices: Dict[str, float] = {}
        self.positions: Dict[str, float] = {}
        
        # Résultats
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
    
    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Charge les données historiques pour un symbole.
        
        :param df: DataFrame avec colonnes: time (index), open, high, low, close, volume
        """
        self.historical_data[symbol] = df.copy()
        print(f"[BacktestRunner] Loaded {len(df)} bars for {symbol}")
    
    def run(self) -> Dict[str, Any]:
        """
        Exécute le backtest.
        Rejoue les données bar par bar.
        """
        print("\n" + "="*70)
        print("BACKTEST RUNNER - Starting")
        print("="*70)
        
        # Notifier la stratégie
        self.strategy.on_start()
        
        # Trouver toutes les timestamps uniques (pour synchroniser les symboles)
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        print(f"[BacktestRunner] Running backtest on {len(timestamps)} bars")
        print(f"[BacktestRunner] Period: {timestamps[0]} to {timestamps[-1]}\n")
        
        # Rejouer bar par bar
        for i, timestamp in enumerate(timestamps):
            # Pour chaque symbole à ce timestamp
            for symbol, df in self.historical_data.items():
                if timestamp in df.index:
                    bar = df.loc[timestamp]
                    
                    bar_dict = {
                        'time': timestamp,
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    }
                    
                    self.current_prices[symbol] = bar['close']
                    
                    # Vérifier les stops
                    if self.risk_manager.check_stop_loss(symbol, bar['close']):
                        pos = self.risk_manager.positions[symbol]
                        self.risk_manager.close_position(symbol, bar['close'], reason="stop_loss")
                        self.positions[symbol] = 0
                    
                    # Appeler la stratégie
                    self.strategy.on_bar(symbol, bar_dict)
            
            # Enregistrer l'equity
            if i % 100 == 0:  # Toutes les 100 barres
                self.equity_curve.append({
                    'time': timestamp,
                    'capital': self.risk_manager.current_capital,
                    'positions': len(self.risk_manager.positions)
                })
        
        # Fermer toutes les positions ouvertes
        for symbol in list(self.risk_manager.positions.keys()):
            price = self.current_prices.get(symbol, 0)
            self.risk_manager.close_position(symbol, price, reason="backtest_end")
        
        # Notifier la stratégie
        self.strategy.on_stop()
        
        # Calculer les résultats
        results = self._calculate_results()
        
        print("\n" + "="*70)
        print("BACKTEST RUNNER - Completed")
        print("="*70)
        self._print_results(results)
        
        return results
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        stats = self.risk_manager.get_statistics()
        
        # Calcul du Sharpe ratio (simplifié)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['capital'] - self.equity_curve[i-1]['capital']) / self.equity_curve[i-1]['capital']
                returns.append(ret)
            
            if returns:
                import statistics
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_values = [e['capital'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        return {
            **stats,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'equity_curve': self.equity_curve,
            'trades': self.risk_manager.trades_history
        }
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Affiche les résultats du backtest."""
        print(f"\n📊 BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"Initial Capital:    ${self.risk_manager.initial_capital:,.2f}")
        print(f"Final Capital:      ${results['current_capital']:,.2f}")
        print(f"Total Return:       {results['return_pct']:+.2f}%")
        print(f"Total P&L:          ${results['total_pnl']:+,.2f}")
        print(f"\nTotal Trades:       {results['total_trades']}")
        print(f"Winning Trades:     {results['winning_trades']}")
        print(f"Losing Trades:      {results['losing_trades']}")
        print(f"Win Rate:           {results['win_rate']:.2f}%")
        
        if results['winning_trades'] > 0:
            print(f"Avg Win:            ${results['avg_win']:,.2f}")
        if results['losing_trades'] > 0:
            print(f"Avg Loss:           ${results['avg_loss']:,.2f}")
        
        print(f"\nSharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
        print(f"{'='*70}\n")
    
    # ============= API pour les stratégies =============
    
    def place_order(self, symbol: str, side: str, quantity: int) -> None:
        """Simule le placement d'un ordre."""
        current_price = self.current_prices.get(symbol, 0)
        
        if side == 'BUY':
            # Vérifier avec le risk manager
            can_open, reason = self.risk_manager.can_open_position(symbol, current_price)
            if not can_open:
                print(f"[BacktestRunner] Cannot BUY {symbol}: {reason}")
                return
            
            # Calculer la taille optimale
            optimal_size = self.risk_manager.calculate_position_size(symbol, current_price)
            quantity = min(quantity, optimal_size)
            
            if quantity <= 0:
                print(f"[BacktestRunner] Optimal size is 0 for {symbol}")
                return
            
            # Ouvrir la position
            self.risk_manager.open_position(symbol, quantity, current_price)
            self.positions[symbol] = quantity
            
            # Notifier la stratégie
            self.strategy.on_order_filled(symbol, 'BUY', quantity, current_price)
        
        elif side == 'SELL':
            if symbol not in self.risk_manager.positions:
                print(f"[BacktestRunner] Cannot SELL {symbol}: no position")
                return
            
            pos = self.risk_manager.positions[symbol]
            quantity = min(quantity, pos['quantity'])
            
            # Fermer la position
            self.risk_manager.close_position(symbol, current_price, reason="strategy")
            self.positions[symbol] = 0
            
            # Notifier la stratégie
            self.strategy.on_order_filled(symbol, 'SELL', quantity, current_price)
    
    def get_position(self, symbol: str) -> float:
        """Retourne la position."""
        if symbol in self.risk_manager.positions:
            return self.risk_manager.positions[symbol]['quantity']
        return 0
    
    def get_bars(self, symbol: str, count: int = None) -> List[Dict]:
        """Retourne les barres (pour compatibilité avec StrategyRunner)."""
        # En backtest, on retourne toutes les barres jusqu'au moment actuel
        # Cette méthode est appelée dans on_bar, donc on a accès à toutes les barres passées
        if symbol not in self.historical_data:
            return []
        
        df = self.historical_data[symbol]
        bars = []
        
        for timestamp, row in df.iterrows():
            bars.append({
                'time': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        
        if count:
            return bars[-count:]
        return bars