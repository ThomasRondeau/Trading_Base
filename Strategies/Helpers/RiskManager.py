from Strategies.TradingStrategyInterface import TradingStrategyInterface
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
from queue import Queue
import pandas as pd


class RiskManager:
    """
    Gère le risque : sizing, stops, max positions, etc.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_size_pct: float = 0.1,  # 10% du capital par position
                 max_risk_per_trade_pct: float = 0.02,  # 2% du capital par trade
                 max_total_exposure_pct: float = 1.0,  # 100% du capital total
                 stop_loss_pct: float = 0.05):  # 5% stop loss
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Tracking
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, entry_price, stop_price}}
        self.trades_history: List[Dict] = []
        self.pnl: float = 0
    
    def calculate_position_size(self, symbol: str, current_price: float) -> int:
        """
        Calcule la taille de position optimale basée sur le capital et le risque.
        
        :return: Nombre d'actions à acheter
        """
        # Taille max basée sur le capital
        max_position_value = self.current_capital * self.max_position_size_pct
        max_shares_by_capital = int(max_position_value / current_price)
        
        # Taille basée sur le risque
        risk_amount = self.current_capital * self.max_risk_per_trade_pct
        stop_distance = current_price * self.stop_loss_pct
        max_shares_by_risk = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        
        # Prendre le minimum des deux
        position_size = min(max_shares_by_capital, max_shares_by_risk)
        
        # Vérifier l'exposition totale
        total_exposure = sum(pos['quantity'] * pos['entry_price'] 
                            for pos in self.positions.values())
        remaining_capacity = (self.current_capital * self.max_total_exposure_pct) - total_exposure
        max_shares_by_exposure = int(remaining_capacity / current_price)
        
        position_size = min(position_size, max_shares_by_exposure)
        
        return max(0, position_size)
    
    def can_open_position(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """
        Vérifie si on peut ouvrir une position.
        
        :return: (can_open, reason)
        """
        # Déjà en position ?
        if symbol in self.positions:
            return False, f"Already in position for {symbol}"
        
        # Calculer la taille
        size = self.calculate_position_size(symbol, current_price)
        if size <= 0:
            return False, "Position size would be 0 or negative"
        
        # Vérifier le capital
        position_value = size * current_price
        if position_value > self.current_capital * self.max_position_size_pct:
            return False, "Would exceed max position size"
        
        return True, f"Can open {size} shares"
    
    def open_position(self, symbol: str, quantity: int, entry_price: float) -> Dict:
        """
        Enregistre l'ouverture d'une position et calcule le stop loss.
        
        :return: Dict avec les infos de la position
        """
        stop_price = entry_price * (1 - self.stop_loss_pct)
        
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'entry_time': datetime.now()
        }
        
        print(f"[RiskManager] Position opened: {quantity} {symbol} @ {entry_price:.2f}")
        print(f"[RiskManager] Stop loss set at: {stop_price:.2f} (-{self.stop_loss_pct*100}%)")
        
        return self.positions[symbol]
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "strategy") -> Dict:
        """
        Ferme une position et calcule le P&L.
        
        :return: Dict avec les infos du trade
        """
        if symbol not in self.positions:
            return {}
        
        pos = self.positions[symbol]
        pnl = (exit_price - pos['entry_price']) * pos['quantity']
        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        
        self.pnl += pnl
        self.current_capital += pnl
        
        trade = {
            'symbol': symbol,
            'quantity': pos['quantity'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        
        self.trades_history.append(trade)
        del self.positions[symbol]
        
        print(f"[RiskManager] Position closed: {symbol}")
        print(f"[RiskManager] P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) - Reason: {reason}")
        
        return trade
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Vérifie si le stop loss est touché.
        
        :return: True si le stop est touché
        """
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        if current_price <= pos['stop_price']:
            print(f"[RiskManager] ⚠️  STOP LOSS triggered for {symbol}!")
            print(f"[RiskManager] Price {current_price:.2f} <= Stop {pos['stop_price']:.2f}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de trading."""
        if not self.trades_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'current_capital': self.current_capital,
                'return_pct': 0
            }
        
        winning = [t for t in self.trades_history if t['pnl'] > 0]
        losing = [t for t in self.trades_history if t['pnl'] < 0]
        
        return {
            'total_trades': len(self.trades_history),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades_history) * 100,
            'total_pnl': self.pnl,
            'avg_win': sum(t['pnl'] for t in winning) / len(winning) if winning else 0,
            'avg_loss': sum(t['pnl'] for t in losing) / len(losing) if losing else 0,
            'current_capital': self.current_capital,
            'return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        }
