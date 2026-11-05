from typing import Dict, Any, Optional, List



class RiskManager:
    """
    Gère le risque : sizing, stops, max positions, etc.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_size_pct: float = 0.1,  # 10% du capital par position
                 max_risk_per_trade_pct: float = 0.02,  # 2% du capital par trade
                 max_total_exposure_pct: float = 1.0,  # 100% du capital total
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.05):  # 5% take profit
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Tracking
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, entry_price, stop_price}}
        self.trades_history: List[Dict] = []
        self.pnl: float = 0
    
    