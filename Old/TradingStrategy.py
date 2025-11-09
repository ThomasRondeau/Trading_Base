import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from Old.CascadeTradingModel import CascadeTradingModel


class TradingStrategy:
    """
    Stratégie de trading qui utilise le modèle cascade pour prendre des décisions.
    
    Cette classe encapsule la logique de trading et gère l'intégration
    avec le système de trading existant. Elle transforme les prédictions
    du modèle en signaux de trading concrets.
    """
    
    def __init__(self, model: CascadeTradingModel, risk_params: Dict = None):
        """
        Initialise la stratégie de trading.
        
        Args:
            model: Instance du modèle cascade entraîné
            risk_params: Paramètres de gestion du risque
        """
        self.model = model
        self.risk_params = risk_params or {
            'max_position_size': 0.02,  # 2% du capital par position
            'stop_loss_multiplier': 2.0,  # Stop loss à 2x l'amplitude prédite
            'take_profit_multiplier': 1.5,  # Take profit à 1.5x l'amplitude prédite
            'min_confidence_threshold': 0.6,  # Confiance minimale pour trader
            'min_amplitude_threshold': 0.003  # Amplitude minimale de 0.3%
        }
        
        self.active_positions = {}
        self.trade_history = []
        self.last_prediction = None
    
    def should_enter_trade(self, prediction: Dict, current_price: float) -> Tuple[bool, Dict]:
        """
        Détermine si nous devons entrer en position.
        
        Cette méthode applique nos filtres de risque et nos critères
        de qualité pour décider si une prédiction mérite d'être tradée.
        
        Args:
            prediction: Prédictions du modèle cascade
            current_price: Prix actuel du marché
            
        Returns:
            Tuple (should_trade, trade_signal)
        """
        # Vérification des prérequis de base
        if not prediction.get('opportunity', 0):
            return False, {'reason': 'Pas d\'opportunité détectée'}
        
        if prediction.get('direction') is None:
            return False, {'reason': 'Direction non déterminée'}
        
        # Vérification des seuils de confiance
        opp_confidence = prediction.get('opportunity_confidence', 0)
        dir_confidence = prediction.get('direction_confidence', 0)
        
        if opp_confidence < self.risk_params['min_confidence_threshold']:
            return False, {'reason': f'Confiance opportunité trop faible: {opp_confidence:.3f}'}
        
        if dir_confidence < self.risk_params['min_confidence_threshold']:
            return False, {'reason': f'Confiance direction trop faible: {dir_confidence:.3f}'}
        
        # Vérification de l'amplitude prédite
        amplitude = prediction.get('amplitude', 0)
        if amplitude < self.risk_params['min_amplitude_threshold']:
            return False, {'reason': f'Amplitude trop faible: {amplitude:.5f}'}
        
        # Calcul de la direction du trade
        direction = prediction['direction']  # 1 pour bullish, 0 pour bearish
        side = 'BUY' if direction == 1 else 'SELL'
        
        # Calcul des niveaux de stop et target
        stop_distance = amplitude * self.risk_params['stop_loss_multiplier']
        target_distance = amplitude * self.risk_params['take_profit_multiplier']
        
        if side == 'BUY':
            stop_price = current_price * (1 - stop_distance)
            target_price = current_price * (1 + target_distance)
        else:
            stop_price = current_price * (1 + stop_distance)
            target_price = current_price * (1 - target_distance)
        
        # Calcul de la taille de position
        risk_per_share = abs(current_price - stop_price)
        position_size = int(self.risk_params['max_position_size'] / risk_per_share) if risk_per_share > 0 else 0
        
        if position_size == 0:
            return False, {'reason': 'Taille de position calculée = 0'}
        
        # Signal de trading valide
        trade_signal = {
            'action': 'ENTER',
            'side': side,
            'entry_price': current_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'position_size': position_size,
            'predicted_amplitude': amplitude,
            'opportunity_confidence': opp_confidence,
            'direction_confidence': dir_confidence,
            'timestamp': datetime.now()
        }
        
        return True, trade_signal
    
    def should_exit_trade(self, position: Dict, current_price: float, 
                         current_time: datetime) -> Tuple[bool, str]:
        """
        Détermine si nous devons sortir d'une position existante.
        
        Cette méthode gère la sortie des positions en fonction du temps,
        des niveaux de stop/target, et des nouvelles prédictions.
        
        Args:
            position: Informations sur la position actuelle
            current_price: Prix actuel du marché
            current_time: Timestamp actuel
            
        Returns:
            Tuple (should_exit, exit_reason)
        """
        entry_price = position['entry_price']
        stop_price = position['stop_price']
        target_price = position['target_price']
        side = position['side']
        entry_time = position['timestamp']
        
        # Vérification du stop loss
        if side == 'BUY' and current_price <= stop_price:
            return True, 'STOP_LOSS'
        elif side == 'SELL' and current_price >= stop_price:
            return True, 'STOP_LOSS'
        
        # Vérification du take profit
        if side == 'BUY' and current_price >= target_price:
            return True, 'TAKE_PROFIT'
        elif side == 'SELL' and current_price <= target_price:
            return True, 'TAKE_PROFIT'
        
        # Sortie sur limite de temps (par défaut 15 minutes)
        time_limit = timedelta(minutes=15)
        if current_time - entry_time > time_limit:
            return True, 'TIME_LIMIT'
        
        # Sortie sur nouveau signal contraire
        if self.last_prediction:
            new_direction = self.last_prediction.get('direction')
            if new_direction is not None:
                current_direction = 1 if side == 'BUY' else 0
                if new_direction != current_direction:
                    confidence = self.last_prediction.get('direction_confidence', 0)
                    if confidence > 0.7:  # Seuil plus élevé pour les inversions
                        return True, 'SIGNAL_REVERSAL'
        
        return False, 'HOLD'
    
    def process_market_data(self, data: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Traite les nouvelles données de marché et génère des signaux.
        
        Cette méthode est appelée à chaque nouvelle bougie pour évaluer
        les opportunités de trading et gérer les positions existantes.
        
        Args:
            data: Données de marché récentes
            current_price: Prix actuel
            
        Returns:
            Signal de trading ou None
        """
        current_time = datetime.now()
        
        try:
            # Génération de la prédiction
            prediction = self.model.predict(data)
            self.last_prediction = prediction
            
            # Gestion des positions existantes
            for symbol, position in list(self.active_positions.items()):
                should_exit, exit_reason = self.should_exit_trade(
                    position, current_price, current_time
                )
                
                if should_exit:
                    # Création du signal de sortie
                    exit_signal = {
                        'action': 'EXIT',
                        'side': 'SELL' if position['side'] == 'BUY' else 'BUY',
                        'exit_price': current_price,
                        'exit_reason': exit_reason,
                        'position_size': position['position_size'],
                        'entry_price': position['entry_price'],
                        'pnl': self._calculate_pnl(position, current_price),
                        'timestamp': current_time
                    }
                    
                    # Enregistrement du trade
                    self.trade_history.append({
                        'entry': position,
                        'exit': exit_signal,
                        'duration': current_time - position['timestamp'],
                        'pnl': exit_signal['pnl']
                    })
                    
                    # Suppression de la position active
                    del self.active_positions[symbol]
                    
                    return exit_signal
            
            # Évaluation des nouvelles entrées si pas de position active
            if not self.active_positions:
                should_enter, trade_signal = self.should_enter_trade(prediction, current_price)
                
                if should_enter:
                    # Enregistrement de la nouvelle position
                    self.active_positions['current'] = trade_signal
                    
                    return trade_signal
            
            return None
            
        except Exception as e:
            print(f"Erreur lors du traitement des données: {e}")
            return None
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """
        Calcule le P&L d'une position.
        
        Args:
            position: Informations sur la position
            exit_price: Prix de sortie
            
        Returns:
            P&L de la position
        """
        entry_price = position['entry_price']
        position_size = position['position_size']
        side = position['side']
        
        if side == 'BUY':
            return (exit_price - entry_price) * position_size
        else:
            return (entry_price - exit_price) * position_size
    
    def get_performance_stats(self) -> Dict:
        """
        Calcule les statistiques de performance de la stratégie.
        
        Returns:
            Dict avec les métriques de performance
        """
        if not self.trade_history:
            return {"error": "Aucun trade dans l'historique"}
        
        trades = self.trade_history
        pnls = [trade['pnl'] for trade in trades]
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'average_win': np.mean(winning_trades) if winning_trades else 0,
            'average_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(pnls),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls),
            'average_trade_duration': np.mean([trade['duration'].total_seconds() / 60 for trade in trades])
        }
        
        return stats
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calcule le drawdown maximum."""
        if not pnls:
            return 0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        
        return np.max(drawdown)
    
    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        """Calcule le ratio de Sharpe."""
        if not pnls or len(pnls) < 2:
            return 0
        
        mean_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        if std_return == 0:
            return 0
        
        return mean_return / std_return