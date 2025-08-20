import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from state.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float  # days
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float


class PerformanceAnalyzer:
    """Analyze and track trading performance metrics."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_strategy_performance(self, strategy_name: str, 
                                     start_date: datetime = None,
                                     end_date: datetime = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a strategy."""
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Get trades and performance data
        trades_data = self._get_strategy_trades(strategy_name, start_date, end_date)
        performance_data = self._get_performance_data(strategy_name, start_date, end_date)
        
        if not trades_data:
            logger.warning(f"No trade data found for {strategy_name}")
            return self._empty_metrics()
        
        # Calculate metrics
        returns = self._calculate_returns(trades_data, performance_data)
        drawdowns = self._calculate_drawdowns(performance_data)
        
        return PerformanceMetrics(
            total_return=self._calculate_total_return(performance_data),
            annualized_return=self._calculate_annualized_return(returns, start_date, end_date),
            volatility=self._calculate_volatility(returns),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(drawdowns),
            win_rate=self._calculate_win_rate(trades_data),
            profit_factor=self._calculate_profit_factor(trades_data),
            total_trades=len(trades_data),
            avg_trade_duration=self._calculate_avg_trade_duration(trades_data),
            best_trade=max([t['pnl'] for t in trades_data] + [0]),
            worst_trade=min([t['pnl'] for t in trades_data] + [0]),
            consecutive_wins=self._calculate_consecutive_wins(trades_data),
            consecutive_losses=self._calculate_consecutive_losses(trades_data),
            calmar_ratio=self._calculate_calmar_ratio(returns, drawdowns),
            sortino_ratio=self._calculate_sortino_ratio(returns)
        )
    
    def _get_strategy_trades(self, strategy_name: str, start_date: datetime, 
                            end_date: datetime) -> List[Dict]:
        """Get trade data for analysis."""
        with self.db.get_session() as session:
            trades = session.query(self.db.Trade).filter(
                self.db.Trade.strategy_name == strategy_name,
                self.db.Trade.created_at >= start_date,
                self.db.Trade.created_at <= end_date,
                self.db.Trade.status == 'filled'
            ).all()
            
            trade_data = []
            for trade in trades:
                duration = None
                if trade.filled_at and trade.created_at:
                    duration = (trade.filled_at - trade.created_at).total_seconds() / 86400  # days
                
                trade_data.append({
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity or 0,
                    'entry_price': trade.price or 0,
                    'exit_price': trade.filled_price or trade.price or 0,
                    'pnl': trade.realized_pnl or 0,
                    'created_at': trade.created_at,
                    'filled_at': trade.filled_at,
                    'duration_days': duration or 0
                })
            
            return trade_data
    
    def _get_performance_data(self, strategy_name: str, start_date: datetime, 
                            end_date: datetime) -> List[Dict]:
        """Get daily performance data."""
        with self.db.get_session() as session:
            metrics = session.query(self.db.PerformanceMetric).filter(
                self.db.PerformanceMetric.strategy_name == strategy_name,
                self.db.PerformanceMetric.date >= start_date.date(),
                self.db.PerformanceMetric.date <= end_date.date()
            ).order_by(self.db.PerformanceMetric.date).all()
            
            return [
                {
                    'date': m.date,
                    'daily_pnl': m.daily_pnl or 0,
                    'cumulative_pnl': m.cumulative_pnl or 0,
                    'account_equity': m.account_equity or 0
                }
                for m in metrics
            ]
    
    def _calculate_returns(self, trades_data: List[Dict], 
                          performance_data: List[Dict]) -> np.ndarray:
        """Calculate daily returns."""
        if performance_data:
            returns = [p['daily_pnl'] / max(p['account_equity'], 1) for p in performance_data]
            return np.array(returns)
        else:
            # Fallback: estimate from trades
            if not trades_data:
                return np.array([])
            
            # Group trades by date and calculate daily returns
            daily_pnl = {}
            for trade in trades_data:
                if trade['filled_at']:
                    date_key = trade['filled_at'].date()
                    daily_pnl[date_key] = daily_pnl.get(date_key, 0) + trade['pnl']
            
            returns = list(daily_pnl.values())
            return np.array(returns)
    
    def _calculate_drawdowns(self, performance_data: List[Dict]) -> np.ndarray:
        """Calculate drawdown series."""
        if not performance_data:
            return np.array([])
        
        cumulative_pnl = [p['cumulative_pnl'] for p in performance_data]
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - peak) / np.maximum(peak, 1)  # Avoid division by zero
        return drawdown
    
    def _calculate_total_return(self, performance_data: List[Dict]) -> float:
        """Calculate total return."""
        if not performance_data:
            return 0.0
        
        final_pnl = performance_data[-1]['cumulative_pnl']
        initial_equity = performance_data[0]['account_equity']
        
        return final_pnl / max(initial_equity, 1) * 100  # Return as percentage
    
    def _calculate_annualized_return(self, returns: np.ndarray, 
                                   start_date: datetime, end_date: datetime) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        total_days = (end_date - start_date).days
        if total_days <= 0:
            return 0.0
        
        daily_return = np.mean(returns)
        annualized = daily_return * 252  # 252 trading days per year
        return annualized * 100  # Return as percentage
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) <= 1:
            return 0.0
        
        daily_vol = np.std(returns, ddof=1)
        annualized_vol = daily_vol * np.sqrt(252)
        return annualized_vol * 100  # Return as percentage
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, drawdowns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(drawdowns) == 0:
            return 0.0
        
        max_dd = np.min(drawdowns)
        return abs(max_dd) * 100  # Return as positive percentage
    
    def _calculate_win_rate(self, trades_data: List[Dict]) -> float:
        """Calculate win rate."""
        if not trades_data:
            return 0.0
        
        winning_trades = sum(1 for trade in trades_data if trade['pnl'] > 0)
        return (winning_trades / len(trades_data)) * 100
    
    def _calculate_profit_factor(self, trades_data: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades_data:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades_data if trade['pnl'] > 0)
        gross_loss = sum(abs(trade['pnl']) for trade in trades_data if trade['pnl'] < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_avg_trade_duration(self, trades_data: List[Dict]) -> float:
        """Calculate average trade duration in days."""
        if not trades_data:
            return 0.0
        
        durations = [trade['duration_days'] for trade in trades_data if trade['duration_days'] > 0]
        
        if not durations:
            return 0.0
        
        return np.mean(durations)
    
    def _calculate_consecutive_wins(self, trades_data: List[Dict]) -> int:
        """Calculate maximum consecutive wins."""
        if not trades_data:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sorted(trades_data, key=lambda x: x['filled_at'] or x['created_at']):
            if trade['pnl'] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades_data: List[Dict]) -> int:
        """Calculate maximum consecutive losses."""
        if not trades_data:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sorted(trades_data, key=lambda x: x['filled_at'] or x['created_at']):
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, drawdowns: np.ndarray) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) == 0 or len(drawdowns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_drawdown = abs(np.min(drawdowns))
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for cases with no data."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    def generate_performance_report(self, strategy_name: str, 
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> Dict[str, any]:
        """Generate comprehensive performance report."""
        
        metrics = self.calculate_strategy_performance(strategy_name, start_date, end_date)
        
        # Get additional context
        with self.db.get_session() as session:
            total_trades = session.query(self.db.Trade).filter(
                self.db.Trade.strategy_name == strategy_name,
                self.db.Trade.status == 'filled'
            ).count()
            
            active_positions = session.query(self.db.Position).filter(
                self.db.Position.strategy_name == strategy_name,
                self.db.Position.is_active == True
            ).count()
        
        return {
            'strategy_name': strategy_name,
            'analysis_period': {
                'start_date': start_date.strftime('%Y-%m-%d') if start_date else None,
                'end_date': end_date.strftime('%Y-%m-%d') if end_date else None
            },
            'returns': {
                'total_return_pct': round(metrics.total_return, 2),
                'annualized_return_pct': round(metrics.annualized_return, 2),
                'volatility_pct': round(metrics.volatility, 2),
                'sharpe_ratio': round(metrics.sharpe_ratio, 3),
                'sortino_ratio': round(metrics.sortino_ratio, 3),
                'calmar_ratio': round(metrics.calmar_ratio, 3)
            },
            'risk': {
                'max_drawdown_pct': round(metrics.max_drawdown, 2),
                'worst_trade': round(metrics.worst_trade, 2),
                'consecutive_losses': metrics.consecutive_losses
            },
            'trading': {
                'total_trades': metrics.total_trades,
                'win_rate_pct': round(metrics.win_rate, 2),
                'profit_factor': round(metrics.profit_factor, 3),
                'best_trade': round(metrics.best_trade, 2),
                'avg_trade_duration_days': round(metrics.avg_trade_duration, 2),
                'consecutive_wins': metrics.consecutive_wins
            },
            'current_status': {
                'active_positions': active_positions,
                'total_historical_trades': total_trades
            }
        }
    
    def export_performance_csv(self, file_path: str, strategy_name: str = None):
        """Export performance metrics to CSV."""
        try:
            with self.db.get_session() as session:
                query = session.query(self.db.PerformanceMetric)
                if strategy_name:
                    query = query.filter(self.db.PerformanceMetric.strategy_name == strategy_name)
                
                metrics = query.order_by(self.db.PerformanceMetric.date).all()
                
                if metrics:
                    data = []
                    for m in metrics:
                        data.append({
                            'date': m.date,
                            'strategy': m.strategy_name,
                            'daily_pnl': m.daily_pnl,
                            'cumulative_pnl': m.cumulative_pnl,
                            'daily_trades': m.daily_trades,
                            'win_rate': m.win_rate,
                            'account_equity': m.account_equity,
                            'active_positions': m.active_positions,
                            'max_drawdown': m.max_drawdown
                        })
                    
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, index=False)
                    logger.info(f"Exported {len(metrics)} performance records to {file_path}")
                else:
                    logger.info("No performance data to export")
                    
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            raise