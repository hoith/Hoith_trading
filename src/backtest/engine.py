"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from strategies.base import Strategy, StrategySignal, StrategyPosition, PositionSide, OrderType
from backtest.broker import BacktestBroker
from backtest.data import BacktestDataProvider
from risk.sizing import PositionSizer, SizingMethod
from risk.breakers import RiskBreaker
from risk.correlation import CorrelationManager


class BacktestStatus(Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    commission_equity: float = 0.0
    commission_options: float = 0.65
    slippage_equity_bps: int = 5
    slippage_options_pct: float = 1.0
    benchmark: str = "SPY"
    
    # Risk management
    max_positions: int = 5
    position_size_pct: float = 20.0
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 3.0
    
    # Execution settings
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    delay_minutes: int = 1  # Execution delay after signal


@dataclass
class BacktestTrade:
    """Individual trade in backtest."""
    trade_id: str
    symbol: str
    strategy: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    
    # Data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[BacktestTrade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    
    # Strategy breakdown
    strategy_results: Dict[str, Dict] = field(default_factory=dict)


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.status = BacktestStatus.PENDING
        
        # Initialize components
        self.broker = BacktestBroker(config)
        self.data_provider = BacktestDataProvider()
        
        # Risk management
        self.position_sizer = PositionSizer(
            account_equity=config.initial_capital,
            max_positions=config.max_positions,
            risk_per_trade_pct=config.risk_per_trade_pct
        )
        
        self.risk_breaker = RiskBreaker(
            account_equity=config.initial_capital
        )
        
        self.correlation_manager = CorrelationManager(
            max_total_positions=config.max_positions
        )
        
        # State
        self.strategies: List[Strategy] = []
        self.current_positions: Dict[str, StrategyPosition] = {}
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trades: List[BacktestTrade] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def add_strategy(self, strategy: Strategy) -> None:
        """Add strategy to backtest.
        
        Args:
            strategy: Strategy instance to backtest
        """
        if not isinstance(strategy, Strategy):
            raise ValueError("Strategy must inherit from Strategy base class")
        
        self.strategies.append(strategy)
        
    def run(self) -> BacktestResults:
        """Run the complete backtest.
        
        Returns:
            Comprehensive backtest results
        """
        try:
            self.status = BacktestStatus.RUNNING
            self.start_time = datetime.now()
            
            # Prepare data
            market_data = self._prepare_market_data()
            if market_data.empty:
                raise ValueError("No market data available for backtest period")
            
            # Run simulation
            self._run_simulation(market_data)
            
            # Calculate results
            results = self._calculate_results()
            
            self.status = BacktestStatus.COMPLETED
            self.end_time = datetime.now()
            
            return results
            
        except Exception as e:
            self.status = BacktestStatus.FAILED
            raise RuntimeError(f"Backtest failed: {str(e)}")
    
    def _prepare_market_data(self) -> pd.DataFrame:
        """Prepare market data for backtesting.
        
        Returns:
            Multi-symbol market data DataFrame
        """
        # Get all symbols from all strategies
        symbols = set()
        for strategy in self.strategies:
            if hasattr(strategy, 'universe'):
                symbols.update(strategy.universe)
        
        if not symbols:
            raise ValueError("No symbols found in strategy universes")
        
        # Load data for all symbols
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        market_data = self.data_provider.get_historical_data(
            symbols=list(symbols),
            start_date=start_date,
            end_date=end_date
        )
        
        return market_data
    
    def _run_simulation(self, market_data: pd.DataFrame) -> None:
        """Run the main simulation loop.
        
        Args:
            market_data: Historical market data
        """
        # Get trading dates
        trading_dates = market_data.index.unique()
        
        for current_date in trading_dates:
            # Update broker with current date
            self.broker.set_current_date(current_date)
            
            # Get current market data
            current_data = self._get_current_market_data(market_data, current_date)
            
            # Update position prices
            self._update_position_prices(current_data)
            
            # Check risk breakers
            current_equity = self.broker.get_account_equity()
            positions_dict = {pos.symbol: {"qty": pos.quantity} for pos in self.current_positions.values()}
            
            breaker_status = self.risk_breaker.check_all_breakers(current_equity, positions_dict)
            
            # Process exits first
            self._process_exits(current_data)
            
            # Process new entries if risk allows
            if self.risk_breaker.can_enter_new_position():
                self._process_entries(current_data)
            
            # Record equity
            self.equity_history.append((current_date, current_equity))
            
            # Update position sizer with current equity
            self.position_sizer.account_equity = current_equity
    
    def _get_current_market_data(self, market_data: pd.DataFrame, date: datetime) -> Dict[str, pd.DataFrame]:
        """Get market data for current date.
        
        Args:
            market_data: Full market data
            date: Current date
            
        Returns:
            Dictionary of symbol -> recent data
        """
        # Get data up to current date for each symbol
        current_data = {}
        
        for strategy in self.strategies:
            if hasattr(strategy, 'universe'):
                for symbol in strategy.universe:
                    if symbol in market_data.columns.levels[0]:
                        symbol_data = market_data[symbol].loc[:date]
                        if not symbol_data.empty:
                            current_data[symbol] = symbol_data
        
        return current_data
    
    def _update_position_prices(self, current_data: Dict[str, pd.DataFrame]) -> None:
        """Update current prices for all positions.
        
        Args:
            current_data: Current market data
        """
        for position in self.current_positions.values():
            if position.symbol in current_data:
                symbol_data = current_data[position.symbol]
                if not symbol_data.empty:
                    position.current_price = symbol_data['close'].iloc[-1]
    
    def _process_exits(self, current_data: Dict[str, pd.DataFrame]) -> None:
        """Process exit signals for current positions.
        
        Args:
            current_data: Current market data
        """
        positions_to_close = []
        
        for position in self.current_positions.values():
            # Find the strategy that owns this position
            strategy = next((s for s in self.strategies if s.name == position.strategy_name), None)
            if not strategy:
                continue
            
            # Check if strategy wants to exit
            exit_signal = strategy.should_exit_position(position, current_data)
            if exit_signal:
                positions_to_close.append((position, exit_signal))
        
        # Execute exits
        for position, exit_signal in positions_to_close:
            self._execute_exit(position, exit_signal, current_data)
    
    def _process_entries(self, current_data: Dict[str, pd.DataFrame]) -> None:
        """Process entry signals from strategies.
        
        Args:
            current_data: Current market data
        """
        # Collect all signals from all strategies
        all_signals = []
        
        for strategy in self.strategies:
            if strategy.enabled:
                signals = strategy.generate_signals(current_data, self.current_positions)
                for signal in signals:
                    if strategy.validate_signal(signal, current_data):
                        all_signals.append((strategy, signal))
        
        # Sort signals by confidence (highest first)
        all_signals.sort(key=lambda x: x[1].confidence, reverse=True)
        
        # Execute signals respecting position limits
        for strategy, signal in all_signals:
            if self._can_enter_position(signal):
                self._execute_entry(strategy, signal, current_data)
    
    def _can_enter_position(self, signal: StrategySignal) -> bool:
        """Check if we can enter a new position.
        
        Args:
            signal: Strategy signal
            
        Returns:
            True if position can be entered
        """
        # Check position limits
        if len(self.current_positions) >= self.config.max_positions:
            return False
        
        # Check correlation limits
        positions_dict = {pos.symbol: {"qty": pos.quantity} for pos in self.current_positions.values()}
        can_add, reason = self.correlation_manager.can_add_position(signal.symbol, positions_dict)
        
        return can_add
    
    def _execute_entry(self, strategy: Strategy, signal: StrategySignal, 
                      current_data: Dict[str, pd.DataFrame]) -> None:
        """Execute entry signal.
        
        Args:
            strategy: Strategy generating signal
            signal: Entry signal
            current_data: Current market data
        """
        # Calculate position size
        current_equity = self.broker.get_account_equity()
        sizing_result = self.position_sizer.calculate_position_size(
            signal, 
            method=SizingMethod.PERCENT_EQUITY,
            current_positions=len(self.current_positions)
        )
        
        if not sizing_result['valid']:
            return
        
        quantity = sizing_result['quantity']
        
        # Get current price
        if signal.symbol not in current_data:
            return
        
        current_price = current_data[signal.symbol]['close'].iloc[-1]
        
        # Apply slippage
        if signal.action == 'buy':
            execution_price = current_price * (1 + self.config.slippage_equity_bps / 10000)
        else:
            execution_price = current_price * (1 - self.config.slippage_equity_bps / 10000)
        
        # Calculate commission
        commission = quantity * self.config.commission_equity
        
        # Execute trade with broker
        trade_id = f"T{len(self.trades) + 1:06d}"
        
        if self.broker.execute_trade(
            symbol=signal.symbol,
            quantity=quantity,
            price=execution_price,
            side=signal.action,
            commission=commission
        ):
            # Create position
            position = StrategyPosition(
                strategy_name=strategy.name,
                symbol=signal.symbol,
                side=PositionSide.LONG if signal.action == 'buy' else PositionSide.SHORT,
                quantity=quantity,
                entry_price=execution_price,
                current_price=execution_price,
                entry_time=self.broker.current_date,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata=signal.metadata
            )
            
            self.current_positions[signal.symbol] = position
            
            # Create trade record
            trade = BacktestTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                strategy=strategy.name,
                side=signal.action,
                quantity=quantity,
                entry_price=execution_price,
                entry_time=self.broker.current_date,
                commission=commission,
                slippage=execution_price - current_price,
                metadata=signal.metadata
            )
            
            self.trades.append(trade)
    
    def _execute_exit(self, position: StrategyPosition, exit_signal: StrategySignal,
                     current_data: Dict[str, pd.DataFrame]) -> None:
        """Execute exit signal.
        
        Args:
            position: Position to exit
            exit_signal: Exit signal
            current_data: Current market data
        """
        if position.symbol not in current_data:
            return
        
        current_price = current_data[position.symbol]['close'].iloc[-1]
        
        # Apply slippage
        if exit_signal.action == 'sell':
            execution_price = current_price * (1 - self.config.slippage_equity_bps / 10000)
        else:
            execution_price = current_price * (1 + self.config.slippage_equity_bps / 10000)
        
        # Calculate commission
        commission = position.quantity * self.config.commission_equity
        
        # Execute with broker
        if self.broker.execute_trade(
            symbol=position.symbol,
            quantity=position.quantity,
            price=execution_price,
            side=exit_signal.action,
            commission=commission
        ):
            # Calculate P&L
            if position.side == PositionSide.LONG:
                pnl = (execution_price - position.entry_price) * position.quantity - commission
            else:
                pnl = (position.entry_price - execution_price) * position.quantity - commission
            
            pnl_pct = pnl / (position.entry_price * position.quantity) * 100
            
            # Find and update trade record
            for trade in self.trades:
                if (trade.symbol == position.symbol and 
                    trade.exit_time is None):
                    
                    trade.exit_price = execution_price
                    trade.exit_time = self.broker.current_date
                    trade.commission += commission
                    trade.slippage += execution_price - current_price
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct
                    trade.exit_reason = exit_signal.metadata.get('exit_reason', 'signal')
                    break
            
            # Update risk breaker with trade result
            self.risk_breaker.update_trade_result(pnl)
            
            # Remove position
            del self.current_positions[position.symbol]
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results.
        
        Returns:
            Complete backtest results
        """
        results = BacktestResults()
        
        if not self.equity_history:
            return results
        
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        results.equity_curve = equity_df
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        results.daily_returns = returns
        
        # Basic metrics
        total_days = (equity_df.index[-1] - equity_df.index[0]).days
        results.duration_days = total_days
        results.start_date = equity_df.index[0]
        results.end_date = equity_df.index[-1]
        
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        
        results.total_return = (final_equity / initial_equity - 1) * 100
        results.annualized_return = ((final_equity / initial_equity) ** (365 / total_days) - 1) * 100
        
        # Risk metrics
        results.volatility = returns.std() * np.sqrt(252) * 100
        
        if results.volatility > 0:
            results.sharpe_ratio = (results.annualized_return - 2.0) / results.volatility  # Assume 2% risk-free rate
        
        # Maximum drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
        results.max_drawdown = drawdown.min()
        
        # Calmar ratio
        if results.max_drawdown < 0:
            results.calmar_ratio = results.annualized_return / abs(results.max_drawdown)
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252) * 100
            if downside_deviation > 0:
                results.sortino_ratio = (results.annualized_return - 2.0) / downside_deviation
        
        # VaR and Expected Shortfall
        if len(returns) > 0:
            results.var_95 = np.percentile(returns, 5) * 100
            es_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns <= es_threshold]
            if len(tail_returns) > 0:
                results.expected_shortfall = tail_returns.mean() * 100
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        results.trades = completed_trades
        results.total_trades = len(completed_trades)
        
        if completed_trades:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = len(winning_trades) / len(completed_trades) * 100
            
            if winning_trades:
                results.avg_win = np.mean([t.pnl for t in winning_trades])
            
            if losing_trades:
                results.avg_loss = np.mean([t.pnl for t in losing_trades])
                
                # Profit factor
                total_wins = sum(t.pnl for t in winning_trades)
                total_losses = abs(sum(t.pnl for t in losing_trades))
                if total_losses > 0:
                    results.profit_factor = total_wins / total_losses
        
        # Strategy breakdown
        strategy_results = {}
        for strategy in self.strategies:
            strategy_trades = [t for t in completed_trades if t.strategy == strategy.name]
            if strategy_trades:
                strategy_pnl = sum(t.pnl for t in strategy_trades)
                strategy_results[strategy.name] = {
                    'trades': len(strategy_trades),
                    'pnl': strategy_pnl,
                    'win_rate': len([t for t in strategy_trades if t.pnl > 0]) / len(strategy_trades) * 100
                }
        
        results.strategy_results = strategy_results
        
        return results