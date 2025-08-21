#!/usr/bin/env python3
"""
RAPID GROWTH OPTIONS STRATEGY - Grow Small Account Quickly
Focus on high-probability, quick-profit options trades:

1. Short-term momentum plays with options (0-7 DTE)
2. Quick profit targets (20-50% gains in 1-3 days)
3. Defined risk with options spreads
4. High frequency trading for rapid compounding
5. Risk 2-5% per trade for 20-50% returns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RapidGrowthOptionsBacktest:
    """Rapid Growth Strategy - Quick options trades for fast account growth"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # High-volume leveraged ETFs - perfect for options with liquidity
        self.symbols = [
            'TQQQ',  # 3x Nasdaq ETF - highest volume options
            'SQQQ',  # 3x Inverse Nasdaq - for downside plays
            'SPXL',  # 3x S&P 500 ETF
            'SPXS',  # 3x Inverse S&P 500
            'QQQ',   # Standard tech ETF - massive options volume
            'SPY',   # S&P 500 ETF - most liquid options in the world
            'IWM'    # Russell 2000 ETF - good small cap exposure
        ]
        
        # Rapid growth parameters
        self.risk_per_trade_pct = 3.0    # Risk 3% per trade
        self.profit_target_pct = 25.0    # Target 25% profit quickly
        self.stop_loss_pct = 50.0        # Stop at 50% loss (options decay fast)
        self.max_hold_days = 3           # Quick in/out - hold max 3 days
        self.min_momentum_threshold = 2.0 # Minimum 2% momentum for entry
        
        # Options simulation parameters (high-volume ETF options)
        self.options_leverage = 8.0      # Higher leverage for liquid ETF options
        self.decay_per_day = 0.08        # 8% theta decay per day (liquid options hold value better)
        
        # ETF-specific parameters for better entries
        self.leveraged_etf_multiplier = {
            'TQQQ': 3, 'SQQQ': 3, 'SPXL': 3, 'SPXS': 3,  # 3x leveraged
            'QQQ': 1, 'SPY': 1, 'IWM': 1                   # 1x leveraged
        }
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Trading state
        self.positions = {}
        self.trades = []
        self.current_time = None
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # Reduce size after 3 losses
        
        logger.info("RAPID GROWTH OPTIONS strategy initialized")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"Risk per trade: {self.risk_per_trade_pct}%")
        logger.info(f"Profit target: {self.profit_target_pct}%")
        logger.info(f"Max hold: {self.max_hold_days} days")
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score for entry signals"""
        if len(data) < 10:
            return 0
        
        # Multiple timeframe momentum
        momentum_3 = (data['close'].iloc[-1] / data['close'].iloc[-4] - 1) * 100
        momentum_5 = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # RSI for overbought/oversold
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Combine factors
        momentum_score = (momentum_3 + momentum_5) / 2
        
        # Boost score for volume confirmation
        if volume_ratio > 1.5:
            momentum_score *= 1.2
        
        # RSI filter - avoid extremes
        if rsi > 80 or rsi < 20:
            momentum_score *= 0.5
        
        return momentum_score
    
    def check_entry_signal(self, symbol: str, data: pd.DataFrame) -> str:
        """Check for rapid momentum entry signals with ETF-specific logic"""
        if len(data) < 20:
            return None
        
        momentum = self.calculate_momentum_score(data)
        current_price = data['close'].iloc[-1]
        
        # Adjust threshold for leveraged ETFs (they move more)
        etf_multiplier = self.leveraged_etf_multiplier.get(symbol, 1)
        adjusted_threshold = self.min_momentum_threshold / max(etf_multiplier, 1)
        
        # Special logic for inverse ETFs (SQQQ, SPXS)
        if symbol in ['SQQQ', 'SPXS']:
            # For inverse ETFs, we want opposite signals
            if momentum < -adjusted_threshold:
                logger.info(f"LONG SIGNAL {symbol} (inverse): Market down {momentum:.1f}% @ ${current_price:.2f}")
                return 'long'  # Buy inverse ETF when market is falling
            elif momentum > adjusted_threshold:
                logger.info(f"SHORT SIGNAL {symbol} (inverse): Market up {momentum:.1f}% @ ${current_price:.2f}")
                return 'short'  # Short inverse ETF when market is rising
        else:
            # Regular ETFs
            if momentum > adjusted_threshold:
                logger.info(f"LONG SIGNAL {symbol}: Momentum {momentum:.1f}% @ ${current_price:.2f}")
                return 'long'
            elif momentum < -adjusted_threshold:
                logger.info(f"SHORT SIGNAL {symbol}: Momentum {momentum:.1f}% @ ${current_price:.2f}")
                return 'short'
        
        return None
    
    def calculate_position_size(self) -> float:
        """Calculate position size based on current capital and risk management"""
        # Base risk amount
        risk_amount = self.capital * (self.risk_per_trade_pct / 100)
        
        # Reduce size after consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            risk_amount *= 0.5  # Half size after 3+ losses
            logger.info(f"Reducing position size due to {self.consecutive_losses} consecutive losses")
        
        return risk_amount
    
    def enter_position(self, symbol: str, direction: str, data: pd.DataFrame) -> bool:
        """Enter rapid growth position (simulated options)"""
        if symbol in self.positions:
            return False  # Already in position
        
        if len(self.positions) >= 3:  # Max 3 concurrent positions
            return False
        
        entry_price = data['close'].iloc[-1]
        risk_amount = self.calculate_position_size()
        
        if risk_amount < 100:  # Minimum position size
            return False
        
        # Simulate options leverage
        notional_value = risk_amount * self.options_leverage
        
        # Calculate stops and targets based on underlying movement
        if direction == 'long':
            # For calls - profit from upward moves
            stop_price = entry_price * (1 - self.stop_loss_pct / 100 / self.options_leverage)
            target_price = entry_price * (1 + self.profit_target_pct / 100 / self.options_leverage)
        else:
            # For puts - profit from downward moves  
            stop_price = entry_price * (1 + self.stop_loss_pct / 100 / self.options_leverage)
            target_price = entry_price * (1 - self.profit_target_pct / 100 / self.options_leverage)
        
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': self.current_time,
            'risk_amount': risk_amount,
            'notional_value': notional_value,
            'stop_price': stop_price,
            'target_price': target_price,
            'days_held': 0
        }
        
        self.positions[symbol] = position
        
        logger.info(f"ENTER {direction.upper()} {symbol}: ${risk_amount:.0f} risk (${notional_value:.0f} notional) @ ${entry_price:.2f}")
        logger.info(f"  Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
        
        return True
    
    def check_exit_conditions(self, symbol: str, data: pd.DataFrame) -> str:
        """Check rapid exit conditions"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        
        direction = position['direction']
        
        # Time decay (options lose value daily)
        position['days_held'] = (self.current_time - position['entry_time']).days
        
        # Quick exit on profit target
        if direction == 'long':
            if current_high >= position['target_price']:
                return 'profit_target'
            if current_low <= position['stop_price']:
                return 'stop_loss'
        else:  # short
            if current_low <= position['target_price']:
                return 'profit_target'
            if current_high >= position['stop_price']:
                return 'stop_loss'
        
        # Time limit - options expire quickly
        if position['days_held'] >= self.max_hold_days:
            return 'time_limit'
        
        return None
    
    def exit_position(self, symbol: str, data: pd.DataFrame, exit_reason: str) -> None:
        """Exit position and calculate P&L"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Determine exit price
        if exit_reason == 'profit_target':
            exit_price = position['target_price']
        elif exit_reason == 'stop_loss':
            exit_price = position['stop_price']
        else:  # time_limit
            exit_price = data['close'].iloc[-1]
        
        # Calculate underlying move
        entry_price = position['entry_price']
        underlying_move = (exit_price - entry_price) / entry_price
        
        # Calculate options P&L with leverage and decay
        if position['direction'] == 'long':
            options_move = underlying_move * self.options_leverage
        else:  # short
            options_move = -underlying_move * self.options_leverage
        
        # Apply time decay
        decay_factor = 1 - (position['days_held'] * self.decay_per_day)
        options_move *= max(decay_factor, 0.1)  # Minimum 10% value retention
        
        # Calculate P&L
        pnl = position['risk_amount'] * options_move
        pnl_pct = options_move * 100
        
        # Update capital
        self.capital += position['risk_amount'] + pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': self.current_time,
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'risk_amount': position['risk_amount'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': position['days_held'],
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        logger.info(f"EXIT {position['direction'].upper()} {symbol}: ${exit_price:.2f} - {exit_reason}")
        logger.info(f"  P&L: ${pnl:+.0f} ({pnl_pct:+.1f}%) in {position['days_held']} days")
        logger.info(f"  New capital: ${self.capital:,.0f}")
        
        # Remove position
        del self.positions[symbol]
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data for symbol"""
        try:
            bars = self.data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Filter for this symbol and reset index
            symbol_data = bars.loc[symbol].reset_index()
            symbol_data = symbol_data.set_index('timestamp')
            
            logger.info(f"Loaded {len(symbol_data)} bars for {symbol}")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> None:
        """Run the rapid growth backtest"""
        
        logger.info("============================================================")
        logger.info("RAPID GROWTH OPTIONS STRATEGY BACKTEST")
        logger.info("============================================================")
        logger.info("Fetching historical data...")
        
        # Get historical data for all symbols
        data = {}
        for symbol in self.symbols:
            symbol_data = self.get_historical_data(symbol, start_date, end_date)
            if not symbol_data.empty:
                data[symbol] = symbol_data
        
        if not data:
            logger.error("No data loaded!")
            return
        
        # Get trading dates from the first symbol with data
        first_symbol = list(data.keys())[0]
        trading_dates = data[first_symbol].index[20:]  # Skip first 20 days for indicators
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            
            # Check exit conditions first
            for symbol in list(self.positions.keys()):
                if symbol in data and current_date in data[symbol].index:
                    exit_reason = self.check_exit_conditions(symbol, data[symbol].loc[:current_date])
                    if exit_reason:
                        self.exit_position(symbol, data[symbol].loc[:current_date], exit_reason)
            
            # Look for new entry opportunities
            if len(self.positions) < 3:  # Max concurrent positions
                for symbol in self.symbols:
                    if symbol not in self.positions and symbol in data and current_date in data[symbol].index:
                        historical_data = data[symbol].loc[:current_date]
                        
                        signal = self.check_entry_signal(symbol, historical_data)
                        if signal:
                            if self.enter_position(symbol, signal, historical_data):
                                break  # Only one entry per day
            
            # Log progress
            if i % 30 == 0 or len(self.trades) != (0 if not hasattr(self, '_last_trade_count') else self._last_trade_count):
                total_return = (self.capital / self.initial_capital - 1) * 100
                logger.info(f"{current_date.date()}: Capital=${self.capital:,.0f} ({total_return:+.1f}%), Positions={len(self.positions)}, Trades={len(self.trades)}")
                self._last_trade_count = len(self.trades)
        
        # Close any remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in data:
                last_data = data[symbol].iloc[-1:]
                self.exit_position(symbol, last_data, 'backtest_end')
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print rapid growth backtest results"""
        if not self.trades:
            logger.info("No trades completed!")
            return
        
        final_value = self.capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        avg_hold_time = trades_df['days_held'].mean()
        
        # Calculate max drawdown
        running_capital = [self.initial_capital]
        for trade in self.trades:
            running_capital.append(running_capital[-1] + trade['pnl'])
        
        peak = self.initial_capital
        max_drawdown = 0
        for capital in running_capital:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        logger.info("============================================================")
        logger.info("RAPID GROWTH OPTIONS STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:+.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        logger.info("")
        logger.info(f"Total Trades: {len(trades_df)}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Win: {avg_win:+.1f}%")
        logger.info(f"Average Loss: {avg_loss:+.1f}%")
        logger.info(f"Average Hold Time: {avg_hold_time:.1f} days")
        logger.info("============================================================")
        
        # Calculate annualized return
        if self.trades:
            days_elapsed = (self.trades[-1]['exit_time'] - self.trades[0]['entry_time']).days
            years = max(days_elapsed / 365.25, 0.1)
            annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100
            logger.info(f"Annualized Return: {annualized_return:.1f}%")
        
        # Show recent trades
        logger.info("\nRecent Trades:")
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        for trade in recent_trades:
            logger.info(f"{trade['symbol']} {trade['direction'].upper()}: {trade['pnl_pct']:+.1f}% in {trade['days_held']}d ({trade['exit_reason']})")

if __name__ == "__main__":
    # Run the rapid growth backtest
    backtest = RapidGrowthOptionsBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)