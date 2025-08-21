#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE MOMENTUM Strategy - Target 50%+ Annual Returns
Revolutionary approach:
1. MASSIVE position sizes (use ALL available capital per trade)
2. Focus ONLY on TQQQ (3x leveraged) for maximum returns
3. Perfect momentum timing with confirmation
4. Very tight stops but huge targets
5. Hold until massive moves complete
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

class UltraAggressiveMomentumBacktest:
    """ULTRA AGGRESSIVE - ALL IN momentum strategy for 50%+ returns"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # FOCUS: ONLY TQQQ for maximum leverage and momentum
        self.symbols = ['TQQQ']  # 3x leveraged tech ETF - pure momentum play
        
        # ULTRA AGGRESSIVE: Use 90% of capital per trade
        self.position_pct = 0.90  # Use 90% of available capital
        
        # REVOLUTIONARY RISK/REWARD - Optimized for big moves
        self.stop_pct = 8.0     # Wider stops to avoid whipsaws
        self.target_pct = 25.0  # MASSIVE profit targets
        self.max_hold_days = 30 # Hold for full momentum cycles
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Trading state
        self.position = None
        self.trades = []
        self.current_time = None
        
        logger.info("ULTRA AGGRESSIVE MOMENTUM strategy initialized")
        logger.info(f"Symbol: {self.symbols[0]} (3x leveraged)")
        logger.info(f"Position size: {self.position_pct*100}% of capital")
        logger.info(f"Risk/Reward: -{self.stop_pct}% / +{self.target_pct}%")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and trend indicators"""
        df = data.copy()
        
        # Multiple timeframe momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Volume analysis
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volatility
        df['atr'] = ((df['high'] - df['low']).rolling(14).mean())
        
        return df
    
    def check_ultra_momentum_signal(self, data: pd.DataFrame) -> str:
        """Check for ULTRA strong momentum signals - very selective"""
        try:
            if len(data) < 60:  # Need substantial history
                return None
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # ULTRA SELECTIVE momentum criteria
            momentum_5 = latest['momentum_5'] * 100
            momentum_10 = latest['momentum_10'] * 100  
            momentum_20 = latest['momentum_20'] * 100
            rsi = latest['rsi']
            volume_ratio = latest['volume_ratio']
            
            # Price vs moving averages
            above_sma10 = latest['close'] > latest['sma_10']
            above_sma20 = latest['close'] > latest['sma_20']
            above_sma50 = latest['close'] > latest['sma_50']
            
            # Trend acceleration (SMAs in proper order)
            sma_trend = (latest['sma_10'] > latest['sma_20'] > latest['sma_50'])
            
            # ULTRA AGGRESSIVE LONG SIGNAL
            if (momentum_5 > 3.0 and           # Strong 5-day momentum
                momentum_10 > 5.0 and         # Stronger 10-day momentum  
                momentum_20 > 8.0 and         # Very strong 20-day momentum
                rsi > 50 and rsi < 85 and     # Bullish but not overbought
                volume_ratio > 1.2 and        # Above average volume
                above_sma10 and above_sma20 and above_sma50 and  # Above all MAs
                sma_trend):                   # Trend accelerating
                
                logger.info(f"ULTRA LONG SIGNAL: Mom5={momentum_5:.1f}%, Mom10={momentum_10:.1f}%, Mom20={momentum_20:.1f}%")
                logger.info(f"  RSI={rsi:.1f}, Volume={volume_ratio:.1f}x, Price=${latest['close']:.2f}")
                return 'long'
            
            # ULTRA AGGRESSIVE SHORT SIGNAL
            elif (momentum_5 < -3.0 and        # Strong 5-day decline
                  momentum_10 < -5.0 and       # Stronger 10-day decline
                  momentum_20 < -8.0 and       # Very strong 20-day decline
                  rsi < 50 and rsi > 15 and    # Bearish but not oversold
                  volume_ratio > 1.2 and       # Above average volume
                  not above_sma10 and not above_sma20 and not above_sma50):  # Below all MAs
                
                logger.info(f"ULTRA SHORT SIGNAL: Mom5={momentum_5:.1f}%, Mom10={momentum_10:.1f}%, Mom20={momentum_20:.1f}%")
                logger.info(f"  RSI={rsi:.1f}, Volume={volume_ratio:.1f}x, Price=${latest['close']:.2f}")
                return 'short'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking momentum signal: {e}")
            return None
    
    def calculate_position_size(self, price: float, direction: str) -> float:
        """Calculate ULTRA AGGRESSIVE position size - use nearly all capital"""
        target_value = self.capital * self.position_pct
        shares = target_value / price
        
        if direction == 'short':
            shares = -shares  # Negative for short positions
        
        logger.info(f"ULTRA POSITION: ${target_value:,.0f} ({self.position_pct*100}%) = {abs(shares):.0f} shares @ ${price:.2f}")
        return shares
    
    def enter_position(self, data: pd.DataFrame, direction: str) -> bool:
        """Enter ULTRA AGGRESSIVE position"""
        if self.position is not None:
            return False  # Already in position
        
        price = float(data['close'].iloc[-1])
        shares = self.calculate_position_size(price, direction)
        
        if abs(shares) < 1:  # Need at least 1 share
            return False
        
        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = price * (1 - self.stop_pct / 100)
            take_profit = price * (1 + self.target_pct / 100)
        else:  # short
            stop_loss = price * (1 + self.stop_pct / 100)  
            take_profit = price * (1 - self.target_pct / 100)
        
        # Record position
        self.position = {
            'direction': direction,
            'shares': shares,
            'entry_price': price,
            'entry_time': self.current_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_value': abs(shares) * price
        }
        
        # Update capital (reserve amount used)
        self.capital -= self.position['entry_value']
        
        logger.info(f"ULTRA {direction.upper()} ENTRY: {abs(shares):.0f} shares @ ${price:.2f}")
        logger.info(f"  Stop: ${stop_loss:.2f} ({-self.stop_pct if direction=='long' else +self.stop_pct}%)")
        logger.info(f"  Target: ${take_profit:.2f} ({+self.target_pct if direction=='long' else -self.target_pct}%)")
        logger.info(f"  Capital used: ${self.position['entry_value']:,.0f}")
        
        return True
    
    def check_exit_conditions(self, data: pd.DataFrame) -> str:
        """Check exit conditions for ULTRA position"""
        if self.position is None:
            return None
        
        current_price = float(data['close'].iloc[-1])
        current_low = float(data['low'].iloc[-1])
        current_high = float(data['high'].iloc[-1])
        
        direction = self.position['direction']
        
        # Check stop loss and take profit
        if direction == 'long':
            # Long position checks
            if current_low <= self.position['stop_loss']:
                return 'stop_loss'
            if current_high >= self.position['take_profit']:
                return 'take_profit'
        else:
            # Short position checks
            if current_high >= self.position['stop_loss']:
                return 'stop_loss'
            if current_low <= self.position['take_profit']:
                return 'take_profit'
        
        # Check time limit
        days_held = (self.current_time - self.position['entry_time']).days
        if days_held >= self.max_hold_days:
            return 'time_limit'
        
        return None
    
    def exit_position(self, data: pd.DataFrame, exit_reason: str) -> None:
        """Exit ULTRA position and record trade"""
        if self.position is None:
            return
        
        # Determine exit price
        if exit_reason == 'stop_loss':
            exit_price = self.position['stop_loss']
        elif exit_reason == 'take_profit':
            exit_price = self.position['take_profit']
        else:  # time_limit
            exit_price = float(data['close'].iloc[-1])
        
        # Calculate P&L
        shares = self.position['shares']
        entry_price = self.position['entry_price']
        
        if self.position['direction'] == 'long':
            pnl = shares * (exit_price - entry_price)
        else:  # short
            pnl = -shares * (exit_price - entry_price)  # shares is negative for short
        
        pnl_pct = (pnl / self.position['entry_value']) * 100
        
        # Days held
        days_held = (self.current_time - self.position['entry_time']).days
        
        # Record trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': self.current_time,
            'direction': self.position['direction'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': abs(shares),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        # Update capital
        exit_value = abs(shares) * exit_price
        self.capital += exit_value + pnl  # Return capital plus P&L
        
        logger.info(f"ULTRA {self.position['direction'].upper()} EXIT: ${exit_price:.2f} - {exit_reason}")
        logger.info(f"  P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%) - Held {days_held} days")
        logger.info(f"  New capital: ${self.capital:,.0f}")
        
        # Clear position
        self.position = None
    
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
        """Run the ULTRA AGGRESSIVE momentum backtest"""
        
        logger.info("============================================================")
        logger.info("ULTRA AGGRESSIVE MOMENTUM STRATEGY BACKTEST")
        logger.info("============================================================")
        logger.info("Fetching historical data...")
        
        # Get historical data for TQQQ
        symbol = self.symbols[0]
        data = self.get_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            logger.error("No data loaded!")
            return
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Get trading dates (skip first 60 days for indicator calculation)
        trading_dates = data.index[60:]
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        trade_count = 0
        max_consecutive_losses = 0
        consecutive_losses = 0
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            
            # Get data up to current date for analysis
            historical_data = data.loc[:current_date]
            
            # Check exit conditions first
            if self.position is not None:
                exit_reason = self.check_exit_conditions(historical_data)
                if exit_reason:
                    self.exit_position(historical_data, exit_reason)
                    trade_count += 1
                    
                    # Track consecutive losses
                    if self.trades[-1]['pnl'] < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
            
            # Check for new entry (only if not in position)
            if self.position is None:
                signal = self.check_ultra_momentum_signal(historical_data)
                if signal:
                    self.enter_position(historical_data, signal)
            
            # Log progress every 30 days or on trades
            if i % 30 == 0 or trade_count != len(self.trades):
                portfolio_value = self.capital
                if self.position is not None:
                    current_price = data.loc[current_date, 'close']
                    if self.position['direction'] == 'long':
                        portfolio_value += self.position['shares'] * current_price
                    else:
                        portfolio_value += self.position['entry_value'] - abs(self.position['shares']) * (current_price - self.position['entry_price'])
                
                return_pct = (portfolio_value / self.initial_capital - 1) * 100
                pos_status = f"In {self.position['direction']}" if self.position else "Cash"
                
                logger.info(f"{current_date.date()}: ${portfolio_value:,.0f} ({return_pct:+.1f}%) | {pos_status} | Trades: {len(self.trades)}")
        
        # Close any remaining position
        if self.position is not None:
            last_data = data.iloc[-1:]
            self.exit_position(last_data, 'backtest_end')
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print ULTRA AGGRESSIVE backtest results"""
        if not self.trades:
            logger.info("No trades completed!")
            return
        
        # Calculate final portfolio value
        final_value = self.capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.01
        profit_factor = gross_profit / gross_loss
        
        # Calculate max drawdown
        running_pnl = trades_df['pnl'].cumsum()
        running_high = running_pnl.cummax()
        drawdown = running_pnl - running_high
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        logger.info("============================================================")
        logger.info("ULTRA AGGRESSIVE MOMENTUM STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:+.1f}%")
        logger.info(f"Max Drawdown: ${max_drawdown:,.0f}")
        logger.info("")
        logger.info(f"Total Trades: {len(trades_df)}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Win: {avg_win:+.1f}%")
        logger.info(f"Average Loss: {avg_loss:+.1f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Risk/Reward Ratio: 1:{self.target_pct/self.stop_pct:.1f}")
        logger.info("============================================================")
        
        # Show all trades
        logger.info("\nAll Trades:")
        for i, trade in enumerate(self.trades, 1):
            logger.info(f"{i:2d}. {trade['direction'].upper()} {trade['entry_time'].date()} -> {trade['exit_time'].date()}: "
                       f"{trade['pnl_pct']:+.1f}% ({trade['exit_reason']}) [{trade['days_held']}d]")

if __name__ == "__main__":
    # Run the ULTRA AGGRESSIVE backtest
    backtest = UltraAggressiveMomentumBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)