#!/usr/bin/env python3
"""
HIGH RETURN AGGRESSIVE Strategy - Optimized for 20%+ Annual Returns
Key changes:
1. Much larger position sizes (3-5x larger)
2. Wider stops (reduce whipsaws)  
3. Higher profit targets
4. Better entry filters (reduce losing trades)
5. Focus on strongest momentum only
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

class HighReturnAggressiveBacktest:
    """Optimized backtest for HIGH RETURNS (20%+ target)"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # OPTIMIZED SYMBOLS - Focus on best performers
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL',  # Tech giants with strong momentum
            'QQQ',                    # Tech ETF - reliable mover
            'TQQQ'                    # 3x leveraged - biggest winner potential
        ]
        
        # MUCH LARGER POSITION SIZING for higher returns
        scale_factor = initial_capital / 100000.0  
        self.base_positions = {
            'AAPL': 800 * scale_factor,   # 4x larger positions
            'MSFT': 800 * scale_factor,   # 4x larger positions  
            'GOOGL': 800 * scale_factor,  # 4x larger positions
            'QQQ': 1000 * scale_factor,   # 3.3x larger positions
            'TQQQ': 1200 * scale_factor   # 4.8x larger - focus on leverage!
        }
        
        # OPTIMIZED RISK/REWARD for higher returns
        self.risk_params = {
            'AAPL': {'stop_pct': 5.0, 'target_pct': 12.0, 'max_hold_days': 20},   # Wider stops, bigger targets
            'MSFT': {'stop_pct': 5.0, 'target_pct': 12.0, 'max_hold_days': 20},   # Wider stops, bigger targets
            'GOOGL': {'stop_pct': 5.0, 'target_pct': 12.0, 'max_hold_days': 20},  # Wider stops, bigger targets
            'QQQ': {'stop_pct': 4.0, 'target_pct': 10.0, 'max_hold_days': 15},    # ETF still conservative
            'TQQQ': {'stop_pct': 6.0, 'target_pct': 15.0, 'max_hold_days': 10}    # Leveraged = big moves
        }
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Trading state
        self.positions = {}
        self.trades = []
        self.current_time = None
        
        logger.info(f"HIGH RETURN backtest initialized: {self.symbols}")
        logger.info(f"Large positions: {self.base_positions}")
        logger.info(f"Optimized risk params: {self.risk_params}")
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def check_high_return_entry_signal(self, symbol: str, data: pd.DataFrame) -> bool:
        """Optimized entry signal for HIGH RETURNS - much more selective"""
        try:
            if len(data) < 30:  # Need more history for better signals
                return False
            
            # Calculate indicators
            rsi = self.calculate_rsi(data['close'], 14)
            lookback = 20  # Longer lookback for stronger signals
            
            if len(data) < lookback + 15:
                return False
            
            current_price = float(data['close'].iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            
            # MUCH MORE SELECTIVE MOMENTUM CALCULATION
            momentum_score = (current_price / data['close'].iloc[-lookback] - 1) * 100
            
            # Volume analysis - require strong volume
            volume_ratio = float(data['volume'].iloc[-1] / data['volume'].iloc[-20:].mean())
            
            # Price trend strength
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            price_above_sma = (current_price / sma_20 - 1) * 100
            
            # MUCH HIGHER ENTRY THRESHOLDS for better quality trades
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Individual stocks
                # MUCH MORE SELECTIVE: Only strongest momentum
                if (momentum_score > 4.0 and          # 4x higher momentum threshold
                    current_rsi > 45 and current_rsi < 70 and  # RSI sweet spot
                    volume_ratio > 1.5 and           # Strong volume required
                    price_above_sma > 2.0):          # Must be well above trend
                    logger.info(f"HIGH RETURN SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x, vs SMA: {price_above_sma:.1f}%")
                    return True
                    
            elif symbol == 'QQQ':  # Standard ETF
                # More selective QQQ criteria
                if (momentum_score > 2.0 and          # Higher threshold
                    current_rsi > 40 and current_rsi < 75 and
                    volume_ratio > 1.2 and           # Good volume
                    price_above_sma > 1.0):          # Above trend
                    logger.info(f"HIGH RETURN SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")
                    return True
                    
            elif symbol == 'TQQQ':  # Leveraged ETF - THE MONEY MAKER!
                # Focus on strongest TQQQ moves only
                if (momentum_score > 6.0 and          # Much higher threshold for leverage
                    current_rsi > 35 and current_rsi < 80 and
                    volume_ratio > 1.0):             # Normal volume OK for TQQQ
                    logger.info(f"HIGH RETURN SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking entry signal for {symbol}: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate LARGE position sizes for higher returns"""
        # LARGE DYNAMIC SCALING
        base_position = self.base_positions[symbol]
        capital_multiplier = self.capital / self.initial_capital
        target_dollars = base_position * capital_multiplier
        
        # Calculate shares (fractional allowed)
        shares = target_dollars / price
        
        logger.debug(f"{symbol}: ${target_dollars:.0f} target = {shares:.3f} shares @ ${price:.2f}")
        
        return shares
    
    def enter_position(self, symbol: str, data: pd.DataFrame) -> bool:
        """Enter new position with optimized parameters"""
        if symbol in self.positions:
            return False  # Already in position
        
        price = float(data['close'].iloc[-1])
        shares = self.calculate_position_size(symbol, price)
        
        if shares <= 0:
            return False
        
        # Get optimized risk parameters for this symbol
        risk_params = self.risk_params[symbol]
        
        # Record position with optimized parameters
        position = {
            'symbol': symbol,
            'shares': shares,
            'entry_price': price,
            'entry_time': self.current_time,
            'stop_loss': price * (1 - risk_params['stop_pct'] / 100),
            'take_profit': price * (1 + risk_params['target_pct'] / 100),
            'max_hold_days': risk_params['max_hold_days'],
            'entry_value': shares * price
        }
        
        self.positions[symbol] = position
        self.capital -= position['entry_value']
        
        logger.info(f"LARGE ENTRY: {symbol} {shares:.3f} shares @ ${price:.2f} = ${position['entry_value']:.2f}")
        logger.info(f"  Stop: ${position['stop_loss']:.2f} (-{risk_params['stop_pct']}%)")
        logger.info(f"  Target: ${position['take_profit']:.2f} (+{risk_params['target_pct']}%)")
        
        return True
    
    def check_exit_conditions(self, symbol: str, data: pd.DataFrame) -> str:
        """Check exit conditions with optimized parameters"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        current_price = float(data['close'].iloc[-1])
        current_low = float(data['low'].iloc[-1])
        current_high = float(data['high'].iloc[-1])
        
        # OPTIMIZED EXIT CONDITIONS
        
        # Check stop loss (intraday low hit)
        if current_low <= position['stop_loss']:
            return 'stop_loss'
        
        # Check take profit (intraday high hit)
        if current_high >= position['take_profit']:
            return 'take_profit'
        
        # Check time limit
        days_held = (self.current_time - position['entry_time']).days
        if days_held >= position['max_hold_days']:
            return 'time_limit'
        
        return None
    
    def exit_position(self, symbol: str, data: pd.DataFrame, exit_reason: str) -> None:
        """Exit position and record trade"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Determine exit price based on reason
        if exit_reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif exit_reason == 'take_profit':
            exit_price = position['take_profit']
        else:  # time_limit
            exit_price = float(data['close'].iloc[-1])
        
        # Calculate P&L
        exit_value = position['shares'] * exit_price
        pnl = exit_value - position['entry_value']
        pnl_pct = (pnl / position['entry_value']) * 100
        
        # Days held
        days_held = (self.current_time - position['entry_time']).days
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': self.current_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        # Update capital
        self.capital += exit_value
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"HIGH RETURN EXIT: {symbol} @ ${exit_price:.2f} - {exit_reason}")
        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) - Held {days_held} days")
    
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
        """Run the high return optimized backtest"""
        
        logger.info("============================================================")
        logger.info("HIGH RETURN AGGRESSIVE STRATEGY BACKTEST")
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
        trading_dates = data[first_symbol].index[1:]  # Skip first day
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        daily_stats = []
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            
            entries_today = 0
            exits_today = 0
            
            # Check exit conditions first
            for symbol in list(self.positions.keys()):
                if symbol in data and current_date in data[symbol].index:
                    exit_reason = self.check_exit_conditions(symbol, data[symbol].loc[:current_date])
                    if exit_reason:
                        self.exit_position(symbol, data[symbol].loc[:current_date], exit_reason)
                        exits_today += 1
            
            # Check for new entries (only if we have room)
            max_positions = 3  # Limit concurrent positions for risk management
            if len(self.positions) < max_positions:
                for symbol in self.symbols:
                    if symbol not in self.positions and symbol in data and current_date in data[symbol].index:
                        # Get data up to current date for signal calculation
                        historical_data = data[symbol].loc[:current_date]
                        
                        if self.check_high_return_entry_signal(symbol, historical_data):
                            if self.enter_position(symbol, historical_data):
                                entries_today += 1
                                # Limit to one entry per day
                                break
            
            # Calculate current portfolio value
            portfolio_value = self.capital
            for symbol, position in self.positions.items():
                if symbol in data and current_date in data[symbol].index:
                    current_price = data[symbol].loc[current_date, 'close']
                    portfolio_value += position['shares'] * current_price
            
            # Log progress
            if i % 50 == 0 or entries_today > 0 or exits_today > 0:
                return_pct = (portfolio_value / self.initial_capital - 1) * 100
                logger.info(f"{current_date.date()}: Portfolio=${portfolio_value:,.2f} ({return_pct:+.2f}%), Positions={len(self.positions)}, Entries={entries_today}, Exits={exits_today}")
            
            daily_stats.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'positions': len(self.positions),
                'entries': entries_today,
                'exits': exits_today
            })
        
        # Close any remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in data:
                last_data = data[symbol].iloc[-1:]
                self.exit_position(symbol, last_data, 'backtest_end')
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print detailed backtest results"""
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
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.01
        profit_factor = gross_profit / gross_loss
        
        logger.info("============================================================")
        logger.info("HIGH RETURN AGGRESSIVE STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info("")
        logger.info(f"Total Trades: {len(trades_df)}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Win: {avg_win:+.2f}%")
        logger.info(f"Average Loss: {avg_loss:+.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info("============================================================")
        
        # Show recent trades
        logger.info("\nRecent Trades:")
        for trade in self.trades[-10:]:
            logger.info(f"{trade['symbol']}: {trade['exit_reason']} - {trade['pnl_pct']:+.2f}% in {trade['days_held']} days")

if __name__ == "__main__":
    # Run the high return backtest
    backtest = HighReturnAggressiveBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)