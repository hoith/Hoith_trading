#!/usr/bin/env python3
"""
Aggressive Strategy Backtest with IEX Data Feed
Uses the exact same parameters as live_aggressive_strategy.py (proven 122.59% returns)
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

class AggressiveStrategyBacktest:
    """Backtest using exact aggressive strategy parameters"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # EXACT AGGRESSIVE SYMBOLS - same as live strategy
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL',  # Individual stocks
            'QQQ', 'SPY',             # Standard ETFs  
            'TQQQ'                    # LEVERAGED ETF - biggest winner!
        ]
        
        # EXACT AGGRESSIVE POSITION SIZING (scaled for backtest capital)
        scale_factor = initial_capital / 100000.0  # Scale from $100k to our capital
        self.base_positions = {
            'AAPL': 200 * scale_factor,   # $200 base position scaled
            'MSFT': 200 * scale_factor,   # $200 base position scaled
            'GOOGL': 200 * scale_factor,  # $200 base position scaled
            'QQQ': 300 * scale_factor,    # $300 base position scaled
            'SPY': 300 * scale_factor,    # $300 base position scaled
            'TQQQ': 250 * scale_factor    # $250 base position scaled
        }
        
        # EXACT AGGRESSIVE RISK/REWARD PARAMETERS
        self.risk_params = {
            'AAPL': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            'MSFT': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            'GOOGL': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            'QQQ': {'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold_days': 12},
            'SPY': {'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold_days': 12},
            'TQQQ': {'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold_days': 8}
        }
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Trading state
        self.positions = {}
        self.trades = []
        self.current_time = None
        
        logger.info(f"Aggressive backtest initialized: {self.symbols}")
        logger.info(f"Base positions: {self.base_positions}")
        logger.info(f"Risk params: {self.risk_params}")
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator - same as live strategy"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def check_aggressive_entry_signal(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check entry signal using EXACT aggressive criteria"""
        try:
            if len(data) < 25:  # Need enough history
                return False
            
            # Calculate indicators - same as live strategy
            rsi = self.calculate_rsi(data['close'], 14)
            lookback = 10  # Proven lookback period
            
            if len(data) < lookback + 10:
                return False
            
            current_price = float(data['close'].iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            
            # EXACT AGGRESSIVE MOMENTUM CALCULATION
            momentum_score = (current_price / data['close'].iloc[-lookback] - 1) * 100
            
            # Volume analysis
            volume_ratio = float(data['volume'].iloc[-1] / data['volume'].iloc[-10:].mean())
            
            # EXACT ENTRY CONDITIONS BY SYMBOL TYPE
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Individual stocks
                # EXACT: Very relaxed criteria - catch more moves
                if (momentum_score > 1.0 and      # 1% momentum threshold
                    current_rsi < 75 and          # Higher RSI allowed
                    volume_ratio > 0.8):          # Below average volume OK
                    logger.info(f"AGGRESSIVE SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")
                    return True
                    
            elif symbol in ['QQQ', 'SPY']:  # Standard ETFs
                # EXACT: Relaxed ETF criteria
                if (momentum_score > 0.5 and      # Very low threshold
                    current_rsi < 80 and          # Very high RSI allowed
                    volume_ratio > 0.7):          # Low volume OK
                    logger.info(f"AGGRESSIVE SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")
                    return True
                    
            elif symbol == 'TQQQ':  # Leveraged ETF - THE MONEY MAKER!
                # EXACT: Most aggressive for highest returns
                if (momentum_score > 2.0 and      # 2% momentum (reduced from 5%)
                    current_rsi < 85):            # Very overbought allowed
                    logger.info(f"AGGRESSIVE SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking entry signal for {symbol}: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size using EXACT aggressive sizing"""
        # EXACT DYNAMIC SCALING
        base_position = self.base_positions[symbol]
        capital_multiplier = self.capital / self.initial_capital
        target_dollars = base_position * capital_multiplier
        
        # Calculate shares (fractional allowed)
        shares = target_dollars / price
        
        logger.debug(f"{symbol}: ${target_dollars:.0f} target = {shares:.2f} shares @ ${price:.2f}")
        
        return shares
    
    def enter_position(self, symbol: str, data: pd.DataFrame) -> bool:
        """Enter new position using aggressive parameters"""
        if symbol in self.positions:
            return False  # Already in position
        
        price = float(data['close'].iloc[-1])
        shares = self.calculate_position_size(symbol, price)
        
        if shares <= 0:
            return False
        
        # Get aggressive risk parameters for this symbol
        risk_params = self.risk_params[symbol]
        
        # Record position with aggressive parameters
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
        
        logger.info(f"ENTER: {symbol} {shares:.2f} shares @ ${price:.2f} = ${position['entry_value']:.2f}")
        logger.info(f"  Stop: ${position['stop_loss']:.2f} (-{risk_params['stop_pct']}%)")
        logger.info(f"  Target: ${position['take_profit']:.2f} (+{risk_params['target_pct']}%)")
        
        return True
    
    def check_exit_conditions(self, symbol: str, data: pd.DataFrame) -> str:
        """Check exit conditions using aggressive parameters"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        current_price = float(data['close'].iloc[-1])
        current_low = float(data['low'].iloc[-1])
        current_high = float(data['high'].iloc[-1])
        
        # EXACT EXIT CONDITIONS
        
        # Check stop loss (intraday low hit)
        if current_low <= position['stop_loss']:
            return 'stop_loss'
        
        # Check take profit (intraday high hit)  
        if current_high >= position['take_profit']:
            return 'take_profit'
        
        # Check time limit (aggressive max hold periods)
        time_held = (self.current_time - position['entry_time']).total_seconds() / 86400  # days
        if time_held >= position['max_hold_days']:
            return 'time_limit'
        
        return None
    
    def exit_position(self, symbol: str, data: pd.DataFrame, exit_reason: str) -> bool:
        """Exit position using aggressive parameters"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        exit_price = float(data['close'].iloc[-1])
        
        # Adjust exit price based on reason
        if exit_reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif exit_reason == 'take_profit':
            exit_price = position['take_profit']
        
        # Calculate P&L
        exit_value = position['shares'] * exit_price
        pnl = exit_value - position['entry_value']
        pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': self.current_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'entry_value': position['entry_value'],
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'hold_days': (self.current_time - position['entry_time']).total_seconds() / 86400
        }
        
        self.trades.append(trade)
        self.capital += exit_value
        
        logger.info(f"EXIT: {symbol} @ ${exit_price:.2f} - {exit_reason}")
        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) - Held {trade['hold_days']:.1f} days")
        
        # Remove from positions
        del self.positions[symbol]
        
        return True
    
    def get_portfolio_value(self, all_data: dict) -> float:
        """Calculate current portfolio value"""
        total_value = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in all_data:
                current_price = float(all_data[symbol]['close'].iloc[-1])
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value
    
    def run_backtest(self, start_date: str, end_date: str) -> dict:
        """Run aggressive strategy backtest"""
        logger.info("=" * 60)
        logger.info("AGGRESSIVE STRATEGY BACKTEST (Proven 122.59% Parameters)")
        logger.info("=" * 60)
        
        # Convert dates
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Get data for all symbols
        logger.info("Fetching historical data...")
        all_data = {}
        for symbol in self.symbols:
            try:
                bars = self.data_client.get_stock_bars(
                    symbols=[symbol],
                    timeframe=TimeFrame.Day,  # Use daily bars for longer backtest
                    start=start,
                    end=end
                )
                
                if not bars.empty:
                    if isinstance(bars.index, pd.MultiIndex):
                        df = bars.xs(symbol, level=0)[["open","high","low","close","volume"]]
                    else:
                        df = bars[["open","high","low","close","volume"]]
                    
                    all_data[symbol] = df.sort_index()
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        if not all_data:
            logger.error("No data loaded!")
            return {}
        
        # Get all dates
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        logger.info(f"Running backtest from {all_dates[0].date()} to {all_dates[-1].date()}")
        logger.info(f"Total trading days: {len(all_dates)}")
        
        # Track portfolio value over time
        portfolio_history = []
        
        # Run through each trading day
        for i, date in enumerate(all_dates):
            self.current_time = date
            
            # Get data for this date for all symbols
            current_data = {}
            for symbol, df in all_data.items():
                if date in df.index:
                    # Get data up to this date for signal calculation
                    hist_data = df.loc[:date].tail(30)  # Last 30 days for indicators
                    current_data[symbol] = hist_data
            
            # Process exits first
            exits = []
            for symbol in list(self.positions.keys()):
                if symbol in current_data:
                    exit_reason = self.check_exit_conditions(symbol, current_data[symbol])
                    if exit_reason:
                        self.exit_position(symbol, current_data[symbol], exit_reason)
                        exits.append(symbol)
            
            # Process entries
            entries = []
            for symbol in self.symbols:
                if symbol not in self.positions and symbol in current_data:
                    if len(current_data[symbol]) >= 25:  # Need enough history
                        if self.check_aggressive_entry_signal(symbol, current_data[symbol]):
                            if self.enter_position(symbol, current_data[symbol]):
                                entries.append(symbol)
            
            # Track portfolio value
            portfolio_value = self.get_portfolio_value(current_data)
            portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.capital,
                'positions': len(self.positions)
            })
            
            # Log progress every 50 days
            if i % 50 == 0 or entries or exits:
                logger.info(f"{date.date()}: Portfolio=${portfolio_value:,.2f}, Positions={len(self.positions)}, Entries={len(entries)}, Exits={len(exits)}")
        
        # Final portfolio value
        final_value = self.get_portfolio_value(current_data)
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Calculate performance metrics
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (sum([t['pnl'] for t in winning_trades]) / 
                        abs(sum([t['pnl'] for t in losing_trades]))) if losing_trades else float('inf')
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'portfolio_history': portfolio_history,
            'trades': self.trades
        }
        
        return results

def main():
    """Run aggressive strategy backtest"""
    
    # Check environment variables
    if not os.getenv('APCA_API_KEY_ID') or not os.getenv('APCA_API_SECRET_KEY'):
        logger.error("Missing APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
        return
    
    # Initialize backtest
    backtest = AggressiveStrategyBacktest(initial_capital=10000.0)
    
    # Run backtest over a longer period
    start_date = "2024-01-01"
    end_date = "2025-08-20"
    
    logger.info(f"Starting aggressive backtest from {start_date} to {end_date}")
    
    results = backtest.run_backtest(start_date, end_date)
    
    if results:
        # Print results
        print("\n" + "=" * 60)
        print("AGGRESSIVE STRATEGY BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:+.2f}%")
        print(f"")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"Average Win: {results['avg_win_pct']:+.2f}%")
        print(f"Average Loss: {results['avg_loss_pct']:+.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print("=" * 60)
        
        # Show recent trades
        if results['trades']:
            print("\nRecent Trades:")
            for trade in results['trades'][-10:]:
                print(f"{trade['symbol']}: {trade['exit_reason']} - "
                      f"{trade['pnl_pct']:+.2f}% in {trade['hold_days']:.1f} days")

if __name__ == "__main__":
    main()