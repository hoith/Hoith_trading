#!/usr/bin/env python3
"""
Test optimized strategy parameters for 2025 period (Jan 1 - Aug 10, 2025)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.alpaca_client import AlpacaDataClient
from data.historical import HistoricalDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_momentum_strategy(data, lookback=15, volume_threshold=300000):
    """Simple momentum breakout strategy with optimized parameters"""
    signals = []
    
    for i in range(lookback, len(data)):
        current_date = data.index[i]
        current_price = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i]
        
        # Calculate momentum indicators
        price_change = data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1
        volume_ratio = current_volume / data['volume'].iloc[i-20:i].mean()
        
        # Entry conditions (optimized thresholds)
        momentum_threshold = 0.03  # 3% price increase over lookback period
        volume_spike = 1.1  # 10% above average volume (reduced from 20%)
        
        if (price_change > momentum_threshold and 
            volume_ratio > volume_spike and 
            current_volume > volume_threshold):
            
            signals.append({
                'date': current_date,
                'action': 'buy',
                'price': current_price,
                'momentum': price_change,
                'volume_ratio': volume_ratio
            })
    
    return signals

def calculate_atr_simple(high, low, close, window=10):
    """Simple ATR calculation"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def download_2025_data():
    """Download fresh data for 2025 period"""
    print("Downloading 2025 market data...")
    
    try:
        # Initialize data client
        data_client = AlpacaDataClient()
        fetcher = HistoricalDataFetcher(data_client)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 8, 10)
        
        # Download data
        combined_data = {}
        
        for symbol in symbols:
            print(f"  Downloading {symbol}...")
            try:
                symbol_data = fetcher.get_historical_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1d'
                )
                
                if not symbol_data.empty:
                    # Convert to expected format
                    symbol_data = symbol_data.rename(columns={
                        'o': 'open',
                        'h': 'high', 
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                    combined_data[symbol] = symbol_data
                    print(f"    Downloaded {len(symbol_data)} days for {symbol}")
                else:
                    print(f"    No data available for {symbol}")
                    
            except Exception as e:
                print(f"    Error downloading {symbol}: {e}")
        
        if combined_data:
            # Create multi-index DataFrame
            df_list = []
            for symbol, data in combined_data.items():
                data_copy = data.copy()
                data_copy.columns = pd.MultiIndex.from_product([[symbol], data_copy.columns])
                df_list.append(data_copy)
            
            if df_list:
                final_df = pd.concat(df_list, axis=1)
                
                # Save to cache
                cache_file = f"data/cache/AAPL-GOOGL-MSFT_20250101_20250810_1d.pkl"
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                final_df.to_pickle(cache_file)
                print(f"Saved data to {cache_file}")
                
                return final_df
        
        print("No data downloaded successfully")
        return None
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def run_2025_backtest():
    """Run backtest for 2025 period"""
    
    print("Running 2025 Strategy Backtest (Jan 1 - Aug 10, 2025)")
    print("=" * 60)
    
    # Try to load cached data first
    cache_file = "data/cache/AAPL-GOOGL-MSFT_20250101_20250810_1d.pkl"
    
    try:
        if os.path.exists(cache_file):
            df = pd.read_pickle(cache_file)
            print(f"Loaded cached 2025 data: {df.shape[0]} days")
        else:
            print("No cached 2025 data found. Downloading fresh data...")
            df = download_2025_data()
            if df is None:
                print("Failed to download 2025 data. Cannot proceed.")
                return
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to download fresh data...")
        df = download_2025_data()
        if df is None:
            print("Failed to download 2025 data. Cannot proceed.")
            return
    
    if df.empty:
        print("No data available for analysis")
        return
        
    print(f"Testing period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Available symbols: {list(df.columns.levels[0])}")
    
    # Test parameters
    initial_capital = 1000.0
    position_size = 50  # $50 per position (optimized)
    
    results = {}
    
    # Test each symbol
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        if symbol not in df.columns.levels[0]:
            print(f"\nSymbol {symbol} not available in dataset")
            continue
            
        print(f"\n--- Testing {symbol} ---")
        symbol_data = df[symbol].dropna()
        
        if len(symbol_data) < 30:
            print(f"Insufficient data for {symbol} ({len(symbol_data)} days)")
            continue
        
        # Generate signals with optimized parameters
        signals = simple_momentum_strategy(
            symbol_data, 
            lookback=15,  # Reduced from 20
            volume_threshold=300000  # Reduced from 500K
        )
        
        if not signals:
            print(f"No signals generated for {symbol}")
            continue
            
        # Simulate trades
        trades = []
        capital = initial_capital
        positions = []
        
        for signal in signals:
            entry_price = signal['price']
            entry_date = signal['date']
            
            # Calculate position size and ATR-based stops
            shares = position_size / entry_price
            
            # Simple ATR calculation for stops
            symbol_subset = symbol_data.loc[:entry_date]
            if len(symbol_subset) < 10:
                continue
                
            atr = calculate_atr_simple(
                symbol_subset['high'], 
                symbol_subset['low'], 
                symbol_subset['close'], 
                window=10
            ).iloc[-1]
            
            # Optimized risk management
            stop_loss = entry_price - (atr * 0.8)  # Tighter stops
            take_profit = entry_price + (atr * 1.6)  # Lower targets
            
            # Find exit
            future_data = symbol_data.loc[entry_date:].iloc[1:]  # Skip entry day
            exit_date = None
            exit_price = None
            exit_reason = 'time'
            
            for exit_idx, (date, row) in enumerate(future_data.iterrows()):
                # Check stops
                if row['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_date = date
                    exit_reason = 'stop_loss'
                    break
                elif row['high'] >= take_profit:
                    exit_price = take_profit
                    exit_date = date
                    exit_reason = 'take_profit' 
                    break
                elif exit_idx >= 5:  # Max 5 days hold
                    exit_price = row['close']
                    exit_date = date
                    exit_reason = 'time_limit'
                    break
            
            if exit_price is None:
                # Position still open at end of test period
                exit_price = symbol_data['close'].iloc[-1]
                exit_date = symbol_data.index[-1]
                exit_reason = 'end_of_period'
                
            # Calculate P&L
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            trade = {
                'symbol': symbol,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'momentum': signal['momentum'],
                'volume_ratio': signal['volume_ratio']
            }
            
            trades.append(trade)
            capital += pnl
            
            print(f"  {entry_date.strftime('%Y-%m-%d')}: {signal['action'].upper()} {shares:.3f} @ ${entry_price:.2f}")
            print(f"    Exit {exit_date.strftime('%Y-%m-%d')}: ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.1f}%), {exit_reason}")
        
        if trades:
            # Calculate statistics
            total_pnl = sum(t['pnl'] for t in trades)
            win_trades = [t for t in trades if t['pnl'] > 0]
            lose_trades = [t for t in trades if t['pnl'] <= 0]
            
            print(f"\n  {symbol} Summary:")
            print(f"    Total trades: {len(trades)}")
            print(f"    Winners: {len(win_trades)} ({len(win_trades)/len(trades)*100:.1f}%)")
            print(f"    Total P&L: ${total_pnl:.2f}")
            print(f"    Avg P&L per trade: ${total_pnl/len(trades):.2f}")
            
            if win_trades:
                avg_win = np.mean([t['pnl'] for t in win_trades])
                print(f"    Avg win: ${avg_win:.2f}")
            if lose_trades:
                avg_loss = np.mean([t['pnl'] for t in lose_trades])
                print(f"    Avg loss: ${avg_loss:.2f}")
        
        results[symbol] = {
            'trades': trades,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100
        }
    
    # Overall results
    print(f"\n{'='*60}")
    print("2025 OVERALL RESULTS")
    print(f"{'='*60}")
    
    all_trades = []
    for symbol_results in results.values():
        all_trades.extend(symbol_results['trades'])
    
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        final_return = total_pnl / initial_capital * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"Test Period: Jan 1, 2025 - Aug 10, 2025")
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Final Return: {final_return:.2f}%")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Win Rate: {len(win_trades)/len(all_trades)*100:.1f}%")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            avg_loss = np.mean([t['pnl'] for t in all_trades if t['pnl'] <= 0])
            print(f"Avg Win: ${avg_win:.2f}")
            print(f"Avg Loss: ${avg_loss:.2f}")
            
        # Risk metrics
        daily_returns = pd.Series([t['pnl_pct'] for t in all_trades])
        if len(daily_returns) > 1:
            volatility = daily_returns.std()
            if volatility > 0:
                sharpe = daily_returns.mean() / volatility
                print(f"Sharpe Ratio: {sharpe:.2f}")
                
        # Compare to 2023 results
        print(f"\nComparison to 2023 Results:")
        print(f"2023: 53 trades, 1.13% return, 49.1% win rate")
        print(f"2025: {len(all_trades)} trades, {final_return:.2f}% return, {len(win_trades)/len(all_trades)*100:.1f}% win rate")
    else:
        print("No trades generated for 2025 period")
        print("This could indicate:")
        print("- Market conditions don't match strategy criteria")
        print("- Need to adjust parameters for current market regime")
        print("- Insufficient data or data quality issues")

if __name__ == "__main__":
    run_2025_backtest()