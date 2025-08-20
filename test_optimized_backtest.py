#!/usr/bin/env python3
"""
Test optimized strategy parameters with a simple backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

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

def run_optimized_backtest():
    """Run backtest with optimized parameters"""
    
    print("Running Optimized Strategy Backtest")
    print("=" * 50)
    
    # Load cached data
    try:
        df = pd.read_pickle('data/cache/AAPL-GOOGL-MSFT_20230101_20230630_1d.pkl')
        print(f"Loaded cached data: {df.shape[0]} days, {len(df.columns.levels[0])} symbols")
    except:
        print("No cached data found. Please run the main backtest first to download data.")
        return
    
    # Test parameters
    initial_capital = 1000.0
    position_size = 50  # $50 per position (optimized)
    
    results = {}
    
    # Test each symbol
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        if symbol not in df.columns.levels[0]:
            continue
            
        print(f"\n--- Testing {symbol} ---")
        symbol_data = df[symbol].dropna()
        
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
                continue
                
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
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    
    all_trades = []
    for symbol_results in results.values():
        all_trades.extend(symbol_results['trades'])
    
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        final_return = total_pnl / initial_capital * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
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
    
    print(f"\nOptimized parameters used:")
    print(f"  - Lookback period: 15 days (reduced from 20)")
    print(f"  - Volume threshold: 300K (reduced from 500K)")
    print(f"  - Volume spike: 1.1x (reduced from 1.2x)")
    print(f"  - Position size: $50 (increased from $30)")
    print(f"  - ATR stop: 0.8x (tighter from 1.0x)")
    print(f"  - ATR target: 1.6x (reduced from 2.0x)")

if __name__ == "__main__":
    run_optimized_backtest()