#!/usr/bin/env python3
"""
Fixed 2025 Backtest - Resolves MultiIndex date formatting issues
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
from alpaca.data.timeframe import TimeFrame

# Configure logging to suppress info messages
logging.basicConfig(level=logging.WARNING)

def calculate_atr_simple(high, low, close, window=10):
    """Simple ATR calculation"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def fix_dataframe_index(df, symbol):
    """Fix MultiIndex to simple datetime index"""
    if isinstance(df.index, pd.MultiIndex):
        # Extract just the timestamp part from (symbol, timestamp) tuples
        timestamps = [idx[1] for idx in df.index if idx[0] == symbol]
        df_clean = df.loc[df.index.get_level_values(0) == symbol].copy()
        df_clean.index = timestamps
        return df_clean
    return df

def simple_momentum_strategy(data, lookback=15, volume_threshold=300000):
    """Generate signals with optimized parameters"""
    signals = []
    
    # Ensure we have clean datetime index
    if len(data) < lookback + 20:
        return signals
    
    for i in range(lookback + 20, len(data)):
        current_date = data.index[i]
        current_price = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i]
        
        if current_volume < volume_threshold:
            continue
            
        # Calculate momentum indicators
        price_change = data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1
        volume_ratio = current_volume / data['volume'].iloc[i-20:i].mean()
        
        # Entry conditions (optimized thresholds)
        momentum_threshold = 0.03  # 3% price increase
        volume_spike = 1.1  # 10% above average volume
        
        if price_change > momentum_threshold and volume_ratio > volume_spike:
            signals.append({
                'date': current_date,
                'action': 'buy',
                'price': current_price,
                'momentum': price_change,
                'volume_ratio': volume_ratio
            })
    
    return signals

def run_fixed_2025_backtest():
    """Run complete 2025 backtest with fixed date handling"""
    
    print("FIXED 2025 STRATEGY BACKTEST")
    print("Period: January 1, 2025 - August 10, 2025")
    print("=" * 60)
    
    data_client = AlpacaDataClient()
    
    # Get full 2025 data
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 8, 10)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    initial_capital = 1000.0
    position_size = 50
    
    all_trades = []
    symbol_results = {}
    
    for symbol in symbols:
        print(f"\n--- {symbol} Analysis ---")
        
        try:
            # Get data
            df_raw = data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if df_raw.empty:
                print(f"No data available")
                continue
            
            # Fix the MultiIndex issue
            df = fix_dataframe_index(df_raw, symbol)
            
            if len(df) < 30:
                print(f"Insufficient data: {len(df)} days")
                continue
            
            print(f"Data: {len(df)} trading days")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # Generate signals
            signals = simple_momentum_strategy(df, lookback=15, volume_threshold=300000)
            
            if not signals:
                print("No signals generated")
                continue
            
            print(f"Signals generated: {len(signals)}")
            
            # Simulate trades
            trades = []
            
            for signal in signals:
                entry_price = signal['price']
                entry_date = signal['date']
                shares = position_size / entry_price
                
                # Calculate ATR for stops
                entry_idx = df.index.get_loc(entry_date)
                symbol_subset = df.iloc[:entry_idx+1]
                
                if len(symbol_subset) < 10:
                    continue
                
                atr = calculate_atr_simple(
                    symbol_subset['high'], 
                    symbol_subset['low'], 
                    symbol_subset['close'], 
                    window=10
                ).iloc[-1]
                
                if pd.isna(atr) or atr <= 0:
                    atr = entry_price * 0.02  # 2% fallback
                
                # Risk management levels
                stop_loss = entry_price - (atr * 0.8)
                take_profit = entry_price + (atr * 1.6)
                
                # Find exit
                future_data = df.iloc[entry_idx+1:]
                exit_date = None
                exit_price = None
                exit_reason = 'time'
                
                for exit_idx, (date, row) in enumerate(future_data.iterrows()):
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
                    elif exit_idx >= 4:  # Max 5 days (0-indexed)
                        exit_price = row['close']
                        exit_date = date
                        exit_reason = 'time_limit'
                        break
                
                if exit_price is None:
                    exit_price = df['close'].iloc[-1]
                    exit_date = df.index[-1]
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
                all_trades.append(trade)
                
                print(f"  {entry_date.strftime('%m/%d')}: BUY {shares:.3f} @ ${entry_price:.2f}")
                print(f"    {exit_date.strftime('%m/%d')}: ${exit_price:.2f}, ${pnl:+.2f} ({pnl_pct:+.1f}%), {exit_reason}")
            
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_trades = [t for t in trades if t['pnl'] > 0]
                
                print(f"\n{symbol} Results:")
                print(f"  Trades: {len(trades)}")
                print(f"  Winners: {len(win_trades)} ({len(win_trades)/len(trades)*100:.0f}%)")
                print(f"  Total P&L: ${total_pnl:+.2f}")
                
                if win_trades:
                    avg_win = np.mean([t['pnl'] for t in win_trades])
                    print(f"  Avg win: ${avg_win:.2f}")
                
                lose_trades = [t for t in trades if t['pnl'] <= 0]
                if lose_trades:
                    avg_loss = np.mean([t['pnl'] for t in lose_trades])
                    print(f"  Avg loss: ${avg_loss:.2f}")
                
                symbol_results[symbol] = {
                    'trades': len(trades),
                    'wins': len(win_trades),
                    'total_pnl': total_pnl
                }
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n{'='*60}")
    print("FIXED 2025 OVERALL RESULTS")
    print(f"{'='*60}")
    
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        final_return = total_pnl / initial_capital * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"Test Period: Jan 1 - Aug 10, 2025")
        print(f"Trading Days: {(end_date - start_date).days} calendar days")
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Final Return: {final_return:+.2f}%")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Win Rate: {len(win_trades)/len(all_trades)*100:.1f}%")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            lose_trades = [t for t in all_trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0
            
            print(f"Avg Win: ${avg_win:+.2f}")
            print(f"Avg Loss: ${avg_loss:+.2f}")
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in win_trades)
            total_losses = abs(sum(t['pnl'] for t in lose_trades)) if lose_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")
        
        # Risk metrics
        returns = [t['pnl_pct'] for t in all_trades]
        if len(returns) > 1:
            volatility = np.std(returns)
            avg_return = np.mean(returns)
            if volatility > 0:
                sharpe = avg_return / volatility
                print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Show detailed trade log
        print(f"\nDetailed Trade Log:")
        for i, trade in enumerate(all_trades, 1):
            print(f"{i:2d}. {trade['symbol']} {trade['entry_date'].strftime('%m/%d')} -> {trade['exit_date'].strftime('%m/%d')}: "
                  f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f}, "
                  f"${trade['pnl']:+.2f} ({trade['pnl_pct']:+.1f}%), {trade['exit_reason']}")
        
        # Compare periods
        print(f"\nComparison to 2023 Results:")
        print(f"2023 (Jan-Jun): 53 trades, +1.13% return, 49.1% win rate")
        print(f"2025 (Jan-Aug): {len(all_trades)} trades, {final_return:+.2f}% return, {len(win_trades)/len(all_trades)*100:.1f}% win rate")
        
        if final_return > 0:
            print(f"\n✅ Strategy PROFITABLE in 2025!")
        else:
            print(f"\n❌ Strategy underperformed in 2025")
            
    else:
        print("No trades executed in 2025 period")
        print("This indicates:")
        print("- Market conditions don't meet current strategy criteria")
        print("- May need parameter adjustment for 2025 market regime")

if __name__ == "__main__":
    run_fixed_2025_backtest()