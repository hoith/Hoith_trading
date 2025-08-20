#!/usr/bin/env python3
"""
Aggressive High Return Strategy - Targeting 10%+ returns
- Larger position sizes
- More aggressive entry criteria  
- Swing trading approach
- Higher leverage utilization
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

# Configure logging
logging.basicConfig(level=logging.WARNING)

def calculate_rsi(prices, window=14):
    """Simple RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fix_dataframe_index(df, symbol):
    """Fix MultiIndex to simple datetime index"""
    if isinstance(df.index, pd.MultiIndex):
        timestamps = [idx[1] for idx in df.index if idx[0] == symbol]
        df_clean = df.loc[df.index.get_level_values(0) == symbol].copy()
        df_clean.index = timestamps
        return df_clean
    return df

def aggressive_momentum_strategy(data, symbol, lookback=10):
    """Aggressive momentum strategy with relaxed criteria"""
    signals = []
    
    if len(data) < lookback + 10:
        return signals
    
    # Calculate indicators
    rsi = calculate_rsi(data['close'], 14)
    
    for i in range(lookback + 10, len(data)):
        current_date = data.index[i]
        current_price = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i]
        current_rsi = rsi.iloc[i]
        
        # Much more aggressive entry criteria
        momentum_score = (data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1) * 100
        volume_ratio = current_volume / data['volume'].iloc[i-10:i].mean()
        
        # Aggressive entry conditions based on symbol
        entry_signal = False
        
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Individual stocks
            # Very relaxed criteria - catch more moves
            if (momentum_score > 1.0 and  # Reduced from 3%
                current_rsi < 75 and      # Much higher RSI allowed
                volume_ratio > 0.8):      # Even below average volume OK
                entry_signal = True
                
        elif symbol in ['QQQ', 'SPY']:  # Standard ETFs
            # Relaxed ETF criteria
            if (momentum_score > 0.5 and  # Very low threshold
                current_rsi < 80 and      # Very high RSI allowed
                volume_ratio > 0.7):      # Low volume OK
                entry_signal = True
                
        elif symbol == 'TQQQ':  # Leveraged ETF
            # Much more aggressive for leverage
            if (momentum_score > 2.0 and  # Reduced from 5%
                current_rsi < 85):        # Very overbought allowed
                entry_signal = True
        
        if entry_signal:
            signals.append({
                'date': current_date,
                'action': 'buy',
                'price': current_price,
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio
            })
    
    return signals

def run_aggressive_strategy():
    """Run aggressive high-return strategy"""
    
    print("AGGRESSIVE HIGH-RETURN STRATEGY - 2025")
    print("Targeting 10%+ Returns")
    print("=" * 60)
    
    data_client = AlpacaDataClient()
    
    # Get 2025 data
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 8, 10)
    
    # Expanded universe including leveraged ETFs
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ']
    initial_capital = 1000.0
    
    all_trades = []
    capital_history = [initial_capital]
    current_capital = initial_capital
    
    for symbol in symbols:
        print(f"\n--- {symbol} Aggressive Analysis ---")
        
        try:
            # Get data
            df_raw = data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if df_raw.empty:
                print(f"No data available for {symbol}")
                continue
            
            # Fix MultiIndex
            df = fix_dataframe_index(df_raw, symbol)
            
            if len(df) < 20:
                print(f"Insufficient data: {len(df)} days")
                continue
            
            print(f"Data: {len(df)} trading days")
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # AGGRESSIVE POSITION SIZING
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                base_position = 200  # Doubled from 100
            elif symbol in ['QQQ', 'SPY']:
                base_position = 300  # Tripled from 100
            elif symbol == 'TQQQ':
                base_position = 250  # Much larger for leverage
            
            # Dynamic position sizing based on current capital
            position_multiplier = current_capital / initial_capital
            position_size = base_position * position_multiplier
            
            # Generate signals with relaxed criteria
            signals = aggressive_momentum_strategy(df, symbol, lookback=10)
            
            if not signals:
                print("No signals generated")
                continue
            
            print(f"Signals generated: {len(signals)}")
            
            # Simulate trades with aggressive parameters
            for signal in signals:
                entry_price = signal['price']
                entry_date = signal['date']
                shares = position_size / entry_price
                
                # AGGRESSIVE RISK/REWARD
                if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                    stop_pct = 3.0   # Wider stops
                    target_pct = 8.0  # Much higher targets
                    max_hold = 15    # Longer holds
                elif symbol in ['QQQ', 'SPY']:
                    stop_pct = 2.5
                    target_pct = 6.0
                    max_hold = 12
                elif symbol == 'TQQQ':
                    stop_pct = 2.0   # Still tight for leverage
                    target_pct = 5.0  # High target for leverage
                    max_hold = 8     # Shorter for leverage
                
                stop_loss = entry_price * (1 - stop_pct / 100)
                take_profit = entry_price * (1 + target_pct / 100)
                
                # Find exit
                entry_idx = df.index.get_loc(entry_date)
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
                    elif exit_idx >= max_hold - 1:
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
                
                # Update capital
                current_capital += pnl
                capital_history.append(current_capital)
                
                trade = {
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': shares,
                    'position_size': position_size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'momentum_score': signal['momentum_score'],
                    'capital_after': current_capital
                }
                
                all_trades.append(trade)
                
                print(f"  {entry_date.strftime('%m/%d')}: BUY {shares:.3f} @ ${entry_price:.2f} (${position_size:.0f})")
                print(f"    {exit_date.strftime('%m/%d')}: ${exit_price:.2f}, ${pnl:+.2f} ({pnl_pct:+.1f}%), {exit_reason}")
                print(f"    Capital: ${current_capital:.2f}")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # AGGRESSIVE STRATEGY RESULTS
    print(f"\n{'='*60}")
    print("AGGRESSIVE HIGH-RETURN STRATEGY RESULTS")
    print(f"{'='*60}")
    
    if all_trades:
        total_pnl = current_capital - initial_capital
        final_return = (current_capital / initial_capital - 1) * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Final Capital: ${current_capital:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"FINAL RETURN: {final_return:+.2f}%")
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
        
        # Largest wins and losses
        largest_win = max(all_trades, key=lambda x: x['pnl'])
        largest_loss = min(all_trades, key=lambda x: x['pnl'])
        
        print(f"\nLargest Win: {largest_win['symbol']} ${largest_win['pnl']:+.2f} ({largest_win['pnl_pct']:+.1f}%)")
        print(f"Largest Loss: {largest_loss['symbol']} ${largest_loss['pnl']:+.2f} ({largest_loss['pnl_pct']:+.1f}%)")
        
        # Monthly returns
        monthly_returns = []
        prev_capital = initial_capital
        for i, trade in enumerate(all_trades):
            if i % 5 == 4:  # Every 5 trades roughly
                month_return = (trade['capital_after'] / prev_capital - 1) * 100
                monthly_returns.append(month_return)
                prev_capital = trade['capital_after']
        
        if monthly_returns:
            avg_monthly = np.mean(monthly_returns)
            print(f"Avg Period Return: {avg_monthly:+.1f}%")
        
        # Compare to conservative strategy
        print(f"\nStrategy Comparison:")
        print(f"Conservative Strategy: +0.47% return")
        print(f"AGGRESSIVE Strategy: {final_return:+.2f}% return")
        improvement = final_return / 0.47 if 0.47 != 0 else float('inf')
        print(f"Improvement: {improvement:.1f}x better!")
        
        # Show capital growth
        print(f"\nCapital Growth:")
        milestones = [0, len(all_trades)//4, len(all_trades)//2, 3*len(all_trades)//4, len(all_trades)-1]
        for i in milestones:
            if i < len(all_trades):
                trade = all_trades[i]
                print(f"  After {i+1:2d} trades: ${trade['capital_after']:.2f}")
        
        # Risk metrics
        returns = [t['pnl_pct'] for t in all_trades]
        if len(returns) > 1:
            volatility = np.std(returns)
            avg_return = np.mean(returns)
            if volatility > 0:
                sharpe = avg_return / volatility
                print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Target achievement
        target_return = 10.0  # 10% target
        print(f"\nTarget Analysis:")
        print(f"Target Return: {target_return}%")
        print(f"Achieved Return: {final_return:.2f}%")
        if final_return >= target_return:
            print("🎯 TARGET ACHIEVED!")
        else:
            needed = target_return - final_return
            print(f"Need {needed:.1f}% more to hit target")
            
    else:
        print("No trades executed")

if __name__ == "__main__":
    run_aggressive_strategy()