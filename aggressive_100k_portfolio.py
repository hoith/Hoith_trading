#!/usr/bin/env python3
"""
Aggressive High Return Strategy - $100,000 Portfolio Test
Scaling up the 122% return strategy to institutional capital levels
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
        
        # Aggressive entry criteria
        momentum_score = (data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1) * 100
        volume_ratio = current_volume / data['volume'].iloc[i-10:i].mean()
        
        # Same relaxed entry conditions as before
        entry_signal = False
        
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            if (momentum_score > 1.0 and current_rsi < 75 and volume_ratio > 0.8):
                entry_signal = True
                
        elif symbol in ['QQQ', 'SPY']:
            if (momentum_score > 0.5 and current_rsi < 80 and volume_ratio > 0.7):
                entry_signal = True
                
        elif symbol == 'TQQQ':
            if (momentum_score > 2.0 and current_rsi < 85):
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

def run_100k_aggressive_strategy():
    """Run aggressive strategy with $100,000 starting capital"""
    
    print("AGGRESSIVE STRATEGY - $100,000 PORTFOLIO TEST")
    print("Scaling 122% Return Strategy to Institutional Capital")
    print("=" * 70)
    
    data_client = AlpacaDataClient()
    
    # Get 2025 data
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 8, 10)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ']
    initial_capital = 100000.0  # $100K starting capital
    
    all_trades = []
    capital_history = [initial_capital]
    current_capital = initial_capital
    
    # Track monthly performance
    monthly_snapshots = []
    
    for symbol in symbols:
        print(f"\n--- {symbol} ($100K Portfolio Analysis) ---")
        
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
            
            # SCALED POSITION SIZING FOR $100K PORTFOLIO
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                base_position = 20000  # $20K per individual stock position
            elif symbol in ['QQQ', 'SPY']:
                base_position = 30000  # $30K per standard ETF position
            elif symbol == 'TQQQ':
                base_position = 25000  # $25K per leveraged ETF position
            
            # Dynamic position sizing based on current capital
            position_multiplier = current_capital / initial_capital
            position_size = base_position * position_multiplier
            
            # Cap individual positions at 40% of portfolio for risk management
            max_position = current_capital * 0.40
            position_size = min(position_size, max_position)
            
            # Generate signals
            signals = aggressive_momentum_strategy(df, symbol, lookback=10)
            
            if not signals:
                print("No signals generated")
                continue
            
            print(f"Signals generated: {len(signals)}")
            
            # Simulate trades
            for signal in signals:
                entry_price = signal['price']
                entry_date = signal['date']
                shares = position_size / entry_price
                
                # Same aggressive risk/reward parameters
                if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                    stop_pct = 3.0
                    target_pct = 8.0
                    max_hold = 15
                elif symbol in ['QQQ', 'SPY']:
                    stop_pct = 2.5
                    target_pct = 6.0
                    max_hold = 12
                elif symbol == 'TQQQ':
                    stop_pct = 2.0
                    target_pct = 5.0
                    max_hold = 8
                
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
                
                # Show significant trades only (> $1000 P&L)
                if abs(pnl) > 1000:
                    print(f"  {entry_date.strftime('%m/%d')}: BUY {shares:.0f} @ ${entry_price:.2f} (${position_size:,.0f})")
                    print(f"    {exit_date.strftime('%m/%d')}: ${exit_price:.2f}, ${pnl:+,.0f} ({pnl_pct:+.1f}%), {exit_reason}")
                    print(f"    Capital: ${current_capital:,.0f}")
                
                # Record monthly snapshots
                if len(all_trades) % 20 == 0:  # Every 20 trades roughly = monthly
                    monthly_snapshots.append({
                        'trades': len(all_trades),
                        'capital': current_capital,
                        'return': (current_capital / initial_capital - 1) * 100
                    })
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # $100K PORTFOLIO RESULTS
    print(f"\n{'='*70}")
    print("$100,000 AGGRESSIVE PORTFOLIO RESULTS")
    print(f"{'='*70}")
    
    if all_trades:
        total_pnl = current_capital - initial_capital
        final_return = (current_capital / initial_capital - 1) * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"PORTFOLIO PERFORMANCE:")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${current_capital:,.2f}")
        print(f"Total P&L: ${total_pnl:+,.2f}")
        print(f"FINAL RETURN: {final_return:+.2f}%")
        print(f"")
        print(f"TRADE STATISTICS:")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Win Rate: {len(win_trades)/len(all_trades)*100:.1f}%")
        print(f"Average Trade P&L: ${total_pnl/len(all_trades):+,.2f}")
        
        if win_trades:
            avg_win = np.mean([t['pnl'] for t in win_trades])
            lose_trades = [t for t in all_trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0
            
            print(f"Avg Win: ${avg_win:+,.2f}")
            print(f"Avg Loss: ${avg_loss:+,.2f}")
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in win_trades)
            total_losses = abs(sum(t['pnl'] for t in lose_trades)) if lose_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")
        
        # Show largest wins and losses
        largest_wins = sorted([t for t in all_trades if t['pnl'] > 0], key=lambda x: x['pnl'], reverse=True)[:5]
        largest_losses = sorted([t for t in all_trades if t['pnl'] < 0], key=lambda x: x['pnl'])[:3]
        
        print(f"\nTOP 5 WINNING TRADES:")
        for i, trade in enumerate(largest_wins, 1):
            print(f"  {i}. {trade['symbol']}: ${trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%)")
        
        print(f"\nLARGEST LOSSES:")
        for i, trade in enumerate(largest_losses, 1):
            print(f"  {i}. {trade['symbol']}: ${trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%)")
        
        # Monthly progression
        print(f"\nMONTHLY PORTFOLIO GROWTH:")
        for snapshot in monthly_snapshots:
            print(f"  After {snapshot['trades']:3d} trades: ${snapshot['capital']:,.0f} ({snapshot['return']:+.1f}%)")
        
        # Compare to benchmarks
        print(f"\nBENCHMARK COMPARISON:")
        print(f"S&P 500 (2025 YTD): ~+15% estimated")
        print(f"Portfolio Return: {final_return:+.2f}%")
        if final_return > 15:
            outperformance = final_return - 15
            print(f"OUTPERFORMANCE: +{outperformance:.1f}% vs S&P 500")
        
        # Potential annual returns
        months_elapsed = 7.3  # Jan 1 to Aug 10
        annualized_return = (final_return / months_elapsed) * 12
        print(f"Annualized Return Potential: {annualized_return:+.1f}%")
        
        # Dollar value achievements
        print(f"\nWEALTH GENERATION:")
        wealth_created = total_pnl
        print(f"Wealth Created: ${wealth_created:+,.0f}")
        
        if wealth_created > 50000:
            print(f"TARGET ACHIEVED: Generated >$50K profit!")
        
        # Risk metrics for large portfolio
        daily_returns = []
        prev_capital = initial_capital
        for trade in all_trades:
            daily_return = (trade['capital_after'] / prev_capital - 1) * 100
            daily_returns.append(daily_return)
            prev_capital = trade['capital_after']
        
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns)
            avg_return = np.mean(daily_returns)
            if volatility > 0:
                sharpe = avg_return / volatility
                print(f"Sharpe Ratio: {sharpe:.2f}")
            
            max_drawdown = 0
            peak = initial_capital
            for capital in capital_history:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            print(f"Maximum Drawdown: -{max_drawdown:.1f}%")
        
        # Position sizing analysis
        avg_position_size = np.mean([t['position_size'] for t in all_trades])
        max_position_size = max([t['position_size'] for t in all_trades])
        print(f"\nPOSITION SIZING:")
        print(f"Average Position: ${avg_position_size:,.0f}")
        print(f"Largest Position: ${max_position_size:,.0f}")
        print(f"Portfolio Utilization: {max_position_size/current_capital*100:.1f}%")
        
    else:
        print("No trades executed")

if __name__ == "__main__":
    run_100k_aggressive_strategy()