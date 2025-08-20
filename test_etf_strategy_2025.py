#!/usr/bin/env python3
"""
Test the new ETF Momentum Strategy with QQQ, SPY, TQQQ for 2025
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

def etf_momentum_strategy(data, symbol, lookback=20, min_volume=1000000):
    """ETF-specific momentum strategy"""
    signals = []
    
    if len(data) < max(lookback, 20):
        return signals
    
    # Calculate indicators
    rsi = calculate_rsi(data['close'], 14)
    volatility = data['close'].pct_change().rolling(14).std() * np.sqrt(252) * 100
    
    for i in range(lookback + 20, len(data)):
        current_date = data.index[i]
        current_price = data['close'].iloc[i]
        current_volume = data['volume'].iloc[i]
        current_rsi = rsi.iloc[i]
        current_vol = volatility.iloc[i]
        
        # Basic volume check
        avg_volume = data['volume'].iloc[i-20:i].mean()
        if avg_volume < min_volume:
            continue
        
        # Calculate momentum
        momentum_score = (data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1) * 100
        volume_ratio = current_volume / avg_volume
        price_change_5d = (data['close'].iloc[i] / data['close'].iloc[i-5] - 1) * 100
        
        # ETF-specific entry conditions
        entry_signal = False
        signal_type = None
        
        if symbol == 'SPY':  # Broad market - conservative
            if (momentum_score > 2.0 and 
                current_rsi < 60 and 
                volume_ratio > 1.2 and 
                price_change_5d > 1.0):
                entry_signal = True
                signal_type = 'broad_market_momentum'
                
        elif symbol == 'QQQ':  # Tech focused - moderate
            if (momentum_score > 3.0 and 
                current_rsi < 65 and 
                volume_ratio > 1.1):
                entry_signal = True
                signal_type = 'tech_momentum'
                
        elif symbol == 'TQQQ':  # Leveraged - strict requirements
            if (momentum_score > 5.0 and 
                current_rsi < 70 and 
                volume_ratio > 1.3 and 
                current_vol < 50):  # Not too volatile
                entry_signal = True
                signal_type = 'leveraged_momentum'
        
        if entry_signal:
            signals.append({
                'date': current_date,
                'action': 'buy',
                'price': current_price,
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'volume_ratio': volume_ratio,
                'volatility': current_vol,
                'signal_type': signal_type,
                'price_change_5d': price_change_5d
            })
    
    return signals

def run_etf_strategy_test():
    """Test ETF momentum strategy with 2025 data"""
    
    print("ETF MOMENTUM STRATEGY TEST - 2025")
    print("Testing QQQ, SPY, TQQQ")
    print("=" * 50)
    
    data_client = AlpacaDataClient()
    
    # Get 2025 data
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 8, 10)
    
    symbols = ['QQQ', 'SPY', 'TQQQ']
    initial_capital = 1000.0
    
    all_trades = []
    symbol_results = {}
    
    for symbol in symbols:
        print(f"\n--- {symbol} ETF Analysis ---")
        
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
            
            if len(df) < 40:
                print(f"Insufficient data: {len(df)} days")
                continue
            
            print(f"Data: {len(df)} trading days")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # Calculate ETF type position sizing
            if symbol == 'SPY':
                position_size = 80  # Conservative for broad market
                stop_pct = 1.5
                target_pct = 3.0
            elif symbol == 'QQQ':
                position_size = 100  # Standard for tech
                stop_pct = 2.0
                target_pct = 4.0
            elif symbol == 'TQQQ':
                position_size = 60  # Smaller for leveraged
                stop_pct = 1.0
                target_pct = 2.5
            
            # Generate signals
            signals = etf_momentum_strategy(df, symbol, lookback=20, min_volume=1000000)
            
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
                
                # Calculate stops
                stop_loss = entry_price * (1 - stop_pct / 100)
                take_profit = entry_price * (1 + target_pct / 100)
                
                # Find exit
                entry_idx = df.index.get_loc(entry_date)
                future_data = df.iloc[entry_idx+1:]
                exit_date = None
                exit_price = None
                exit_reason = 'time'
                
                max_hold_days = 10 if symbol != 'TQQQ' else 5  # Shorter holds for leveraged
                
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
                    elif exit_idx >= max_hold_days - 1:
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
                    'signal_type': signal['signal_type'],
                    'momentum_score': signal['momentum_score'],
                    'rsi': signal['rsi']
                }
                
                trades.append(trade)
                all_trades.append(trade)
                
                print(f"  {entry_date.strftime('%m/%d')}: {signal['signal_type']} - BUY {shares:.3f} @ ${entry_price:.2f}")
                print(f"    RSI: {signal['rsi']:.1f}, Momentum: {signal['momentum_score']:.1f}%")
                print(f"    {exit_date.strftime('%m/%d')}: ${exit_price:.2f}, ${pnl:+.2f} ({pnl_pct:+.1f}%), {exit_reason}")
            
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_trades = [t for t in trades if t['pnl'] > 0]
                
                print(f"\n{symbol} ETF Results:")
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
    
    # Overall ETF strategy results
    print(f"\n{'='*50}")
    print("ETF MOMENTUM STRATEGY RESULTS")
    print(f"{'='*50}")
    
    if all_trades:
        total_pnl = sum(t['pnl'] for t in all_trades)
        final_return = total_pnl / initial_capital * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"Test Period: Jan 1 - Aug 10, 2025")
        print(f"ETFs Tested: QQQ, SPY, TQQQ")
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
        
        # Strategy breakdown by ETF type
        print(f"\nBreakdown by ETF Type:")
        for symbol in symbols:
            if symbol in symbol_results:
                result = symbol_results[symbol]
                etf_type = {'SPY': 'Broad Market', 'QQQ': 'Tech Focused', 'TQQQ': 'Leveraged 3x'}[symbol]
                print(f"  {symbol} ({etf_type}): {result['trades']} trades, "
                      f"{result['wins']} wins, ${result['total_pnl']:+.2f} P&L")
        
        # Compare to individual stock strategy
        print(f"\nStrategy Comparison:")
        print(f"Individual Stocks (AAPL/MSFT/GOOGL): 32 trades, +0.43% return")
        print(f"ETF Momentum (QQQ/SPY/TQQQ): {len(all_trades)} trades, {final_return:+.2f}% return")
        
        if final_return > 0.43:
            print(f"ETF strategy OUTPERFORMED individual stocks!")
        else:
            print(f"Individual stocks performed better in this period")
            
        # Show signal type breakdown
        signal_types = {}
        for trade in all_trades:
            sig_type = trade['signal_type']
            if sig_type not in signal_types:
                signal_types[sig_type] = {'count': 0, 'pnl': 0}
            signal_types[sig_type]['count'] += 1
            signal_types[sig_type]['pnl'] += trade['pnl']
        
        print(f"\nSignal Type Performance:")
        for sig_type, data in signal_types.items():
            avg_pnl = data['pnl'] / data['count']
            print(f"  {sig_type}: {data['count']} trades, ${avg_pnl:+.2f} avg P&L")
            
    else:
        print("No ETF trades executed in 2025 period")

if __name__ == "__main__":
    run_etf_strategy_test()