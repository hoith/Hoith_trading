#!/usr/bin/env python3
"""
Detailed Trade Log Generator for Expanded Aggressive Strategy
Creates comprehensive CSV and text logs with every trade detail
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
import csv

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

def expanded_momentum_strategy(data, symbol, lookback=10):
    """Expanded momentum strategy with symbol-specific criteria"""
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
        
        # Momentum and volume analysis
        momentum_score = (data['close'].iloc[i] / data['close'].iloc[i-lookback] - 1) * 100
        volume_ratio = current_volume / data['volume'].iloc[i-10:i].mean()
        
        # Symbol-specific entry criteria
        entry_signal = False
        
        # Large cap individual stocks (conservative)
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'CRM']:
            if (momentum_score > 1.0 and current_rsi < 75 and volume_ratio > 0.8):
                entry_signal = True
        
        # High-volatility growth stocks (more aggressive)        
        elif symbol in ['TSLA', 'NVDA', 'NFLX', 'AMD', 'UBER']:
            if (momentum_score > 1.5 and current_rsi < 70 and volume_ratio > 0.9):
                entry_signal = True
                
        # Standard ETFs (moderate)
        elif symbol in ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF']:
            if (momentum_score > 0.5 and current_rsi < 80 and volume_ratio > 0.7):
                entry_signal = True
        
        # Innovation/Growth ETFs (balanced)
        elif symbol in ['ARKK']:
            if (momentum_score > 1.0 and current_rsi < 75 and volume_ratio > 0.8):
                entry_signal = True
                
        # Leveraged ETFs (strict criteria due to high volatility)
        elif symbol in ['TQQQ', 'SOXL', 'SPXL']:
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

def get_position_config(symbol):
    """Get position configuration for each symbol"""
    configs = {
        # Original proven symbols
        'AAPL': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'MSFT': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'GOOGL': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'QQQ': {'base_size': 30000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
        'SPY': {'base_size': 30000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
        'TQQQ': {'base_size': 25000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8},
        
        # High-momentum individual stocks
        'TSLA': {'base_size': 20000, 'stop_pct': 4.0, 'target_pct': 10.0, 'max_hold': 10},
        'NVDA': {'base_size': 20000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
        'META': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'AMZN': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'NFLX': {'base_size': 18000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
        'AMD': {'base_size': 18000, 'stop_pct': 4.0, 'target_pct': 10.0, 'max_hold': 10},
        'CRM': {'base_size': 18000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
        'UBER': {'base_size': 16000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
        
        # Sector and leveraged ETFs
        'IWM': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
        'XLK': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
        'XLF': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
        'ARKK': {'base_size': 20000, 'stop_pct': 3.5, 'target_pct': 8.0, 'max_hold': 10},
        'SOXL': {'base_size': 22000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8},
        'SPXL': {'base_size': 25000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8}
    }
    return configs.get(symbol, {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15})

def run_detailed_backtest_with_logs():
    """Run backtest and generate detailed trade logs"""
    
    print("GENERATING DETAILED TRADE LOGS - 20 TICKER BACKTEST")
    print("Creating comprehensive CSV and text logs...")
    print("=" * 80)
    
    data_client = AlpacaDataClient()
    
    # Test period
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    # Expanded symbol universe
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ',
        'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD', 'CRM', 'UBER',
        'IWM', 'XLK', 'XLF', 'ARKK', 'SOXL', 'SPXL'
    ]
    
    initial_capital = 100000.0
    all_trades = []
    current_capital = initial_capital
    
    # Create CSV file for detailed logs
    csv_filename = 'detailed_trade_log.csv'
    txt_filename = 'detailed_trade_log.txt'
    
    # CSV Headers
    csv_headers = [
        'Trade_ID', 'Symbol', 'Entry_Date', 'Entry_Time', 'Entry_Price', 'Shares',
        'Position_Size', 'Stop_Loss', 'Take_Profit', 'Exit_Date', 'Exit_Time', 
        'Exit_Price', 'Exit_Reason', 'Hold_Days', 'PnL_Dollar', 'PnL_Percent',
        'Momentum_Score', 'RSI', 'Volume_Ratio', 'Capital_Before', 'Capital_After',
        'Running_Return_Pct', 'Symbol_Category'
    ]
    
    # Open CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        
        # Open text file for detailed logs
        with open(txt_filename, 'w') as txtfile:
            txtfile.write("DETAILED TRADE LOG - EXPANDED AGGRESSIVE STRATEGY\n")
            txtfile.write("=" * 80 + "\n")
            txtfile.write(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
            txtfile.write(f"Initial Capital: ${initial_capital:,.2f}\n")
            txtfile.write(f"Universe: {len(symbols)} symbols\n")
            txtfile.write("=" * 80 + "\n\n")
            
            trade_counter = 0
            
            for symbol in symbols:
                print(f"\nProcessing {symbol}...")
                txtfile.write(f"\n--- {symbol} TRADES ---\n")
                
                try:
                    # Get data
                    df_raw = data_client.get_stock_bars(
                        symbols=[symbol],
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    
                    if df_raw.empty:
                        print(f"No data for {symbol}")
                        continue
                    
                    # Fix MultiIndex
                    df = fix_dataframe_index(df_raw, symbol)
                    
                    if len(df) < 20:
                        print(f"Insufficient data for {symbol}")
                        continue
                    
                    # Get position configuration
                    config = get_position_config(symbol)
                    
                    # Dynamic position sizing
                    position_multiplier = current_capital / initial_capital
                    position_size = config['base_size'] * position_multiplier
                    max_position = current_capital * 0.40
                    position_size = min(position_size, max_position)
                    
                    # Generate signals
                    signals = expanded_momentum_strategy(df, symbol, lookback=10)
                    
                    if not signals:
                        txtfile.write(f"No signals generated for {symbol}\n")
                        continue
                    
                    # Determine symbol category
                    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'CRM']:
                        category = 'Large_Cap_Stock'
                    elif symbol in ['TSLA', 'NVDA', 'NFLX', 'AMD', 'UBER']:
                        category = 'Growth_Stock'
                    elif symbol in ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF']:
                        category = 'Standard_ETF'
                    elif symbol in ['ARKK']:
                        category = 'Growth_ETF'
                    elif symbol in ['TQQQ', 'SOXL', 'SPXL']:
                        category = 'Leveraged_ETF'
                    else:
                        category = 'Other'
                    
                    # Simulate trades
                    for signal in signals:
                        trade_counter += 1
                        capital_before = current_capital
                        
                        entry_price = signal['price']
                        entry_date = signal['date']
                        shares = position_size / entry_price
                        
                        # Risk parameters
                        stop_pct = config['stop_pct']
                        target_pct = config['target_pct']
                        max_hold = config['max_hold']
                        
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
                        hold_days = (exit_date - entry_date).days
                        
                        # Update capital
                        current_capital += pnl
                        running_return = (current_capital / initial_capital - 1) * 100
                        
                        # Write to CSV
                        csv_row = [
                            trade_counter, symbol, entry_date.strftime('%Y-%m-%d'), 
                            entry_date.strftime('%H:%M:%S'), f"{entry_price:.2f}", f"{shares:.3f}",
                            f"{position_size:.2f}", f"{stop_loss:.2f}", f"{take_profit:.2f}",
                            exit_date.strftime('%Y-%m-%d'), exit_date.strftime('%H:%M:%S'),
                            f"{exit_price:.2f}", exit_reason, hold_days, f"{pnl:.2f}",
                            f"{pnl_pct:.2f}", f"{signal['momentum_score']:.2f}", 
                            f"{signal['rsi']:.2f}", f"{signal['volume_ratio']:.2f}",
                            f"{capital_before:.2f}", f"{current_capital:.2f}",
                            f"{running_return:.2f}", category
                        ]
                        writer.writerow(csv_row)
                        
                        # Write to text file (detailed format)
                        txtfile.write(f"\nTrade #{trade_counter:4d} - {symbol}\n")
                        txtfile.write(f"  Entry: {entry_date.strftime('%Y-%m-%d %H:%M')} @ ${entry_price:8.2f} ({shares:8.3f} shares)\n")
                        txtfile.write(f"  Position Size: ${position_size:10,.2f} | Stop: ${stop_loss:8.2f} | Target: ${take_profit:8.2f}\n")
                        txtfile.write(f"  Signal: Momentum {signal['momentum_score']:6.2f}% | RSI {signal['rsi']:5.1f} | Volume {signal['volume_ratio']:5.2f}x\n")
                        txtfile.write(f"  Exit:  {exit_date.strftime('%Y-%m-%d %H:%M')} @ ${exit_price:8.2f} ({exit_reason})\n")
                        txtfile.write(f"  Hold:  {hold_days:3d} days | P&L: ${pnl:+10.2f} ({pnl_pct:+6.2f}%)\n")
                        txtfile.write(f"  Capital: ${capital_before:12,.2f} → ${current_capital:12,.2f} (Return: {running_return:+7.2f}%)\n")
                        txtfile.write(f"  Category: {category}\n")
                        
                        # Store trade record
                        trade_record = {
                            'trade_id': trade_counter,
                            'symbol': symbol,
                            'category': category,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'exit_reason': exit_reason,
                            'hold_days': hold_days,
                            'capital_after': current_capital
                        }
                        
                        all_trades.append(trade_record)
                
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {e}"
                    print(error_msg)
                    txtfile.write(f"{error_msg}\n")
            
            # Write summary to text file
            if all_trades:
                txtfile.write(f"\n{'='*80}\n")
                txtfile.write("BACKTEST SUMMARY\n")
                txtfile.write(f"{'='*80}\n")
                
                total_pnl = current_capital - initial_capital
                final_return = (current_capital / initial_capital - 1) * 100
                win_trades = [t for t in all_trades if t['pnl'] > 0]
                
                txtfile.write(f"Total Trades: {len(all_trades)}\n")
                txtfile.write(f"Winning Trades: {len(win_trades)} ({len(win_trades)/len(all_trades)*100:.1f}%)\n")
                txtfile.write(f"Initial Capital: ${initial_capital:,.2f}\n")
                txtfile.write(f"Final Capital: ${current_capital:,.2f}\n")
                txtfile.write(f"Total P&L: ${total_pnl:+,.2f}\n")
                txtfile.write(f"Final Return: {final_return:+.2f}%\n")
    
    print(f"\n✅ DETAILED LOGS GENERATED:")
    print(f"📊 CSV File: {csv_filename}")
    print(f"📄 Text File: {txt_filename}")
    print(f"📈 Total Trades Logged: {len(all_trades)}")
    print(f"💰 Final Return: {(current_capital/initial_capital-1)*100:+.2f}%")

if __name__ == "__main__":
    run_detailed_backtest_with_logs()