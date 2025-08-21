#!/usr/bin/env python3
"""
Daily Bar Backtest - Real Data Demo
Demonstrates the system with daily bars (which should be available on free tier)
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Set environment variables first
os.environ['APCA_API_KEY_ID'] = 'PKBE0ZILABT1R6BZ6AT3'
os.environ['APCA_API_SECRET_KEY'] = 'hg3jPrh2JAJMw6ksZUND5GCxLtISHjvzbqiuxdMZ'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import shared strategy core
from shared.strategy_core import (
    Params, compute_features, decide_entries, decide_exits, 
    size_order, log_bar_freshness, is_fresh_data
)
from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('daily_backtest.log')
    ]
)

def get_daily_bars(data_client, symbols: list, lookback_days: int = 30) -> dict:
    """Fetch daily bars from Alpaca for backtesting."""
    data = {}
    end = datetime.now(timezone.utc) - timedelta(days=1)  # Yesterday to avoid real-time data issues
    start = end - timedelta(days=lookback_days)
    
    logging.info(f"Fetching daily bars from {start.date()} to {end.date()} ({lookback_days} days)")
    
    for symbol in symbols:
        try:
            bars = data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            if not bars.empty:
                # Handle MultiIndex if present
                if isinstance(bars.index, pd.MultiIndex):
                    df = bars.xs(symbol, level=0)[["open","high","low","close","volume"]]
                else:
                    df = bars[["open","high","low","close","volume"]]
                    
                data[symbol] = df
                logging.info(f"Loaded {len(df)} daily bars for {symbol}")
                print(f"[OK] {symbol}: {len(df)} days, ${df['close'].iloc[-1]:.2f} last close")
            else:
                logging.warning(f"No data for {symbol}")
                
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            
    return data

def run_daily_backtest():
    """Run backtest with daily bars"""
    print("DAILY BAR BACKTEST - Real Data Demo")
    print("Using shared strategy core for backtest <-> live parity")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    lookback_days = 60  # 2 months of data
    initial_capital = 10000.0
    
    print(f"Symbols: {symbols}")
    print(f"Lookback: {lookback_days} days")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print()
    
    # Initialize data client
    try:
        data_client = AlpacaDataClient()
        print("[OK] Data client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize data client: {e}")
        return
    
    # Fetch daily bars
    print("Fetching daily bars...")
    data = get_daily_bars(data_client, symbols, lookback_days)
    
    if not data:
        print("[ERROR] No data fetched. Check symbols and API credentials.")
        return
    
    print(f"[OK] Loaded data for {len(data)} symbols")
    print()
    
    # Create strategy parameters (adapted for daily bars)
    params = Params(
        lookback_sma=10,   # Shorter for daily bars
        rsi_len=14,
        atr_len=14,
        equity_notional=3.0,  # $3K per trade
        submit_bps=5.0,
        min_volume_5m=100000,  # Higher for daily
        max_spread_bps=20.0
    )
    
    print(f"Strategy parameters (adapted for daily bars): {params}")
    print()
    
    # Run simple backtest
    print("Running daily bar backtest...")
    
    results = {
        'trades': [],
        'initial_capital': initial_capital,
        'total_trades': 0,
        'winning_trades': 0
    }
    
    current_capital = initial_capital
    positions = {}
    
    # Get all timestamps
    all_timestamps = set()
    for df in data.values():
        all_timestamps.update(df.index)
    
    timestamps = sorted(all_timestamps)
    print(f"Processing {len(timestamps)} daily bars...")
    
    signal_count = 0
    entry_count = 0
    
    for i, timestamp in enumerate(timestamps):
        if i < 20:  # Need enough history for indicators
            continue
            
        # Process each symbol
        for symbol, df in data.items():
            if timestamp not in df.index:
                continue
                
            # Get historical data up to current point
            historical_data = df.loc[:timestamp]
            if len(historical_data) < 20:
                continue
            
            # Compute features using shared core
            feat = compute_features(historical_data, params)
            if feat.empty:
                continue
            
            # Check entry signal at t-1 (previous day)
            entries = decide_entries(feat, params)
            if len(entries) < 2:
                continue
                
            entry_signal = entries.iloc[-2]  # Signal at t-1
            
            if entry_signal:
                signal_count += 1
                current_price = float(historical_data['close'].iloc[-1])
                rsi_val = float(feat['rsi'].iloc[-1]) if 'rsi' in feat.columns else 0
                
                print(f"SIGNAL: {symbol} on {timestamp.date()} - RSI: {rsi_val:.1f}, "
                      f"Price: ${current_price:.2f}")
                
                # Execute at next day open (simulate t+1 execution)
                if symbol not in positions:
                    exec_price = float(df.loc[timestamp, 'open'])  # Use current day's open as proxy
                    qty = size_order(exec_price, params, fractionable=True)
                    trade_value = qty * exec_price
                    
                    if trade_value <= current_capital * 0.8:  # Keep some cash
                        # Enter position
                        positions[symbol] = {
                            'qty': qty,
                            'entry_price': exec_price,
                            'entry_time': timestamp
                        }
                        
                        current_capital -= trade_value
                        entry_count += 1
                        
                        print(f"ENTRY: {symbol} {qty:.2f} shares @ ${exec_price:.2f}")
            
            # Check exits for existing positions
            if symbol in positions:
                position = positions[symbol]
                current_price = float(df.loc[timestamp, 'close'])
                
                # Daily bar exit rules: 5% profit target or 3% stop loss
                pnl_pct = (current_price / position['entry_price'] - 1) * 100
                
                should_exit = False
                exit_reason = "hold"
                
                if pnl_pct >= 5.0:
                    should_exit = True
                    exit_reason = "profit_target"
                elif pnl_pct <= -3.0:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif (timestamp - position['entry_time']).days >= 10:  # 10 days max
                    should_exit = True
                    exit_reason = "time_limit"
                
                if should_exit:
                    # Exit position
                    exit_value = position['qty'] * current_price
                    current_capital += exit_value
                    
                    pnl = exit_value - (position['qty'] * position['entry_price'])
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    }
                    
                    results['trades'].append(trade)
                    results['total_trades'] += 1
                    
                    if pnl > 0:
                        results['winning_trades'] += 1
                    
                    print(f"EXIT: {symbol} @ ${current_price:.2f} - {exit_reason} - "
                          f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                    
                    del positions[symbol]
    
    # Final results
    final_value = current_capital + sum(
        pos['qty'] * float(data[symbol]['close'].iloc[-1])
        for symbol, pos in positions.items()
    )
    
    total_return = (final_value / initial_capital - 1) * 100
    win_rate = (results['winning_trades'] / max(1, results['total_trades'])) * 100
    
    # Display results
    print("\n" + "=" * 60)
    print("DAILY BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Signals Generated: {signal_count}")
    print(f"Positions Entered: {entry_count}")
    print(f"Trades Completed: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {win_rate:.1f}%")
    
    if results['trades']:
        print(f"\nAll Trades:")
        for i, trade in enumerate(results['trades'], 1):
            entry_date = trade['entry_time'].strftime('%Y-%m-%d')
            exit_date = trade['exit_time'].strftime('%Y-%m-%d')
            days_held = (trade['exit_time'] - trade['entry_time']).days
            print(f"  {i}. {trade['symbol']}: {entry_date} -> {exit_date} ({days_held}d) "
                  f"{trade['pnl_pct']:+.1f}% (${trade['pnl']:+.2f}) - {trade['exit_reason']}")
    
    if positions:
        print(f"\nActive Positions:")
        for symbol, pos in positions.items():
            current_price = float(data[symbol]['close'].iloc[-1])
            unrealized_pnl = (current_price - pos['entry_price']) * pos['qty']
            unrealized_pct = (current_price / pos['entry_price'] - 1) * 100
            days_held = (timestamps[-1] - pos['entry_time']).days
            print(f"  {symbol}: {pos['qty']:.2f} @ ${pos['entry_price']:.2f} ({days_held}d) "
                  f"Unrealized: ${unrealized_pnl:+.2f} ({unrealized_pct:+.1f}%)")
    
    print(f"\n[OK] Daily bar backtest completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Real Alpaca data (daily bars)")
    print("- Signal at t-1, execute at t open")
    print("- Shared strategy core (compute_features, decide_entries)")
    print("- Position sizing and risk management")
    print("- Comprehensive trade logging")

if __name__ == '__main__':
    run_daily_backtest()