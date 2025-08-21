#!/usr/bin/env python3
"""
Aggressive Strategy Backtest - Real Implementation
Uses the aggressive parameters and logic we just built, bypassing auto-hook
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Set environment variables first
os.environ['APCA_API_KEY_ID'] = 'PKBE0ZILABT1R6BZ6AT3'
os.environ['APCA_API_SECRET_KEY'] = 'hg3jPrh2JAJMw6ksZUND5GCxLtISHjvzbqiuxdMZ'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(level=logging.INFO)

def calculate_aggressive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features for aggressive momentum strategy"""
    feat = df.copy()
    
    # Aggressive momentum indicators
    feat['sma_20'] = feat['close'].rolling(window=20).mean()
    feat['sma_50'] = feat['close'].rolling(window=50).mean()
    
    # RSI for overbought/oversold
    delta = feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    feat['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum scores
    feat['momentum_5'] = feat['close'] / feat['close'].shift(5) - 1  # 5-day momentum
    feat['momentum_10'] = feat['close'] / feat['close'].shift(10) - 1  # 10-day momentum
    feat['momentum_20'] = feat['close'] / feat['close'].shift(20) - 1  # 20-day momentum
    
    # Volume analysis
    feat['volume_sma'] = feat['volume'].rolling(window=10).mean()
    feat['volume_ratio'] = feat['volume'] / feat['volume_sma']
    
    # ATR for volatility
    high_low = feat['high'] - feat['low']
    high_close = np.abs(feat['high'] - feat['close'].shift())
    low_close = np.abs(feat['low'] - feat['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    feat['atr'] = true_range.rolling(window=14).mean()
    
    return feat

def aggressive_entry_signals(symbol: str, feat: pd.DataFrame) -> pd.Series:
    """Generate aggressive entry signals based on symbol type and momentum"""
    entries = pd.Series(False, index=feat.index)
    
    if len(feat) < 50:
        return entries
    
    # Aggressive thresholds by symbol type
    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'CRM']:
        # Large cap growth stocks - moderate aggression
        momentum_threshold = 0.02  # 2% momentum over 10 days
        rsi_max = 75
        volume_min = 1.2
        
    elif symbol in ['TSLA', 'NVDA', 'NFLX', 'AMD', 'UBER']:
        # High-volatility growth stocks - high aggression  
        momentum_threshold = 0.03  # 3% momentum over 10 days
        rsi_max = 70
        volume_min = 1.3
        
    elif symbol in ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF']:
        # ETFs - conservative aggression
        momentum_threshold = 0.015  # 1.5% momentum over 10 days
        rsi_max = 80
        volume_min = 1.1
        
    elif symbol in ['TQQQ', 'SOXL', 'SPXL']:
        # Leveraged ETFs - ultra aggressive
        momentum_threshold = 0.05  # 5% momentum over 10 days  
        rsi_max = 85
        volume_min = 1.0
        
    else:
        # Default moderate settings
        momentum_threshold = 0.02
        rsi_max = 75
        volume_min = 1.2
    
    # Multi-factor aggressive entry conditions
    momentum_strong = feat['momentum_10'] > momentum_threshold
    momentum_accelerating = feat['momentum_5'] > feat['momentum_10'] * 0.5  # Recent acceleration
    rsi_not_overbought = feat['rsi'] < rsi_max
    volume_surge = feat['volume_ratio'] > volume_min
    
    # Price above key moving averages (trend following)
    price_above_sma20 = feat['close'] > feat['sma_20']
    sma20_above_sma50 = feat['sma_20'] > feat['sma_50']  # Uptrend
    
    # Combined aggressive entry signal
    entries = (momentum_strong & 
               momentum_accelerating & 
               rsi_not_overbought & 
               volume_surge & 
               price_above_sma20 & 
               sma20_above_sma50)
    
    return entries

def aggressive_position_size(price: float, symbol: str, capital: float) -> float:
    """Calculate aggressive position sizes based on our live trading strategy"""
    
    # Aggressive position sizing by symbol (from live_aggressive_strategy.py)
    position_configs = {
        # Original proven symbols
        'AAPL': 20000, 'MSFT': 20000, 'GOOGL': 20000,
        'QQQ': 30000, 'SPY': 30000, 'TQQQ': 25000,
        
        # High-momentum individual stocks  
        'TSLA': 20000, 'NVDA': 20000, 'META': 20000,
        'AMZN': 20000, 'NFLX': 18000, 'AMD': 18000,
        'CRM': 18000, 'UBER': 16000,
        
        # Sector and leveraged ETFs
        'IWM': 25000, 'XLK': 25000, 'XLF': 25000,
        'ARKK': 20000, 'SOXL': 22000, 'SPXL': 25000
    }
    
    base_position = position_configs.get(symbol, 15000)  # Default $15K
    
    # Scale with capital growth
    capital_multiplier = capital / 100000.0  # Base on $100K
    target_dollars = base_position * capital_multiplier
    
    # Cap at 40% of portfolio  
    max_position = capital * 0.40
    target_dollars = min(target_dollars, max_position)
    
    return target_dollars / price

def run_aggressive_backtest():
    """Run our actual aggressive momentum strategy"""
    print("AGGRESSIVE MOMENTUM STRATEGY BACKTEST")
    print("Real implementation of live_aggressive_strategy.py logic")
    print("=" * 60)
    
    # Aggressive symbols from live trading strategy
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ',
        'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD'
    ]
    
    lookback_days = 60
    initial_capital = 100000.0  # $100K like in live strategy
    
    print(f"Symbols: {symbols}")  
    print(f"Lookback: {lookback_days} days")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print()
    
    # Get data
    data_client = AlpacaDataClient()
    print("Fetching real market data...")
    
    data = {}
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    
    for symbol in symbols:
        try:
            bars = data_client.get_stock_bars([symbol], TimeFrame.Day, start, end)
            if not bars.empty:
                if isinstance(bars.index, pd.MultiIndex):
                    df = bars.xs(symbol, level=0)[["open","high","low","close","volume"]]
                else:
                    df = bars[["open","high","low","close","volume"]]
                data[symbol] = df
                print(f"[OK] {symbol}: {len(df)} days, last close ${df['close'].iloc[-1]:.2f}")
            else:
                print(f"[SKIP] {symbol}: No data")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
    
    print(f"\n[OK] Loaded data for {len(data)} symbols")
    print()
    
    # Run aggressive backtest
    print("Running AGGRESSIVE momentum strategy...")
    
    current_capital = initial_capital
    positions = {}
    trades = []
    
    # Get all timestamps
    all_timestamps = set()
    for df in data.values():
        all_timestamps.update(df.index)
    timestamps = sorted(all_timestamps)
    
    signal_count = 0
    entry_count = 0
    
    for i, timestamp in enumerate(timestamps):
        if i < 50:  # Need history for indicators
            continue
            
        for symbol, df in data.items():
            if timestamp not in df.index:
                continue
                
            # Get historical data
            historical_data = df.loc[:timestamp]
            if len(historical_data) < 50:
                continue
            
            # Calculate aggressive features
            feat = calculate_aggressive_features(historical_data)
            
            # Generate aggressive entry signals  
            entries = aggressive_entry_signals(symbol, feat)
            if len(entries) < 2:
                continue
                
            entry_signal = entries.iloc[-2]  # Signal at t-1
            
            if entry_signal:
                signal_count += 1
                current_price = float(historical_data['close'].iloc[-1])
                momentum = float(feat['momentum_10'].iloc[-1]) * 100
                rsi = float(feat['rsi'].iloc[-1])
                
                print(f"AGGRESSIVE SIGNAL: {symbol} on {timestamp.date()}")
                print(f"  Price: ${current_price:.2f}, Momentum: {momentum:+.1f}%, RSI: {rsi:.1f}")
                
                # Execute aggressive position sizing
                if symbol not in positions:
                    exec_price = float(df.loc[timestamp, 'open'])
                    qty = aggressive_position_size(exec_price, symbol, current_capital)
                    trade_value = qty * exec_price
                    
                    if trade_value <= current_capital * 0.9:  # Keep some cash
                        positions[symbol] = {
                            'qty': qty,
                            'entry_price': exec_price,
                            'entry_time': timestamp,
                            'signal': {
                                'momentum_10': momentum,
                                'rsi': rsi,
                                'volume_ratio': float(feat['volume_ratio'].iloc[-1])
                            }
                        }
                        
                        current_capital -= trade_value
                        entry_count += 1
                        
                        print(f"AGGRESSIVE ENTRY: {symbol} {qty:.1f} shares @ ${exec_price:.2f} (${trade_value:,.0f})")
                        print()
            
            # Aggressive exit management
            if symbol in positions:
                position = positions[symbol]
                current_price = float(df.loc[timestamp, 'close'])
                
                # Aggressive exit rules by symbol type
                if symbol in ['TQQQ', 'SOXL', 'SPXL']:
                    # Leveraged ETFs - tight stops, quick profits
                    profit_target = 0.05  # 5%
                    stop_loss = 0.02      # 2%  
                    max_days = 8
                elif symbol in ['TSLA', 'NVDA', 'AMD']:
                    # High volatility - wider stops
                    profit_target = 0.10  # 10%
                    stop_loss = 0.04      # 4%
                    max_days = 10
                else:
                    # Default aggressive settings
                    profit_target = 0.08  # 8%
                    stop_loss = 0.03      # 3%
                    max_days = 15
                
                pnl_pct = (current_price / position['entry_price'] - 1)
                days_held = (timestamp - position['entry_time']).days
                
                should_exit = False
                exit_reason = "hold"
                
                if pnl_pct >= profit_target:
                    should_exit = True
                    exit_reason = "profit_target"
                elif pnl_pct <= -stop_loss:
                    should_exit = True  
                    exit_reason = "stop_loss"
                elif days_held >= max_days:
                    should_exit = True
                    exit_reason = "time_limit"
                
                if should_exit:
                    # Exit position
                    exit_value = position['qty'] * current_price
                    current_capital += exit_value
                    
                    pnl = exit_value - (position['qty'] * position['entry_price'])
                    pnl_pct_final = pnl_pct * 100
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct_final,
                        'days_held': days_held,
                        'exit_reason': exit_reason,
                        'signal_data': position['signal']
                    }
                    
                    trades.append(trade)
                    
                    print(f"AGGRESSIVE EXIT: {symbol} @ ${current_price:.2f} ({days_held}d)")
                    print(f"  {exit_reason.upper()}: ${pnl:+,.0f} ({pnl_pct_final:+.1f}%)")
                    print()
                    
                    del positions[symbol]
    
    # Final results
    final_value = current_capital + sum(
        pos['qty'] * float(data[symbol]['close'].iloc[-1])
        for symbol, pos in positions.items()
    )
    
    total_return = (final_value / initial_capital - 1) * 100
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = (winning_trades / max(1, len(trades))) * 100
    
    avg_winner = np.mean([t['pnl_pct'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loser = np.mean([t['pnl_pct'] for t in trades if t['pnl'] < 0]) if len(trades) > winning_trades else 0
    
    # Results
    print("=" * 60)
    print("AGGRESSIVE STRATEGY RESULTS")
    print("=" * 60)
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Final Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Signals Generated: {signal_count}")
    print(f"Positions Entered: {entry_count}")
    print(f"Trades Completed: {len(trades)}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Winner: {avg_winner:+.1f}%")
    print(f"Avg Loser: {avg_loser:+.1f}%")
    
    if trades:
        print(f"\nAll AGGRESSIVE Trades:")
        for i, trade in enumerate(trades, 1):
            entry_date = trade['entry_time'].strftime('%m/%d')
            exit_date = trade['exit_time'].strftime('%m/%d')
            print(f"  {i:2d}. {trade['symbol']:5s}: {entry_date}->{exit_date} ({trade['days_held']:2d}d) "
                  f"{trade['pnl_pct']:+6.1f}% ${trade['pnl']:+8,.0f} - {trade['exit_reason']}")
    
    if positions:
        print(f"\nActive Positions:")
        for symbol, pos in positions.items():
            current_price = float(data[symbol]['close'].iloc[-1])
            unrealized_pnl_pct = (current_price / pos['entry_price'] - 1) * 100
            days_held = (timestamps[-1] - pos['entry_time']).days
            print(f"  {symbol}: ${unrealized_pnl_pct:+.1f}% ({days_held}d held)")

if __name__ == '__main__':
    run_aggressive_backtest()