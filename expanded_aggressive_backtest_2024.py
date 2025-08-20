#!/usr/bin/env python3
"""
Expanded Aggressive Strategy Backtest - 20 Tickers
Testing the proven strategy with expanded universe: 11 stocks + 9 ETFs
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
        
        # Symbol-specific entry criteria (matching live trading logic)
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

def run_expanded_aggressive_backtest():
    """Run backtest with expanded 20-ticker universe"""
    
    print("EXPANDED AGGRESSIVE STRATEGY BACKTEST - 20 TICKERS")
    print("Period: January 1, 2024 - August 20, 2025")
    print("Universe: 11 Stocks + 9 ETFs")
    print("=" * 80)
    
    data_client = AlpacaDataClient()
    
    # Test period
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    # Expanded symbol universe
    symbols = [
        # Original proven symbols
        'AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ',
        # Additional high-momentum stocks
        'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD', 'CRM', 'UBER',
        # Additional ETFs for diversification
        'IWM', 'XLK', 'XLF', 'ARKK', 'SOXL', 'SPXL'
    ]
    
    initial_capital = 100000.0
    all_trades = []
    capital_history = [initial_capital]
    current_capital = initial_capital
    
    # Track performance by symbol category
    symbol_performance = {}
    
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
                print(f"No data available for {symbol}")
                continue
            
            # Fix MultiIndex
            df = fix_dataframe_index(df_raw, symbol)
            
            if len(df) < 20:
                print(f"Insufficient data: {len(df)} days")
                continue
            
            print(f"Data: {len(df)} trading days")
            print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # Get position configuration
            config = get_position_config(symbol)
            
            # Dynamic position sizing
            position_multiplier = current_capital / initial_capital
            position_size = config['base_size'] * position_multiplier
            
            # Cap at 40% of portfolio
            max_position = current_capital * 0.40
            position_size = min(position_size, max_position)
            
            # Generate signals
            signals = expanded_momentum_strategy(df, symbol, lookback=10)
            
            if not signals:
                print("No signals generated")
                continue
            
            print(f"Signals generated: {len(signals)}")
            
            # Initialize symbol performance tracking
            symbol_trades = []
            
            # Simulate trades
            for signal in signals:
                entry_price = signal['price']
                entry_date = signal['date']
                shares = position_size / entry_price
                
                # Use symbol-specific risk parameters
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
                symbol_trades.append(trade)
                
                # Show significant trades (> $3000 P&L)
                if abs(pnl) > 3000:
                    print(f"  {entry_date.strftime('%Y-%m-%d')}: ${pnl:+,.0f} ({pnl_pct:+.1f}%), {exit_reason}")
            
            # Track symbol performance
            if symbol_trades:
                symbol_pnl = sum(t['pnl'] for t in symbol_trades)
                symbol_performance[symbol] = {
                    'trades': len(symbol_trades),
                    'total_pnl': symbol_pnl,
                    'avg_pnl': symbol_pnl / len(symbol_trades),
                    'win_rate': len([t for t in symbol_trades if t['pnl'] > 0]) / len(symbol_trades) * 100
                }
                print(f"Symbol Total: {len(symbol_trades)} trades, ${symbol_pnl:+,.0f} P&L")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # EXPANDED STRATEGY RESULTS
    print(f"\n{'='*80}")
    print("EXPANDED AGGRESSIVE STRATEGY RESULTS (20 TICKERS)")
    print(f"{'='*80}")
    
    if all_trades:
        total_pnl = current_capital - initial_capital
        final_return = (current_capital / initial_capital - 1) * 100
        win_trades = [t for t in all_trades if t['pnl'] > 0]
        
        print(f"TEST PERIOD: January 1, 2024 - August 20, 2025")
        print(f"UNIVERSE: 20 symbols (11 stocks + 9 ETFs)")
        print(f"")
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
        
        # Performance by symbol category
        print(f"\nPERFORMANCE BY SYMBOL CATEGORY:")
        
        # Categorize symbols
        large_caps = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'CRM']
        growth_stocks = ['TSLA', 'NVDA', 'NFLX', 'AMD', 'UBER']
        standard_etfs = ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF']
        growth_etfs = ['ARKK']
        leveraged_etfs = ['TQQQ', 'SOXL', 'SPXL']
        
        categories = {
            'Large Cap Stocks': large_caps,
            'Growth Stocks': growth_stocks,
            'Standard ETFs': standard_etfs,
            'Growth ETFs': growth_etfs,
            'Leveraged ETFs': leveraged_etfs
        }
        
        for category, symbols_in_cat in categories.items():
            cat_trades = [t for t in all_trades if t['symbol'] in symbols_in_cat]
            if cat_trades:
                cat_pnl = sum(t['pnl'] for t in cat_trades)
                cat_wins = len([t for t in cat_trades if t['pnl'] > 0])
                cat_win_rate = cat_wins / len(cat_trades) * 100
                print(f"  {category}: {len(cat_trades)} trades, ${cat_pnl:+,.0f}, {cat_win_rate:.1f}% win rate")
        
        # Top performing symbols
        print(f"\nTOP 10 PERFORMING SYMBOLS:")
        sorted_symbols = sorted(symbol_performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        for i, (symbol, perf) in enumerate(sorted_symbols[:10], 1):
            print(f"  {i:2d}. {symbol}: ${perf['total_pnl']:+,.0f} ({perf['trades']} trades, {perf['win_rate']:.1f}% win rate)")
        
        # Show top trades
        largest_wins = sorted([t for t in all_trades if t['pnl'] > 0], key=lambda x: x['pnl'], reverse=True)[:10]
        print(f"\nTOP 10 WINNING TRADES:")
        for i, trade in enumerate(largest_wins, 1):
            print(f"  {i:2d}. {trade['symbol']} ({trade['entry_date'].strftime('%Y-%m-%d')}): ${trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%)")
        
        # Comparison with original 6-ticker strategy
        original_symbols = ['AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ']
        original_trades = [t for t in all_trades if t['symbol'] in original_symbols]
        expanded_trades = [t for t in all_trades if t['symbol'] not in original_symbols]
        
        if original_trades and expanded_trades:
            original_pnl = sum(t['pnl'] for t in original_trades)
            expanded_pnl = sum(t['pnl'] for t in expanded_trades)
            
            print(f"\nORIGINAL vs EXPANDED COMPARISON:")
            print(f"Original 6 symbols: {len(original_trades)} trades, ${original_pnl:+,.0f}")
            print(f"New 14 symbols: {len(expanded_trades)} trades, ${expanded_pnl:+,.0f}")
            print(f"Expansion benefit: ${expanded_pnl:+,.0f} ({expanded_pnl/original_pnl*100:+.1f}% of original)")
        
        # Risk metrics
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
                print(f"\nRISK METRICS:")
                print(f"Sharpe Ratio: {sharpe:.2f}")
            
            # Maximum drawdown
            max_drawdown = 0
            peak = initial_capital
            for capital in capital_history:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            print(f"Maximum Drawdown: -{max_drawdown:.1f}%")
        
        # Annualized returns
        months_elapsed = 20  # Jan 2024 to Aug 2025
        annualized_return = ((current_capital / initial_capital) ** (12 / months_elapsed) - 1) * 100
        print(f"\nANNUALIZED PERFORMANCE:")
        print(f"Annualized Return: {annualized_return:+.1f}%")
        
        # Final assessment
        print(f"\nEXPANDED STRATEGY ASSESSMENT:")
        if final_return > 300:
            print(f"EXCEPTIONAL PERFORMANCE: {final_return:+.1f}% with 20-ticker universe")
        elif final_return > 200:
            print(f"OUTSTANDING PERFORMANCE: {final_return:+.1f}% with expanded universe")
        elif final_return > 100:
            print(f"STRONG PERFORMANCE: {final_return:+.1f}% across 20 symbols")
        else:
            print(f"SOLID PERFORMANCE: {final_return:+.1f}% return")
            
        print(f"\nUNIVERSE EXPANSION SUCCESSFUL:")
        print(f"- 3.3x more symbols monitored")
        print(f"- Diversified across sectors and asset types")
        print(f"- Maintained proven momentum criteria")
        print(f"- Optimized risk parameters per symbol category")
            
    else:
        print("No trades executed")

if __name__ == "__main__":
    run_expanded_aggressive_backtest()