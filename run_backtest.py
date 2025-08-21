#!/usr/bin/env python3
"""
Main script to run backtests for trading strategies.
Updated for minute bars and shared strategy core.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import shared strategy core
from shared.strategy_core import (
    Params, compute_features, decide_entries, decide_exits, 
    size_order, log_bar_freshness, is_fresh_data
)
from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    # Get log level from environment or use default
    log_level = os.getenv('LOG_LEVEL', 'DEBUG' if verbose else 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backtest.log')
        ]
    )


def get_minute_bars(data_client, symbols: list, lookback_minutes: int = 720) -> dict:
    """Fetch minute bars from Alpaca for backtesting.
    
    Args:
        data_client: Alpaca data client
        symbols: List of symbols to fetch
        lookback_minutes: Number of minutes to look back
        
    Returns:
        Dictionary of symbol -> DataFrame
    """
    data = {}
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(minutes=lookback_minutes)
    
    logging.info(f"Fetching minute bars from {start} to {end} ({lookback_minutes} minutes)")
    
    for symbol in symbols:
        try:
            bars = data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Minute,
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
                logging.info(f"Loaded {len(df)} minute bars for {symbol}")
            else:
                logging.warning(f"No data for {symbol}")
                
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            
    return data


def run_shared_core_backtest(data: dict, params: Params, initial_capital: float = 10000.0) -> dict:
    """Run backtest using shared strategy core.
    
    Args:
        data: Dictionary of symbol -> DataFrame with OHLCV data
        params: Strategy parameters
        initial_capital: Starting capital
        
    Returns:
        Backtest results dictionary
    """
    results = {
        'trades': [],
        'portfolio_values': [],
        'initial_capital': initial_capital,
        'total_return': 0.0,
        'total_trades': 0,
        'winning_trades': 0
    }
    
    current_capital = initial_capital
    positions = {}
    trade_count = 0
    
    logging.info(f"Starting backtest with ${initial_capital:,.2f} initial capital")
    
    # Get all unique timestamps across all symbols
    all_timestamps = set()
    for df in data.values():
        all_timestamps.update(df.index)
    
    timestamps = sorted(all_timestamps)
    logging.info(f"Backtesting across {len(timestamps)} minute bars")
    
    for i, timestamp in enumerate(timestamps):
        if i == 0:
            continue  # Skip first bar
            
        current_value = current_capital
        
        # Process each symbol at this timestamp
        for symbol, df in data.items():
            if timestamp not in df.index or i >= len(df):
                continue
                
            # Get data up to current point
            historical_data = df.loc[:timestamp]
            if len(historical_data) < 50:  # Need enough history
                continue
                
            # Compute features
            feat = compute_features(historical_data, params)
            if feat.empty:
                continue
                
            # Check entry signal at t-1 (previous bar)
            entries = decide_entries(feat, params)
            if len(entries) < 2:
                continue
                
            entry_signal = entries.iloc[-2]  # Signal at t-1
            
            # Execute at t (current bar open)
            if entry_signal and symbol not in positions:
                # Get execution price (current bar open)
                if i < len(df) and timestamp in df.index:
                    exec_price = float(df.loc[timestamp, 'open'])
                    
                    # Calculate position size
                    qty = size_order(exec_price, params, fractionable=True)
                    trade_value = qty * exec_price
                    
                    if trade_value <= current_capital * 0.9:  # Leave some cash
                        # Enter position
                        positions[symbol] = {
                            'qty': qty,
                            'entry_price': exec_price,
                            'entry_time': timestamp,
                            'trade_id': trade_count
                        }
                        
                        current_capital -= trade_value
                        trade_count += 1
                        
                        logging.debug(f"ENTRY: {symbol} {qty:.2f} @ ${exec_price:.2f} at {timestamp}")
            
            # Check exits for existing positions
            if symbol in positions:
                position = positions[symbol]
                current_price = float(df.loc[timestamp, 'close'])
                
                # Simple exit: 5% profit target or 2% stop loss
                pnl_pct = (current_price / position['entry_price'] - 1) * 100
                
                should_exit = False
                exit_reason = "hold"
                
                if pnl_pct >= 5.0:
                    should_exit = True
                    exit_reason = "profit_target"
                elif pnl_pct <= -2.0:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif (timestamp - position['entry_time']).total_seconds() > 7200:  # 2 hours max hold
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
                        
                    logging.debug(f"EXIT: {symbol} ${current_price:.2f} PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) - {exit_reason}")
                    
                    del positions[symbol]
                else:
                    # Add position value to current portfolio value
                    current_value += position['qty'] * current_price
        
        # Record portfolio value
        total_portfolio_value = current_capital + sum(
            pos['qty'] * float(data[symbol].loc[timestamp, 'close'])
            for symbol, pos in positions.items()
            if timestamp in data[symbol].index
        )
        
        results['portfolio_values'].append({
            'timestamp': timestamp,
            'value': total_portfolio_value
        })
    
    # Calculate final results
    final_value = results['portfolio_values'][-1]['value'] if results['portfolio_values'] else initial_capital
    results['total_return'] = (final_value / initial_capital - 1) * 100
    results['final_value'] = final_value
    results['win_rate'] = (results['winning_trades'] / max(1, results['total_trades'])) * 100
    
    return results


def run_backtest(config_file: str = None, start_date: str = None, 
                end_date: str = None, output_dir: str = None,
                strategies: list = None) -> None:
    """Run a complete backtest using minute bars and shared strategy core.
    
    Args:
        config_file: Path to configuration file (optional)
        start_date: Start date for backtest (optional)
        end_date: End date for backtest (optional)
        output_dir: Directory to save results (optional)
        strategies: List of specific strategies to test (optional)
    """
    print("Starting minute bar backtest with shared strategy core...")
    
    # Configuration
    symbols = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOGL,SPY,QQQ').split(',')
    lookback_minutes = int(os.getenv('LOOKBACK_MINUTES', '720'))  # 12 hours default
    initial_capital = 10000.0
    
    print(f"Symbols: {symbols}")
    print(f"Lookback: {lookback_minutes} minutes")
    print(f"Initial capital: ${initial_capital:,.2f}")
    
    # Initialize data client
    try:
        data_client = AlpacaDataClient()
        print("[OK] Data client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize data client: {e}")
        return
    
    # Fetch minute bars
    print("\nFetching minute bars...")
    data = get_minute_bars(data_client, symbols, lookback_minutes)
    
    if not data:
        print("[ERROR] No data fetched. Check symbols and API credentials.")
        return
    
    print(f"[OK] Loaded data for {len(data)} symbols")
    
    # Log data freshness for each symbol
    for symbol, df in data.items():
        log_bar_freshness(symbol, df)
    
    # Create strategy parameters
    params = Params(
        lookback_sma=20,
        rsi_len=14,
        atr_len=14,
        equity_notional=2.0,  # $2K per trade for small test account
        submit_bps=5.0,
        min_volume_5m=1000,   # Lower for minute bars
        max_spread_bps=20.0
    )
    
    print(f"\nStrategy parameters: {params}")
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        results = run_shared_core_backtest(data, params, initial_capital)
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS (Minute Bars + Shared Core)")
        print("="*50)
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:+.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        
        if results['trades']:
            print("\nRecent Trades:")
            for trade in results['trades'][-5:]:  # Last 5 trades
                print(f"  {trade['symbol']}: {trade['pnl_pct']:+.1f}% (${trade['pnl']:+.2f}) - {trade['exit_reason']}")
        
        # Log final summary
        logging.info(f"Backtest complete: {results['total_return']:+.2f}% return, {results['total_trades']} trades")
        print(f"\n[OK] Backtest completed! Check backtest.log for detailed logs.")
        
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        logging.error(f"Backtest error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run trading strategy backtest with minute bars')
    
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file path (optional)')
    parser.add_argument('--start-date', '-s', type=str,
                       help='Start date (YYYY-MM-DD, optional)')
    parser.add_argument('--end-date', '-e', type=str,
                       help='End date (YYYY-MM-DD, optional)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for results (optional)')
    parser.add_argument('--strategies', nargs='+',
                       help='Specific strategies to test (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated symbols to test (default: AAPL,MSFT,GOOGL,SPY,QQQ)')
    parser.add_argument('--lookback-minutes', type=int, default=720,
                       help='Minutes of historical data to fetch (default: 720)')
    
    args = parser.parse_args()
    
    # Set environment variables from args
    if args.symbols:
        os.environ['SYMBOLS'] = args.symbols
    if args.lookback_minutes:
        os.environ['LOOKBACK_MINUTES'] = str(args.lookback_minutes)
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("Minute Bar Backtest - Signal at t-1, Execute at t+1 Open")
    print("Using shared strategy core for backtest <-> live parity")
    print()
    
    # Run backtest
    run_backtest(
        config_file=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        strategies=args.strategies
    )


if __name__ == '__main__':
    main()