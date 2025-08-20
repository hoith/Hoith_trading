#!/usr/bin/env python3
"""
Simple test for 2025 data using existing infrastructure
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_2025_data_availability():
    """Test if we can get 2025 data from Alpaca"""
    
    print("Testing 2025 Data Availability")
    print("=" * 40)
    
    try:
        # Initialize data client
        data_client = AlpacaDataClient()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Test different date ranges
        test_periods = [
            ("2025-01-01", "2025-01-31", "January 2025"),
            ("2025-07-01", "2025-08-10", "July-August 2025"), 
            ("2025-01-01", "2025-08-10", "Full Period"),
        ]
        
        for start_str, end_str, description in test_periods:
            print(f"\n--- Testing {description} ---")
            
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            
            for symbol in symbols:
                try:
                    # Use the working get_stock_bars method
                    df = data_client.get_stock_bars(
                        symbols=[symbol],
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    
                    if not df.empty:
                        print(f"  {symbol}: {len(df)} days available")
                        print(f"    Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                        print(f"    Latest price: ${df['close'].iloc[-1]:.2f}")
                    else:
                        print(f"  {symbol}: No data available")
                        
                except Exception as e:
                    print(f"  {symbol}: Error - {e}")
    
    except Exception as e:
        print(f"Failed to initialize data client: {e}")
        return False
    
    return True

def run_simple_2025_test():
    """Run a simple test with whatever 2025 data is available"""
    
    print("\n" + "=" * 50)
    print("Simple 2025 Strategy Test")
    print("=" * 50)
    
    try:
        data_client = AlpacaDataClient()
        
        # Try to get recent data (last 30 days from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Testing last 30 days: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = {}
        
        for symbol in symbols:
            try:
                df = data_client.get_stock_bars(
                    symbols=[symbol],
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                if not df.empty and len(df) >= 10:
                    print(f"\n{symbol}: {len(df)} days of data")
                    
                    # Simple momentum check
                    returns = df['close'].pct_change()
                    volatility = returns.std() * np.sqrt(252) * 100
                    recent_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    
                    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                    print(f"  Recent return: {recent_return:.1f}%")
                    print(f"  Volatility: {volatility:.1f}%")
                    
                    # Check for any momentum signals in last 5 days
                    if len(df) >= 15:
                        lookback_return = (df['close'].iloc[-1] / df['close'].iloc[-15] - 1)
                        avg_volume = df['volume'].tail(20).mean()
                        recent_volume = df['volume'].iloc[-1]
                        volume_ratio = recent_volume / avg_volume
                        
                        print(f"  15-day momentum: {lookback_return*100:.1f}%")
                        print(f"  Volume ratio: {volume_ratio:.1f}x")
                        
                        # Check if would trigger our strategy
                        if lookback_return > 0.03 and volume_ratio > 1.1:
                            print(f"  ✅ Would trigger BUY signal!")
                        else:
                            print(f"  ❌ No signal (momentum: {lookback_return*100:.1f}%, volume: {volume_ratio:.1f}x)")
                    
                    results[symbol] = {
                        'data_points': len(df),
                        'recent_return': recent_return,
                        'volatility': volatility,
                        'latest_price': df['close'].iloc[-1]
                    }
                else:
                    print(f"\n{symbol}: Insufficient data ({len(df) if not df.empty else 0} days)")
                    
            except Exception as e:
                print(f"\n{symbol}: Error getting data - {e}")
        
        if results:
            print(f"\n{'='*30}")
            print("SUMMARY")
            print(f"{'='*30}")
            
            for symbol, data in results.items():
                print(f"{symbol}: ${data['latest_price']:.2f}, {data['recent_return']:+.1f}% return, {data['volatility']:.0f}% vol")
            
            print(f"\nStrategy parameters that would be used:")
            print(f"- Momentum threshold: 3.0% over 15 days")
            print(f"- Volume spike: 1.1x average") 
            print(f"- Position size: $50")
            print(f"- ATR stop: 0.8x, Target: 1.6x")
            
        else:
            print("\nNo usable data found for recent period")
            
    except Exception as e:
        print(f"Error in simple test: {e}")

if __name__ == "__main__":
    # First test data availability
    if test_2025_data_availability():
        # Then run simple test
        run_simple_2025_test()
    else:
        print("Cannot proceed - data client issues")