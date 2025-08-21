#!/usr/bin/env python3
"""
LAUNCHER FOR LIVE ENHANCED HIGH-FREQUENCY OPTIONS TRADER
Easy startup script with configuration options
"""

import os
import sys
from datetime import datetime

def main():
    print("=" * 60)
    print("LIVE ENHANCED HIGH-FREQUENCY OPTIONS TRADER")
    print("=" * 60)
    print(f"Starting at: {datetime.now()}")
    print()
    
    # Configuration options
    print("Configuration:")
    print(f"- Initial Capital: $10,000 (simulated options)")
    print(f"- Target: 15 trades/day")
    print(f"- Universe: 49 high-volume tickers")
    print(f"- Strategy: Multi-signal options simulation")
    print(f"- Backtested Return: +367.8%")
    print()
    
    # Environment variables (optional)
    env_vars = {
        'LOOKBACK_DAYS': '30',          # Historical data lookback
        'SLEEP_SECONDS': '300',         # 5 minutes between scans
        'MAX_DAILY_TRADES': '20',       # Daily trade limit
        'LOG_LEVEL': 'INFO'             # Logging level
    }
    
    print("Setting environment variables:")
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: {os.environ[key]} (existing)")
    
    print()
    print("IMPORTANT NOTES:")
    print("- This simulates options trading (no real options trades)")
    print("- Uses paper trading account for underlying stock data")
    print("- Runs during market hours (9 AM - 4 PM ET)")
    print("- Press Ctrl+C to stop and see daily summary")
    print("- Logs saved to: live_options_trading.log")
    print()
    
    # Confirmation
    response = input("Start live trading? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\nStarting live trader...")
    print("=" * 60)
    
    # Import and run the live trader
    try:
        from live_enhanced_high_frequency_options import LiveEnhancedOptionsTrader
        
        trader = LiveEnhancedOptionsTrader(initial_capital=10000.0)
        trader.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRADING STOPPED BY USER")
        print("=" * 60)
        if 'trader' in locals():
            trader.print_daily_summary()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Check live_options_trading.log for details")

if __name__ == "__main__":
    main()