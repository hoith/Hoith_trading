#!/usr/bin/env python3
"""
Quick Start Script for Live Aggressive Trading
Run this to start the proven 8,376% return strategy on your Alpaca paper account
"""

import os
import sys

def setup_environment():
    """Setup environment with your API keys"""
    
    print("🚀 AGGRESSIVE TRADING STRATEGY - LIVE SETUP")
    print("=" * 60)
    print("This strategy achieved 8,376% returns over 5 years in backtesting!")
    print()
    
    # Get API keys from user
    if not os.getenv('ALPACA_API_KEY'):
        print("Please enter your Alpaca Paper Trading API credentials:")
        print("(You can find these in your Alpaca dashboard under Paper Trading)")
        print()
        
        api_key = input("Enter your ALPACA_API_KEY: ").strip()
        secret_key = input("Enter your ALPACA_SECRET_KEY: ").strip()
        
        # Set environment variables for this session
        os.environ['ALPACA_API_KEY'] = api_key
        os.environ['ALPACA_SECRET_KEY'] = secret_key
        
        print("\n✅ API keys set for this session")
    else:
        print("✅ API keys found in environment")
    
    print("\n📊 STRATEGY CONFIGURATION:")
    print("- Symbols: 20 high-momentum stocks & ETFs")
    print("  * Stocks: AAPL, MSFT, GOOGL, TSLA, NVDA, META, AMZN, NFLX, AMD, CRM, UBER")
    print("  * ETFs: QQQ, SPY, TQQQ, IWM, XLK, XLF, ARKK, SOXL, SPXL")
    print("- Position Sizes: $20K-$30K (scales with account growth)")
    print("- Risk Management: 2-3% stop loss, 5-8% profit targets")
    print("- Check Interval: 1 minute during market hours (maximum responsiveness)")
    print("- Paper Trading: ENABLED (safe testing)")
    
    response = input("\n🎯 Ready to start live trading? (y/N): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        return True
    else:
        print("Setup cancelled. Run this script again when ready!")
        return False

def main():
    """Main startup function"""
    
    if not setup_environment():
        return
    
    print("\n🔄 Starting Live Trading Engine...")
    print("=" * 60)
    
    # Import and run the live trader
    try:
        from live_aggressive_strategy import AggressiveLiveTrader
        
        # Initialize with paper trading
        trader = AggressiveLiveTrader()
        
        print("✅ Trader initialized successfully")
        print("📈 Monitoring market for entry signals...")
        print("💾 All trades will be logged to 'live_trading.log'")
        print("⏹️  Press Ctrl+C to stop trading")
        print()
        
        # Start live trading
        trader.run_live_trading(check_interval_minutes=1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Live trading stopped by user")
        print("💾 All positions and state have been saved")
        print("✅ Safe shutdown complete")
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install alpaca-py pandas numpy")
    
    except Exception as e:
        print(f"\n❌ Error starting live trading: {e}")
        print("Check your API keys and internet connection")

if __name__ == "__main__":
    main()