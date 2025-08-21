#!/usr/bin/env python3
"""
Thin Wrapper for Live Trading with Environment Controls
Updated for minute bars, asset-class routing, and backtest ↔ live parity
"""

import os
import sys
import argparse
from typing import Optional

def setup_environment(args: Optional[object] = None):
    """Setup environment variables for live trading"""
    
    print("LIVE TRADING SYSTEM - MINUTE BARS + ASSET-CLASS ROUTING")
    print("=" * 70)
    print("Features:")
    print("- Backtest <-> Live parity with shared strategy core")
    print("- Minute bar timeframe for both backtest and live")
    print("- Asset-class routing: Equities (extended hours) vs Options (RTH only)")
    print("- Comprehensive logging with configurable levels")
    print("- Paper trading enabled for safety")
    print()
    
    # Set environment variables from command line args or prompts
    if args:
        # Command line mode
        if args.symbols:
            os.environ['SYMBOLS'] = args.symbols
            print(f"[SET] Symbols: {args.symbols}")
        
        if args.lookback_minutes:
            os.environ['LOOKBACK_MINUTES'] = str(args.lookback_minutes)
            print(f"[SET] Lookback: {args.lookback_minutes} minutes")
        
        if args.sleep_seconds:
            os.environ['SLEEP_SECONDS'] = str(args.sleep_seconds)
            print(f"[SET] Check interval: {args.sleep_seconds} seconds")
        
        if args.log_level:
            os.environ['LOG_LEVEL'] = args.log_level
            print(f"[SET] Log level: {args.log_level}")
    
    # Set defaults for any missing environment variables
    defaults = {
        'SYMBOLS': 'AAPL,MSFT,GOOGL,SPY,QQQ',
        'LOOKBACK_MINUTES': '720',
        'SLEEP_SECONDS': '60',
        'LOG_LEVEL': 'INFO'
    }
    
    for key, default_value in defaults.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"[DEFAULT] {key}: {default_value}")
    
    # Check for required API keys
    if not os.getenv('APCA_API_KEY_ID') or not os.getenv('APCA_API_SECRET_KEY'):
        print("\n[WARNING] API CREDENTIALS REQUIRED")
        print("Please set your Alpaca Paper Trading API credentials:")
        print("(Find these in your Alpaca dashboard under Paper Trading)")
        print()
        
        if not args or args.interactive:
            # Interactive mode
            api_key = input("Enter APCA_API_KEY_ID: ").strip()
            secret_key = input("Enter APCA_API_SECRET_KEY: ").strip()
            
            if api_key and secret_key:
                os.environ['APCA_API_KEY_ID'] = api_key
                os.environ['APCA_API_SECRET_KEY'] = secret_key
                print("[OK] API keys set for this session")
            else:
                print("[ERROR] API keys required. Exiting.")
                return False
        else:
            # Non-interactive mode
            print("[ERROR] Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
            print("   Example: export APCA_API_KEY_ID=your_key")
            print("   Example: export APCA_API_SECRET_KEY=your_secret")
            return False
    else:
        print("[OK] API credentials found")
    
    print("\nCURRENT CONFIGURATION:")
    print(f"- Symbols: {os.environ['SYMBOLS']}")
    print(f"- Lookback: {os.environ['LOOKBACK_MINUTES']} minutes")
    print(f"- Check interval: {os.environ['SLEEP_SECONDS']} seconds") 
    print(f"- Log level: {os.environ['LOG_LEVEL']}")
    print(f"- Paper trading: ENABLED")
    
    return True

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Start live trading with minute bars and asset-class routing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_live_trading.py
  python start_live_trading.py --symbols AAPL,MSFT,SPY --log-level DEBUG
  python start_live_trading.py --lookback-minutes 480 --sleep-seconds 30
  python start_live_trading.py --non-interactive
        """
    )
    
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated symbols to trade (default: AAPL,MSFT,GOOGL,SPY,QQQ)')
    parser.add_argument('--lookback-minutes', type=int, default=720,
                       help='Minutes of historical data to fetch (default: 720)')
    parser.add_argument('--sleep-seconds', type=int, default=60,
                       help='Seconds between trading cycles (default: 60)')
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run without prompting for API keys')
    
    args = parser.parse_args()
    args.interactive = not args.non_interactive
    
    # Setup environment
    if not setup_environment(args):
        sys.exit(1)
    
    print("\nSTARTING LIVE TRADING ENGINE...")
    print("=" * 70)
    
    # Import and run the live trader
    try:
        # Import main_loop from live_aggressive_strategy
        from live_aggressive_strategy import main_loop
        
        print("[OK] Live trading module loaded")
        print("Monitoring market for signals...")
        print("Logs will be written to 'live_trading.log'")
        print("Press Ctrl+C to stop trading safely")
        print()
        
        # Start the main trading loop
        main_loop()
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Live trading stopped by user")
        print("All positions and state have been saved")
        print("Safe shutdown complete")
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("Required dependencies:")
        print("- alpaca-py: pip install alpaca-py")
        print("- pandas: pip install pandas")
        print("- numpy: pip install numpy")
        print("- python-dotenv: pip install python-dotenv (optional)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Error starting live trading: {e}")
        print("Check:")
        print("- API credentials are correct")
        print("- Internet connection is stable") 
        print("- All required dependencies are installed")
        import traceback
        print("\nFull error details:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()