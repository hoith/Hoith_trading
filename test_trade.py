#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.loader import load_config, get_alpaca_config
from data.alpaca_client import AlpacaDataClient
from execution.router import OrderRouter
from strategies.fractional_breakout import FractionalBreakoutStrategy
from strategies.base import StrategySignal, OrderType
from risk.sizing import PositionSizer
from utils.logging import setup_logging


def test_simple_trade():
    """Test a simple fractional equity trade."""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Trading System Test ===")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        alpaca_config = get_alpaca_config()
        
        logger.info(f"Paper trading: {alpaca_config['paper']}")
        logger.info(f"Dry run: {alpaca_config['dry_run']}")
        
        # Initialize data client
        logger.info("Initializing Alpaca client...")
        data_client = AlpacaDataClient()
        
        # Test account access
        logger.info("Testing account access...")
        account = data_client.get_account()
        logger.info(f"Account equity: ${account['equity']:,.2f}")
        
        # Initialize order router
        logger.info("Initializing order router...")
        order_router = OrderRouter(data_client)
        
        # Initialize position sizer
        position_sizer = PositionSizer(
            account_equity=float(account['equity']),
            max_positions=5,
            risk_per_trade_pct=1.0
        )
        
        # Create a test signal for AAPL fractional purchase
        logger.info("Creating test signal...")
        test_signal = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=0.1,  # Buy $30 worth (will be calculated)
            order_type=OrderType.MARKET,
            metadata={
                'strategy': 'fractional_breakout',
                'position_size_usd': 30,
                'test_trade': True
            }
        )
        
        # Get current price
        logger.info("Getting current AAPL price...")
        quotes = data_client.get_stock_quotes(['AAPL'])
        if 'AAPL' in quotes:
            current_price = (quotes['AAPL']['bid_price'] + quotes['AAPL']['ask_price']) / 2
            logger.info(f"Current AAPL price: ${current_price:.2f}")
            
            # Calculate exact fractional quantity for $30
            test_signal.quantity = round(30 / current_price, 6)
            test_signal.price = current_price
            
            logger.info(f"Will buy {test_signal.quantity} shares for ~$30")
        else:
            logger.error("Could not get AAPL quote")
            return False
        
        # Execute the trade
        logger.info("Executing test trade...")
        result = order_router.execute_signal(test_signal)
        
        # Log results
        logger.info("=== Trade Execution Result ===")
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        
        if result.get('success'):
            logger.info("✅ Trade executed successfully!")
            
            if result.get('dry_run'):
                logger.info("📝 This was a dry run - no actual trade was placed")
            else:
                logger.info("💰 Real trade was placed!")
                order_id = result.get('order_id')
                if order_id:
                    logger.info(f"Order ID: {order_id}")
        else:
            logger.error("❌ Trade execution failed!")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Get execution summary
        summary = order_router.get_execution_summary()
        logger.info("=== Execution Summary ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        return result.get('success', False)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_account_info():
    """Test basic account information retrieval."""
    
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing account information...")
        
        # Initialize client
        data_client = AlpacaDataClient()
        
        # Get account info
        account = data_client.get_account()
        logger.info("=== Account Information ===")
        for key, value in account.items():
            if isinstance(value, float):
                logger.info(f"{key}: ${value:,.2f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Get positions
        positions = data_client.get_positions()
        logger.info(f"\n=== Positions ({len(positions)}) ===")
        for pos in positions:
            logger.info(f"{pos['symbol']}: {pos['qty']} shares, value: ${pos['market_value']:,.2f}")
        
        # Test market status
        is_open = data_client.is_market_open()
        logger.info(f"\nMarket is {'OPEN' if is_open else 'CLOSED'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Account test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Alpaca Trading System Test")
    print("=" * 50)
    
    # First test account access
    print("\n1. Testing account access...")
    if not test_account_info():
        print("❌ Account test failed!")
        sys.exit(1)
    
    print("✅ Account test passed!")
    
    # Then test trade execution
    print("\n2. Testing trade execution...")
    if test_simple_trade():
        print("✅ All tests passed!")
    else:
        print("❌ Trade test failed!")
        sys.exit(1)