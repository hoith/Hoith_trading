#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / 'src')
sys.path.insert(0, src_path)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_connection():
    """Test basic Alpaca connection without complex imports."""
    
    logger.info("=== Simple Alpaca Connection Test ===")
    
    try:
        # Import and test basic connection
        from alpaca.trading.client import TradingClient
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Get API credentials
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            logger.error("Missing API credentials in .env file")
            return False
        
        logger.info(f"Using API Key: {api_key[:8]}...")
        logger.info(f"Base URL: {base_url}")
        
        # Initialize client
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True  # Force paper trading for safety
        )
        
        # Test account access
        logger.info("Testing account access...")
        account = client.get_account()
        
        logger.info("=== Account Information ===")
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Status: {account.status}")
        logger.info(f"Equity: ${float(account.equity):,.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Day Trade Count: {getattr(account, 'daytrade_count', getattr(account, 'day_trade_count', 'N/A'))}")
        
        # Test positions
        logger.info("\n=== Current Positions ===")
        positions = client.get_all_positions()
        
        if positions:
            for pos in positions:
                logger.info(f"{pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
                logger.info(f"  Market Value: ${float(pos.market_value):,.2f}")
                logger.info(f"  P&L: ${float(pos.unrealized_pl):,.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")
        else:
            logger.info("No current positions")
        
        # Test market status
        logger.info("\n=== Market Status ===")
        clock = client.get_clock()
        logger.info(f"Market Open: {clock.is_open}")
        logger.info(f"Next Open: {clock.next_open}")
        logger.info(f"Next Close: {clock.next_close}")
        
        logger.info("\nConnection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_market_order():
    """Test placing a simple market order in dry run mode."""
    
    logger.info("\n=== Simple Market Order Test ===")
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Initialize clients
        trading_client = TradingClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            paper=True
        )
        
        data_client = StockHistoricalDataClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY')
        )
        
        # Get current AAPL price
        logger.info("Getting AAPL quote...")
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=["AAPL"])
        quotes = data_client.get_stock_latest_quote(quote_request)
        
        if "AAPL" in quotes:
            aapl_quote = quotes["AAPL"]
            current_price = (aapl_quote.bid_price + aapl_quote.ask_price) / 2
            logger.info(f"AAPL Price: ${current_price:.2f} (Bid: ${aapl_quote.bid_price}, Ask: ${aapl_quote.ask_price})")
        else:
            logger.error("Could not get AAPL quote")
            return False
        
        # Check if we want to place a real order or just simulate
        dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual order will be placed")
            
            # Simulate order
            quantity = round(30 / current_price, 6)  # $30 worth
            logger.info(f"Would place order: BUY {quantity} AAPL @ market price")
            logger.info(f"Estimated cost: ${quantity * current_price:.2f}")
            
            # Create order object for demonstration
            order_request = MarketOrderRequest(
                symbol="AAPL",
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            logger.info(f"Order request created: {order_request}")
            logger.info("Dry run order simulation successful!")
            
        else:
            logger.info("LIVE MODE - Placing real order!")
            
            # Calculate fractional shares for $30
            target_amount = 30.0
            quantity = round(target_amount / current_price, 6)
            
            logger.info(f"Placing order: BUY {quantity} AAPL (${target_amount} worth)")
            
            # Create market order
            order_request = MarketOrderRequest(
                symbol="AAPL",
                notional=target_amount,  # Use notional for fractional shares
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = trading_client.submit_order(order_request)
            
            logger.info(f"Order submitted successfully!")
            logger.info(f"Order ID: {order.id}")
            logger.info(f"Status: {order.status}")
            logger.info(f"Symbol: {order.symbol}")
            logger.info(f"Side: {order.side}")
            logger.info(f"Quantity: {order.qty}")
            
        return True
        
    except Exception as e:
        logger.error(f"Order test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Simple Alpaca Trading Test")
    print("=" * 50)
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("Basic connection failed!")
        sys.exit(1)
    
    # Test 2: Simple order
    print("\n" + "=" * 50)
    if test_simple_market_order():
        print("\nAll tests passed!")
    else:
        print("\nOrder test failed!")
        sys.exit(1)