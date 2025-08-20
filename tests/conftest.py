import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test configuration
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'  # In-memory database for tests
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['DRY_RUN'] = 'true'


@pytest.fixture
def sample_config():
    """Sample trading configuration for tests."""
    return {
        "account": {
            "starting_equity": 1000.0,
            "max_positions": 5,
            "per_trade_risk_pct": 1.0,
            "daily_drawdown_pct": 3.0,
            "correlation_limit": 1
        },
        "strategies": {
            "fractional_breakout": {
                "enabled": True,
                "universe": ["AAPL", "MSFT"],
                "position_size_usd": 30,
                "atr_window": 14,
                "atr_stop_multiplier": 1.0,
                "atr_target_multiplier": 2.0
            }
        },
        "assets": {
            "min_volume": 100000,
            "max_bid_ask_spread_pct": 2.0
        }
    }


@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 150.0
    
    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    current_price = base_price
    for i in range(len(dates)):
        # Simple random walk with some volatility
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% daily vol
        
        open_price = current_price
        close_price = current_price * (1 + daily_return)
        
        # Intraday range
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        volume = np.random.randint(1000000, 5000000)
        
        data['open'].append(open_price)
        data['high'].append(high_price)
        data['low'].append(low_price)
        data['close'].append(close_price)
        data['volume'].append(volume)
        
        current_price = close_price
    
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def mock_alpaca_client(monkeypatch):
    """Mock Alpaca client for testing."""
    class MockAlpacaClient:
        def __init__(self):
            self.paper = True
        
        def get_account(self):
            return {
                'equity': 1000.0,
                'buying_power': 1000.0,
                'cash': 1000.0,
                'trading_blocked': False,
                'account_blocked': False
            }
        
        def get_positions(self):
            return []
        
        def get_orders(self, status=None):
            return []
        
        def get_stock_quotes(self, symbols):
            return {
                symbol: {
                    'bid_price': 150.0,
                    'ask_price': 150.1,
                    'bid_size': 100,
                    'ask_size': 100,
                    'timestamp': datetime.now()
                }
                for symbol in symbols
            }
        
        def get_asset_info(self, symbol):
            return {
                'symbol': symbol,
                'tradable': True,
                'fractionable': symbol in ['AAPL', 'MSFT', 'GOOGL']
            }
        
        def is_market_open(self):
            return True
    
    return MockAlpacaClient()


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    from state.database import DatabaseManager
    
    # Use in-memory SQLite database
    db = DatabaseManager('sqlite:///:memory:')
    return db


@pytest.fixture
def sample_strategy_signal():
    """Sample strategy signal for testing."""
    from strategies.base import StrategySignal, OrderType
    
    return StrategySignal(
        symbol="AAPL",
        signal_type='entry',
        action='buy',
        confidence=85.0,
        quantity=0.2,
        price=150.0,
        order_type=OrderType.MARKET,
        stop_loss=145.0,
        take_profit=160.0,
        metadata={
            'strategy': 'fractional_breakout',
            'test_signal': True
        }
    )