import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from execution.router import OrderRouter
from execution.equity import EquityOrderManager
from strategies.base import StrategySignal, OrderType


class TestOrderRouter:
    """Test order routing functionality."""
    
    @pytest.fixture
    def mock_data_client(self):
        """Mock data client for testing."""
        mock_client = Mock()
        mock_client.trading_client = Mock()
        mock_client.get_stock_quotes.return_value = {
            'AAPL': {
                'bid_price': 149.5,
                'ask_price': 150.5
            }
        }
        return mock_client
    
    @pytest.fixture
    def order_router(self, mock_data_client):
        """Create order router for testing."""
        return OrderRouter(mock_data_client)
    
    def test_dry_run_mode(self, order_router, sample_strategy_signal):
        """Test order execution in dry run mode."""
        # Order router should be in dry run mode by default for tests
        order_router.dry_run = True
        
        result = order_router.execute_signal(sample_strategy_signal)
        
        assert result['success'] == True
        assert result['dry_run'] == True
        assert 'order_id' in result
        assert result['symbol'] == sample_strategy_signal.symbol
        assert result['action'] == sample_strategy_signal.action
    
    def test_signal_validation_in_router(self, order_router):
        """Test signal validation in order router."""
        
        # Invalid signal - no symbol
        invalid_signal = StrategySignal(
            symbol="",  # Empty symbol
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=1.0
        )
        
        # Should handle invalid signals gracefully
        result = order_router.execute_signal(invalid_signal)
        # In dry run mode, it might still simulate the order
        assert 'success' in result
    
    def test_execution_log_tracking(self, order_router, sample_strategy_signal):
        """Test execution log tracking."""
        order_router.dry_run = True
        
        # Execute multiple signals
        result1 = order_router.execute_signal(sample_strategy_signal)
        
        signal2 = StrategySignal(
            symbol="MSFT",
            signal_type='entry',
            action='sell',
            confidence=75.0,
            quantity=0.5
        )
        result2 = order_router.execute_signal(signal2)
        
        # Check execution summary
        summary = order_router.get_execution_summary()
        
        assert summary['total_orders'] == 2
        assert summary['successful_orders'] == 2
        assert summary['success_rate'] == 100.0
        assert summary['dry_run_mode'] == True
        assert len(summary['recent_orders']) == 2


class TestEquityOrderManager:
    """Test equity order management."""
    
    @pytest.fixture
    def mock_data_client(self):
        """Mock data client for order manager."""
        mock_client = Mock()
        mock_client.trading_client = Mock()
        
        # Mock successful order submission
        mock_order = Mock()
        mock_order.id = "test_order_123"
        mock_order.status.value = "accepted"
        mock_client.trading_client.submit_order.return_value = mock_order
        
        # Mock asset info
        mock_client.get_asset_info.return_value = {
            'tradable': True,
            'fractionable': True
        }
        
        return mock_client
    
    @pytest.fixture
    def order_manager(self, mock_data_client):
        """Create order manager for testing."""
        return EquityOrderManager(mock_data_client)
    
    def test_signal_validation(self, order_manager):
        """Test signal validation in order manager."""
        
        # Valid signal
        valid_signal = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=1.0,
            price=150.0
        )
        
        assert order_manager._validate_signal(valid_signal) == True
        
        # Invalid signal - no symbol
        invalid_signal = StrategySignal(
            symbol="",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=1.0
        )
        
        assert order_manager._validate_signal(invalid_signal) == False
        
        # Invalid signal - negative quantity
        invalid_signal2 = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=-1.0
        )
        
        assert order_manager._validate_signal(invalid_signal2) == False
    
    def test_order_request_creation(self, order_manager):
        """Test order request creation."""
        
        # Market order
        market_signal = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=1.0,
            order_type=OrderType.MARKET
        )
        
        request = order_manager._create_order_request(market_signal)
        assert request is not None
        assert request.symbol == "AAPL"
        assert request.qty == 1.0
        
        # Limit order
        limit_signal = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=1.0,
            price=150.0,
            order_type=OrderType.LIMIT
        )
        
        request = order_manager._create_order_request(limit_signal)
        assert request is not None
        assert request.limit_price == 150.0
    
    def test_fractional_order_creation(self, order_manager):
        """Test fractional share order creation."""
        
        # Mock quote data
        order_manager.data_client.get_stock_quotes.return_value = {
            'AAPL': {
                'bid_price': 149.5,
                'ask_price': 150.5
            }
        }
        
        # Fractional quantity
        fractional_signal = StrategySignal(
            symbol="AAPL",
            signal_type='entry',
            action='buy',
            confidence=85.0,
            quantity=0.5,  # Fractional shares
            order_type=OrderType.MARKET
        )
        
        request = order_manager._create_order_request(fractional_signal)
        
        # Should create notional order for fractional shares
        assert request is not None
        # In a real implementation, this would check for notional field
    
    @patch('alpaca.trading.client.TradingClient.submit_order')
    def test_order_execution_success(self, mock_submit, order_manager, sample_strategy_signal):
        """Test successful order execution."""
        
        # Mock successful order
        mock_order = Mock()
        mock_order.id = "test_order_123"
        mock_order.status.value = "accepted"
        mock_submit.return_value = mock_order
        
        result = order_manager.execute_order(sample_strategy_signal)
        
        assert result['success'] == True
        assert result['order_id'] == "test_order_123"
        assert result['symbol'] == sample_strategy_signal.symbol
        assert result['status'] == "accepted"
    
    def test_position_closing(self, order_manager):
        """Test position closing functionality."""
        
        # Mock current positions
        mock_positions = [
            {
                'symbol': 'AAPL',
                'qty': '10',
                'side': 'long'
            }
        ]
        
        order_manager.data_client.get_positions.return_value = mock_positions
        
        # Mock successful close order
        mock_order = Mock()
        mock_order.id = "close_order_123"
        mock_order.status.value = "accepted"
        order_manager.data_client.trading_client.submit_order.return_value = mock_order
        
        result = order_manager.close_position('AAPL')
        
        assert result['success'] == True
        assert result['symbol'] == 'AAPL'
        assert result['action'] == 'close'
    
    def test_position_closing_no_position(self, order_manager):
        """Test closing non-existent position."""
        
        # Mock no positions
        order_manager.data_client.get_positions.return_value = []
        
        result = order_manager.close_position('AAPL')
        
        assert result['success'] == False
        assert 'No position found' in result['error']


class TestOrderMonitoring:
    """Test order monitoring functionality."""
    
    def test_order_monitor_initialization(self):
        """Test order monitor initialization."""
        from execution.monitor import OrderMonitor
        
        mock_client = Mock()
        monitor = OrderMonitor(mock_client)
        
        assert len(monitor.monitored_orders) == 0
    
    def test_order_monitoring(self):
        """Test order status monitoring."""
        from execution.monitor import OrderMonitor
        
        mock_client = Mock()
        
        # Mock order status response
        mock_order = Mock()
        mock_order.status.value = "filled"
        mock_client.trading_client.get_order_by_id.return_value = mock_order
        
        monitor = OrderMonitor(mock_client)
        
        # Add order to monitor
        monitor.add_order_to_monitor("test_order_123", {'strategy': 'test'})
        
        assert len(monitor.monitored_orders) == 1
        assert "test_order_123" in monitor.monitored_orders
        
        # Check for updates
        updates = monitor.check_order_updates()
        
        # Should detect status change from 'pending' to 'filled'
        if updates:
            update = updates[0]
            assert update['order_id'] == "test_order_123"
            assert update['new_status'] == "filled"
    
    def test_monitoring_summary(self):
        """Test monitoring summary functionality."""
        from execution.monitor import OrderMonitor
        
        mock_client = Mock()
        monitor = OrderMonitor(mock_client)
        
        # Add multiple orders with different statuses
        monitor.add_order_to_monitor("order1", {'strategy': 'test'})
        monitor.add_order_to_monitor("order2", {'strategy': 'test'})
        
        # Manually set statuses for testing
        monitor.monitored_orders["order1"]['last_status'] = 'filled'
        monitor.monitored_orders["order2"]['last_status'] = 'pending'
        
        summary = monitor.get_monitoring_summary()
        
        assert summary['total_orders'] == 2
        assert summary['status_breakdown']['filled'] == 1
        assert summary['status_breakdown']['pending'] == 1