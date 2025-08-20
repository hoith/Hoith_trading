import pytest
from datetime import datetime, timedelta

from risk.sizing import PositionSizer, SizingMethod
from risk.breakers import RiskBreaker, BreakerType, BreakerConfig
from risk.correlation import CorrelationManager, CorrelationGroup
from strategies.base import StrategySignal, OrderType


class TestPositionSizer:
    """Test position sizing functionality."""
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        sizer = PositionSizer(
            account_equity=10000.0,
            max_positions=5,
            risk_per_trade_pct=2.0
        )
        
        assert sizer.account_equity == 10000.0
        assert sizer.max_positions == 5
        assert sizer.risk_per_trade_pct == 2.0
        assert sizer.max_position_value == 2000.0  # 20% of equity
    
    def test_percent_equity_sizing(self, sample_strategy_signal):
        """Test percentage of equity position sizing."""
        sizer = PositionSizer(account_equity=1000.0, risk_per_trade_pct=1.0)
        
        # Test with stop loss
        sample_strategy_signal.price = 100.0
        sample_strategy_signal.stop_loss = 95.0  # 5% stop loss
        
        result = sizer.calculate_position_size(
            sample_strategy_signal,
            method=SizingMethod.PERCENT_EQUITY
        )
        
        assert result['valid'] == True
        assert result['method'] == 'percent_equity'
        # Risk amount should be $10 (1% of $1000)
        # Risk per share is $5 (100 - 95)
        # So quantity should be 2 shares
        assert result['quantity'] == 2.0
        assert result['actual_risk'] == 10.0
    
    def test_fixed_dollar_sizing(self, sample_strategy_signal):
        """Test fixed dollar amount sizing."""
        sizer = PositionSizer(account_equity=1000.0)
        
        sample_strategy_signal.price = 50.0
        sample_strategy_signal.metadata['position_size_usd'] = 100.0
        
        result = sizer.calculate_position_size(
            sample_strategy_signal,
            method=SizingMethod.FIXED_DOLLAR,
            fixed_amount=100.0
        )
        
        assert result['valid'] == True
        assert result['method'] == 'fixed_dollar'
        assert result['quantity'] == 2.0  # $100 / $50 = 2 shares
        assert result['position_value'] == 100.0
    
    def test_max_position_limit(self, sample_strategy_signal):
        """Test maximum position limit enforcement."""
        sizer = PositionSizer(
            account_equity=1000.0,
            max_positions=2,
            risk_per_trade_pct=1.0
        )
        
        # Should work with 1 current position
        result = sizer.calculate_position_size(
            sample_strategy_signal,
            current_positions=1
        )
        assert result['valid'] == True
        
        # Should fail with max positions reached
        result = sizer.calculate_position_size(
            sample_strategy_signal,
            current_positions=2
        )
        assert result['valid'] == False
        assert result['reason'] == 'max_positions_reached'
    
    def test_position_validation(self):
        """Test position size validation."""
        sizer = PositionSizer(account_equity=1000.0, max_positions=5)
        
        current_positions = {}
        
        # Valid position
        validation = sizer.validate_position_size(
            symbol="AAPL",
            quantity=1.0,
            price=100.0,
            current_positions=current_positions
        )
        assert validation['valid'] == True
        assert len(validation['errors']) == 0
        
        # Position too large
        validation = sizer.validate_position_size(
            symbol="AAPL",
            quantity=30.0,  # $3000 position on $1000 account
            price=100.0,
            current_positions=current_positions
        )
        assert validation['valid'] == False
        assert len(validation['errors']) > 0


class TestRiskBreaker:
    """Test risk breaker functionality."""
    
    def test_risk_breaker_initialization(self):
        """Test risk breaker initialization."""
        breaker = RiskBreaker(account_equity=1000.0)
        
        assert breaker.account_equity == 1000.0
        assert breaker.session_start_equity == 1000.0
        assert len(breaker.breakers) > 0  # Should have default breakers
        
        # Check for default breakers
        assert BreakerType.DAILY_LOSS in breaker.breakers
        assert BreakerType.DRAWDOWN in breaker.breakers
    
    def test_daily_loss_breaker(self):
        """Test daily loss breaker."""
        breaker = RiskBreaker(account_equity=1000.0)
        
        # Simulate daily loss
        current_equity = 950.0  # $50 loss (5% of equity)
        current_positions = {}
        
        breaker_status = breaker.check_all_breakers(current_equity, current_positions)
        
        # Should trigger daily loss breaker (3% threshold)
        assert BreakerType.DAILY_LOSS in breaker_status
        assert breaker_status[BreakerType.DAILY_LOSS] == True
        assert breaker.can_enter_new_position() == False
    
    def test_consecutive_losses(self):
        """Test consecutive losses breaker."""
        breaker = RiskBreaker(account_equity=1000.0)
        
        # Simulate 6 consecutive losing trades
        for i in range(6):
            breaker.update_trade_result(-10.0)  # $10 loss each
        
        assert breaker.consecutive_losses == 6
        
        # Check breakers
        breaker_status = breaker.check_all_breakers(1000.0, {})
        assert BreakerType.CONSECUTIVE_LOSSES in breaker_status
        assert breaker_status[BreakerType.CONSECUTIVE_LOSSES] == True
    
    def test_breaker_cooldown(self):
        """Test breaker cooldown mechanism."""
        breaker = RiskBreaker(account_equity=1000.0)
        
        # Trigger a breaker
        current_equity = 950.0
        breaker.check_all_breakers(current_equity, {})
        
        # Should be in cooldown
        assert not breaker.can_enter_new_position()
        
        # Check if breaker is in cooldown
        assert breaker.is_breaker_active(BreakerType.DAILY_LOSS)
    
    def test_custom_breaker(self):
        """Test adding custom risk breaker."""
        breaker = RiskBreaker(account_equity=1000.0)
        
        # Add custom breaker
        custom_breaker = BreakerConfig(
            breaker_type=BreakerType.TOTAL_LOSS,
            threshold=100.0,  # $100 total loss threshold
            action="close_all_positions"
        )
        
        breaker.add_breaker(custom_breaker)
        assert BreakerType.TOTAL_LOSS in breaker.breakers
        assert breaker.breakers[BreakerType.TOTAL_LOSS].threshold == 100.0


class TestCorrelationManager:
    """Test correlation management."""
    
    def test_correlation_manager_initialization(self):
        """Test correlation manager initialization."""
        manager = CorrelationManager(max_total_positions=5)
        
        assert manager.max_total_positions == 5
        assert len(manager.correlation_groups) > 0
        
        # Check for predefined groups
        assert 'tech_mega_caps' in manager.correlation_groups
        assert 'broad_index' in manager.correlation_groups
    
    def test_position_limits(self):
        """Test correlation-based position limits."""
        manager = CorrelationManager(max_total_positions=5)
        
        current_positions = {}
        
        # Should allow first tech position
        can_add, reason = manager.can_add_position("AAPL", current_positions)
        assert can_add == True
        
        # Add AAPL position
        current_positions["AAPL"] = {"qty": 10}
        
        # Should allow second tech position (limit is 2)
        can_add, reason = manager.can_add_position("MSFT", current_positions)
        assert can_add == True
        
        # Add MSFT position
        current_positions["MSFT"] = {"qty": 5}
        
        # Should reject third tech position
        can_add, reason = manager.can_add_position("GOOGL", current_positions)
        assert can_add == False
        assert "tech_mega_caps" in reason
    
    def test_index_correlation_limit(self):
        """Test index ETF correlation limits."""
        manager = CorrelationManager()
        
        current_positions = {"SPY": {"qty": 100}}
        
        # Should reject second index position (limit is 1)
        can_add, reason = manager.can_add_position("QQQ", current_positions)
        assert can_add == False
        assert "broad_index" in reason
    
    def test_total_position_limit(self):
        """Test total position limit."""
        manager = CorrelationManager(max_total_positions=2)
        
        current_positions = {"AAPL": {"qty": 10}, "XOM": {"qty": 5}}
        
        # Should reject new position (at max)
        can_add, reason = manager.can_add_position("GOOGL", current_positions)
        assert can_add == False
        assert "Maximum total positions reached" in reason
    
    def test_custom_correlation_group(self):
        """Test adding custom correlation group."""
        manager = CorrelationManager()
        
        # Add custom group
        custom_group = CorrelationGroup(
            name="test_group",
            symbols={"TEST1", "TEST2", "TEST3"},
            max_positions=1
        )
        
        manager.add_correlation_group(custom_group)
        
        assert 'test_group' in manager.correlation_groups
        
        # Test the custom group limit
        current_positions = {"TEST1": {"qty": 10}}
        can_add, reason = manager.can_add_position("TEST2", current_positions)
        assert can_add == False
    
    def test_portfolio_analysis(self):
        """Test portfolio correlation analysis."""
        manager = CorrelationManager()
        
        # Mock correlation matrix
        import pandas as pd
        import numpy as np
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        corr_matrix = pd.DataFrame(
            [[1.0, 0.8, 0.7],
             [0.8, 1.0, 0.9],
             [0.7, 0.9, 1.0]],
            index=symbols,
            columns=symbols
        )
        
        manager.correlation_matrix = corr_matrix
        
        current_positions = {"AAPL": {"qty": 10}, "MSFT": {"qty": 5}}
        
        analysis = manager.analyze_portfolio_correlation(current_positions)
        
        assert 'avg_correlation' in analysis
        assert 'max_correlation' in analysis
        assert 'diversification_ratio' in analysis
        assert analysis['avg_correlation'] == 0.8  # Correlation between AAPL and MSFT