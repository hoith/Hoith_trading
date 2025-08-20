import pytest
import tempfile
import yaml
from pathlib import Path

from config.loader import load_config
from config.schemas import TradingConfig


def test_config_loading():
    """Test configuration loading and validation."""
    
    # Create a temporary config file
    config_data = {
        'account': {
            'starting_equity': 1000.0,
            'max_positions': 5,
            'per_trade_risk_pct': 1.0,
            'daily_drawdown_pct': 3.0,
            'correlation_limit': 1
        },
        'strategies': {
            'iron_condor': {
                'enabled': True,
                'universe': ['SPY', 'QQQ'],
                'iv_lookback': 20,
                'delta_short': 15,
                'delta_long': 10,
                'spread_width': 1.0,
                'min_credit': 0.20,
                'max_credit': 0.40,
                'profit_target': 0.5,
                'max_loss_multiplier': 1.3,
                'max_dte': 2
            },
            'momentum_vertical': {
                'enabled': False,
                'universe': ['AAPL'],
                'spread_width': 1.0,
                'max_debit': 0.35,
                'profit_target': 0.8,
                'breakout_lookback': 20,
                'volume_threshold': 1000000
            },
            'fractional_breakout': {
                'enabled': True,
                'universe': ['AAPL', 'MSFT'],
                'position_size_usd': 30,
                'atr_window': 14,
                'atr_stop_multiplier': 1.0,
                'atr_target_multiplier': 2.0,
                'breakout_lookback': 20,
                'min_volume': 500000
            }
        },
        'assets': {
            'min_volume': 100000,
            'max_bid_ask_spread_pct': 2.0,
            'options_min_volume': 50,
            'options_max_bid_ask_spread': 0.10
        },
        'slippage': {
            'equity_bps': 5,
            'options_cents': 0.02,
            'options_pct': 1.0
        },
        'fees': {
            'equity_per_share': 0.0,
            'options_per_contract': 0.65,
            'regulatory_fee_pct': 0.0001
        },
        'scheduling': {
            'market_open': '09:30',
            'entry_time': '09:35',
            'monitor_start': '10:00',
            'monitor_end': '15:45',
            'cleanup_time': '15:50'
        },
        'notifications': {
            'enabled': False,
            'webhook_url': None,
            'notify_on': ['fills', 'rejections']
        },
        'backtest': {
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 1000.0,
            'commission_equity': 0.0,
            'commission_options': 0.65,
            'slippage_equity_bps': 5,
            'slippage_options_pct': 1.0
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        # Load and validate configuration
        config = load_config(temp_path)
        
        # Assertions
        assert isinstance(config, TradingConfig)
        assert config.account.starting_equity == 1000.0
        assert config.account.max_positions == 5
        assert config.strategies.iron_condor.enabled == True
        assert config.strategies.momentum_vertical.enabled == False
        assert config.strategies.fractional_breakout.enabled == True
        assert len(config.strategies.fractional_breakout.universe) == 2
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_config_validation_errors():
    """Test configuration validation catches errors."""
    
    # Test invalid starting equity
    invalid_config = {
        'account': {
            'starting_equity': -1000.0,  # Invalid: negative
            'max_positions': 5,
            'per_trade_risk_pct': 1.0,
            'daily_drawdown_pct': 3.0,
            'correlation_limit': 1
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(invalid_config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(Exception):  # Should raise validation error
            load_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_iron_condor_config_validation():
    """Test Iron Condor specific validation."""
    from config.schemas import IronCondorConfig
    
    # Valid config
    valid_config = {
        'enabled': True,
        'universe': ['SPY'],
        'iv_lookback': 20,
        'delta_short': 15,
        'delta_long': 10,
        'spread_width': 1.0,
        'min_credit': 0.20,
        'max_credit': 0.40,
        'profit_target': 0.5,
        'max_loss_multiplier': 1.3,
        'max_dte': 2
    }
    
    config = IronCondorConfig(**valid_config)
    assert config.enabled == True
    assert config.min_credit == 0.20
    assert config.max_credit == 0.40
    
    # Invalid config - max_credit <= min_credit
    invalid_config = valid_config.copy()
    invalid_config['max_credit'] = 0.15  # Less than min_credit
    
    with pytest.raises(Exception):
        IronCondorConfig(**invalid_config)


def test_time_validation():
    """Test time format validation in scheduling."""
    from config.schemas import SchedulingConfig
    
    # Valid times
    valid_config = {
        'market_open': '09:30',
        'entry_time': '09:35',
        'monitor_start': '10:00',
        'monitor_end': '15:45',
        'cleanup_time': '15:50'
    }
    
    config = SchedulingConfig(**valid_config)
    assert config.entry_time == '09:35'
    
    # Invalid time format
    invalid_config = valid_config.copy()
    invalid_config['entry_time'] = '9:35'  # Missing leading zero
    
    # This should still work as time.fromisoformat accepts this
    config = SchedulingConfig(**invalid_config)
    assert config.entry_time == '9:35'
    
    # Truly invalid time
    invalid_config['entry_time'] = 'invalid_time'
    with pytest.raises(Exception):
        SchedulingConfig(**invalid_config)