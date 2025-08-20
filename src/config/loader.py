import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from .schemas import TradingConfig


def load_config(config_path: str = "config.yml") -> TradingConfig:
    """Load and validate trading configuration from YAML file."""
    
    # Load environment variables
    load_dotenv()
    
    # Find config file
    if not os.path.isabs(config_path):
        # Look in current directory first, then parent directories
        current_dir = Path.cwd()
        config_file = None
        
        for path in [current_dir] + list(current_dir.parents):
            potential_file = path / config_path
            if potential_file.exists():
                config_file = potential_file
                break
        
        if not config_file:
            raise FileNotFoundError(f"Config file {config_path} not found")
    else:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_path} not found")
    
    # Load YAML configuration
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Override with environment variables where applicable
    config_data = _apply_env_overrides(config_data)
    
    # Validate and return config
    return TradingConfig(**config_data)


def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    
    # Account overrides
    if os.getenv('STARTING_EQUITY'):
        config_data['account']['starting_equity'] = float(os.getenv('STARTING_EQUITY'))
    
    if os.getenv('MAX_POSITIONS'):
        config_data['account']['max_positions'] = int(os.getenv('MAX_POSITIONS'))
    
    if os.getenv('PER_TRADE_RISK_PCT'):
        config_data['account']['per_trade_risk_pct'] = float(os.getenv('PER_TRADE_RISK_PCT'))
    
    if os.getenv('DAILY_DRAWDOWN_PCT'):
        config_data['account']['daily_drawdown_pct'] = float(os.getenv('DAILY_DRAWDOWN_PCT'))
    
    # Strategy toggles
    if os.getenv('IRON_CONDOR_ENABLED'):
        config_data['strategies']['iron_condor']['enabled'] = os.getenv('IRON_CONDOR_ENABLED').lower() == 'true'
    
    if os.getenv('MOMENTUM_VERTICAL_ENABLED'):
        config_data['strategies']['momentum_vertical']['enabled'] = os.getenv('MOMENTUM_VERTICAL_ENABLED').lower() == 'true'
    
    if os.getenv('FRACTIONAL_BREAKOUT_ENABLED'):
        config_data['strategies']['fractional_breakout']['enabled'] = os.getenv('FRACTIONAL_BREAKOUT_ENABLED').lower() == 'true'
    
    # Notifications
    if os.getenv('SLACK_WEBHOOK_URL'):
        config_data['notifications']['webhook_url'] = os.getenv('SLACK_WEBHOOK_URL')
        config_data['notifications']['enabled'] = True
    
    return config_data


def get_alpaca_config() -> Dict[str, str]:
    """Get Alpaca API configuration from environment variables."""
    load_dotenv()
    
    config = {
        'api_key': os.getenv('APCA_API_KEY_ID'),
        'secret_key': os.getenv('APCA_API_SECRET_KEY'),
        'base_url': os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
        'paper': os.getenv('PAPER_TRADING', 'true').lower() == 'true',
        'dry_run': os.getenv('DRY_RUN', 'false').lower() == 'true'
    }
    
    if not config['api_key'] or not config['secret_key']:
        raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
    
    return config