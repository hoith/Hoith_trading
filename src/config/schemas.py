from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import time


class AccountConfig(BaseModel):
    starting_equity: float = Field(gt=0, description="Starting account equity")
    max_positions: int = Field(gt=0, le=20, description="Maximum concurrent positions")
    per_trade_risk_pct: float = Field(gt=0, le=5, description="Risk per trade as % of equity")
    daily_drawdown_pct: float = Field(gt=0, le=10, description="Daily loss breaker as % of equity")
    correlation_limit: int = Field(ge=0, le=5, description="Max correlated positions")


class IronCondorConfig(BaseModel):
    enabled: bool = True
    universe: List[str] = Field(min_items=1, description="Trading universe")
    iv_lookback: int = Field(gt=0, le=60, description="IV rank lookback period")
    delta_short: float = Field(gt=0, le=50, description="Short strike delta target")
    delta_long: float = Field(gt=0, le=50, description="Long strike delta target")
    spread_width: float = Field(gt=0, description="Spread width in dollars")
    min_credit: float = Field(gt=0, description="Minimum credit to receive")
    max_credit: float = Field(gt=0, description="Maximum credit to target")
    profit_target: float = Field(gt=0, le=1, description="Profit target as % of credit")
    max_loss_multiplier: float = Field(gt=1, description="Max loss as multiple of credit")
    max_dte: int = Field(gt=0, le=30, description="Maximum days to expiration")

    @validator('max_credit')
    def max_credit_gt_min(cls, v, values):
        if 'min_credit' in values and v <= values['min_credit']:
            raise ValueError('max_credit must be greater than min_credit')
        return v


class MomentumVerticalConfig(BaseModel):
    enabled: bool = True
    universe: List[str] = Field(min_items=1)
    spread_width: float = Field(gt=0)
    max_debit: float = Field(gt=0)
    profit_target: float = Field(gt=0, le=1)
    breakout_lookback: int = Field(gt=0, le=60)
    volume_threshold: int = Field(gt=0)


class FractionalBreakoutConfig(BaseModel):
    enabled: bool = True
    universe: List[str] = Field(min_items=1)
    position_size_usd: float = Field(gt=0, le=100)
    atr_window: int = Field(gt=0, le=60)
    atr_stop_multiplier: float = Field(gt=0)
    atr_target_multiplier: float = Field(gt=0)
    breakout_lookback: int = Field(gt=0, le=60)
    min_volume: int = Field(gt=0)


class StrategiesConfig(BaseModel):
    iron_condor: IronCondorConfig
    momentum_vertical: MomentumVerticalConfig
    fractional_breakout: FractionalBreakoutConfig


class AssetsConfig(BaseModel):
    min_volume: int = Field(gt=0)
    max_bid_ask_spread_pct: float = Field(gt=0, le=10)
    options_min_volume: int = Field(gt=0)
    options_max_bid_ask_spread: float = Field(gt=0)


class SlippageConfig(BaseModel):
    equity_bps: float = Field(ge=0, le=100)
    options_cents: float = Field(ge=0)
    options_pct: float = Field(ge=0, le=10)


class FeesConfig(BaseModel):
    equity_per_share: float = Field(ge=0)
    options_per_contract: float = Field(ge=0)
    regulatory_fee_pct: float = Field(ge=0)


class SchedulingConfig(BaseModel):
    market_open: str = "09:30"
    entry_time: str = "09:35"
    monitor_start: str = "10:00"
    monitor_end: str = "15:45"
    cleanup_time: str = "15:50"

    @validator('market_open', 'entry_time', 'monitor_start', 'monitor_end', 'cleanup_time')
    def validate_time_format(cls, v):
        try:
            time.fromisoformat(v)
        except ValueError:
            raise ValueError(f'Invalid time format: {v}. Use HH:MM format.')
        return v


class NotificationsConfig(BaseModel):
    enabled: bool = False
    webhook_url: Optional[str] = None
    notify_on: List[str] = ['fills', 'rejections', 'breaker_triggered', 'daily_summary']


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = Field(gt=0)
    commission_equity: float = Field(ge=0)
    commission_options: float = Field(ge=0)
    slippage_equity_bps: float = Field(ge=0)
    slippage_options_pct: float = Field(ge=0)


class TradingConfig(BaseModel):
    account: AccountConfig
    strategies: StrategiesConfig
    assets: AssetsConfig
    slippage: SlippageConfig
    fees: FeesConfig
    scheduling: SchedulingConfig
    notifications: NotificationsConfig
    backtest: BacktestConfig

    class Config:
        extra = "forbid"  # Prevent unexpected fields