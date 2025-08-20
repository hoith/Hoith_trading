from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Trade(Base):
    """Trade execution record."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, index=True)
    strategy_name = Column(String(50), index=True)
    symbol = Column(String(10), index=True)
    side = Column(String(10))  # 'buy' or 'sell'
    quantity = Column(Float)
    price = Column(Float)
    filled_price = Column(Float, nullable=True)
    filled_quantity = Column(Float, nullable=True)
    order_type = Column(String(20))
    status = Column(String(20), index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    filled_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # P&L tracking
    realized_pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    
    # Metadata
    metadata_json = Column(Text)  # JSON string for strategy-specific data
    
    # Relationships
    strategy_run_id = Column(Integer, ForeignKey('strategy_runs.id'), nullable=True)
    strategy_run = relationship("StrategyRun", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(order_id='{self.order_id}', symbol='{self.symbol}', side='{self.side}', qty={self.quantity})>"


class Position(Base):
    """Current position record."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), index=True)
    symbol = Column(String(10), index=True)
    side = Column(String(10))  # 'long' or 'short'
    quantity = Column(Float)
    avg_entry_price = Column(Float)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    max_loss_limit = Column(Float, nullable=True)
    
    # Timestamps
    entry_time = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata_json = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', qty={self.quantity}, pnl={self.unrealized_pnl})>"


class StrategyRun(Base):
    """Strategy execution session record."""
    __tablename__ = 'strategy_runs'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), index=True)
    session_id = Column(String(50), index=True)  # Daily session identifier
    
    # Session info
    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(20), default='running', index=True)  # 'running', 'completed', 'stopped', 'error'
    
    # Performance metrics
    starting_equity = Column(Float)
    ending_equity = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    
    # Configuration snapshot
    config_snapshot = Column(Text)  # JSON string of strategy config
    
    # Relationships
    trades = relationship("Trade", back_populates="strategy_run")
    
    def __repr__(self):
        return f"<StrategyRun(strategy='{self.strategy_name}', session='{self.session_id}', trades={self.total_trades})>"


class PerformanceMetric(Base):
    """Daily performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    strategy_name = Column(String(50), index=True)
    
    # Daily metrics
    daily_pnl = Column(Float, default=0.0)
    daily_trades = Column(Integer, default=0)
    daily_wins = Column(Integer, default=0)
    daily_losses = Column(Integer, default=0)
    
    # Running totals
    cumulative_pnl = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    
    # Account metrics
    account_equity = Column(Float)
    active_positions = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<PerformanceMetric(date={self.date.date()}, strategy='{self.strategy_name}', pnl={self.daily_pnl})>"


class RiskEvent(Base):
    """Risk management events log."""
    __tablename__ = 'risk_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), index=True)  # 'breaker_triggered', 'limit_exceeded', etc.
    severity = Column(String(20), index=True)  # 'low', 'medium', 'high', 'critical'
    
    # Event details
    description = Column(Text)
    trigger_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    action_taken = Column(String(100))
    
    # Context
    strategy_name = Column(String(50), nullable=True, index=True)
    symbol = Column(String(10), nullable=True, index=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<RiskEvent(type='{self.event_type}', severity='{self.severity}')>"


# Create indexes for performance
Index('idx_trades_strategy_symbol_date', Trade.strategy_name, Trade.symbol, Trade.created_at)
Index('idx_positions_strategy_active', Position.strategy_name, Position.is_active)
Index('idx_performance_date_strategy', PerformanceMetric.date, PerformanceMetric.strategy_name)
Index('idx_risk_events_date_type', RiskEvent.created_at, RiskEvent.event_type)