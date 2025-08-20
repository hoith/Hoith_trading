import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from sqlalchemy import create_engine, func, and_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, Trade, Position, StrategyRun, PerformanceMetric, RiskEvent
from strategies.base import StrategyPosition

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database operations for the trading system."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            connect_args={'check_same_thread': False} if 'sqlite' in self.database_url else {}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self.init_db()
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def init_db(self):
        """Initialize database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def record_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Record a trade execution."""
        with self.get_session() as session:
            try:
                trade = Trade(
                    order_id=trade_data.get('order_id'),
                    strategy_name=trade_data.get('strategy', 'unknown'),
                    symbol=trade_data.get('symbol'),
                    side=trade_data.get('action', trade_data.get('side')),
                    quantity=trade_data.get('quantity'),
                    price=trade_data.get('price'),
                    filled_price=trade_data.get('filled_price'),
                    filled_quantity=trade_data.get('filled_quantity'),
                    order_type=trade_data.get('order_type', 'market'),
                    status=trade_data.get('status', 'submitted'),
                    commission=trade_data.get('commission', 0.0),
                    metadata_json=json.dumps(trade_data.get('metadata', {}))
                )
                
                session.add(trade)
                session.commit()
                logger.info(f"Recorded trade: {trade}")
                return trade
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error recording trade: {e}")
                raise
    
    def update_trade_status(self, order_id: str, status: str, 
                           filled_price: float = None, filled_quantity: float = None):
        """Update trade status and fill information."""
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(Trade.order_id == order_id).first()
                if trade:
                    trade.status = status
                    if filled_price is not None:
                        trade.filled_price = filled_price
                    if filled_quantity is not None:
                        trade.filled_quantity = filled_quantity
                    if status == 'filled':
                        trade.filled_at = datetime.utcnow()
                    
                    session.commit()
                    logger.debug(f"Updated trade {order_id}: status={status}")
                else:
                    logger.warning(f"Trade not found: {order_id}")
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating trade status: {e}")
    
    def record_position(self, position: StrategyPosition) -> Position:
        """Record or update a position."""
        with self.get_session() as session:
            try:
                # Check if position already exists
                existing = session.query(Position).filter(
                    and_(
                        Position.strategy_name == position.strategy_name,
                        Position.symbol == position.symbol,
                        Position.is_active == True
                    )
                ).first()
                
                if existing:
                    # Update existing position
                    existing.quantity = position.quantity
                    existing.current_price = position.current_price
                    existing.market_value = position.market_value
                    existing.unrealized_pnl = position.unrealized_pnl
                    existing.stop_loss = position.stop_loss
                    existing.take_profit = position.take_profit
                    existing.updated_at = datetime.utcnow()
                    existing.metadata_json = json.dumps(position.metadata)
                    
                    session.commit()
                    return existing
                else:
                    # Create new position
                    pos = Position(
                        strategy_name=position.strategy_name,
                        symbol=position.symbol,
                        side=position.side.value,
                        quantity=position.quantity,
                        avg_entry_price=position.entry_price,
                        current_price=position.current_price,
                        market_value=position.market_value,
                        unrealized_pnl=position.unrealized_pnl,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        entry_time=position.entry_time,
                        metadata_json=json.dumps(position.metadata)
                    )
                    
                    session.add(pos)
                    session.commit()
                    logger.info(f"Recorded position: {pos}")
                    return pos
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error recording position: {e}")
                raise
    
    def close_position(self, strategy_name: str, symbol: str, realized_pnl: float = None):
        """Mark a position as closed."""
        with self.get_session() as session:
            try:
                position = session.query(Position).filter(
                    and_(
                        Position.strategy_name == strategy_name,
                        Position.symbol == symbol,
                        Position.is_active == True
                    )
                ).first()
                
                if position:
                    position.is_active = False
                    position.updated_at = datetime.utcnow()
                    if realized_pnl is not None:
                        # Store realized P&L in metadata
                        metadata = json.loads(position.metadata_json or '{}')
                        metadata['realized_pnl'] = realized_pnl
                        position.metadata_json = json.dumps(metadata)
                    
                    session.commit()
                    logger.info(f"Closed position: {strategy_name} {symbol}")
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error closing position: {e}")
    
    def get_active_positions(self, strategy_name: str = None) -> List[Position]:
        """Get active positions."""
        with self.get_session() as session:
            try:
                query = session.query(Position).filter(Position.is_active == True)
                
                if strategy_name:
                    query = query.filter(Position.strategy_name == strategy_name)
                
                return query.all()
                
            except SQLAlchemyError as e:
                logger.error(f"Error getting active positions: {e}")
                return []
    
    def start_strategy_run(self, strategy_name: str, session_id: str, 
                          starting_equity: float, config: Dict[str, Any]) -> StrategyRun:
        """Start a new strategy run session."""
        with self.get_session() as session:
            try:
                strategy_run = StrategyRun(
                    strategy_name=strategy_name,
                    session_id=session_id,
                    starting_equity=starting_equity,
                    config_snapshot=json.dumps(config)
                )
                
                session.add(strategy_run)
                session.commit()
                logger.info(f"Started strategy run: {strategy_name} [{session_id}]")
                return strategy_run
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error starting strategy run: {e}")
                raise
    
    def update_strategy_run(self, session_id: str, **kwargs):
        """Update strategy run metrics."""
        with self.get_session() as session:
            try:
                run = session.query(StrategyRun).filter(StrategyRun.session_id == session_id).first()
                if run:
                    for key, value in kwargs.items():
                        if hasattr(run, key):
                            setattr(run, key, value)
                    
                    session.commit()
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating strategy run: {e}")
    
    def record_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Record daily performance metrics."""
        with self.get_session() as session:
            try:
                # Check if metric for this date/strategy already exists
                existing = session.query(PerformanceMetric).filter(
                    and_(
                        PerformanceMetric.date == metric_data['date'],
                        PerformanceMetric.strategy_name == metric_data['strategy_name']
                    )
                ).first()
                
                if existing:
                    # Update existing metric
                    for key, value in metric_data.items():
                        if hasattr(existing, key) and key not in ['id', 'date', 'strategy_name']:
                            setattr(existing, key, value)
                    session.commit()
                    return existing
                else:
                    # Create new metric
                    metric = PerformanceMetric(**metric_data)
                    session.add(metric)
                    session.commit()
                    return metric
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error recording performance metric: {e}")
                raise
    
    def log_risk_event(self, event_type: str, severity: str, description: str,
                      trigger_value: float = None, threshold_value: float = None,
                      action_taken: str = None, strategy_name: str = None,
                      symbol: str = None) -> RiskEvent:
        """Log a risk management event."""
        with self.get_session() as session:
            try:
                event = RiskEvent(
                    event_type=event_type,
                    severity=severity,
                    description=description,
                    trigger_value=trigger_value,
                    threshold_value=threshold_value,
                    action_taken=action_taken,
                    strategy_name=strategy_name,
                    symbol=symbol
                )
                
                session.add(event)
                session.commit()
                logger.info(f"Logged risk event: {event_type} [{severity}]")
                return event
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error logging risk event: {e}")
                raise
    
    def get_trading_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get trading summary for recent period."""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow().date() - timedelta(days=days)
                
                # Get trade counts and P&L
                trades = session.query(Trade).filter(Trade.created_at >= cutoff_date).all()
                
                total_trades = len(trades)
                filled_trades = [t for t in trades if t.status == 'filled']
                total_pnl = sum(t.realized_pnl for t in filled_trades if t.realized_pnl)
                
                # Get active positions
                active_positions = session.query(Position).filter(Position.is_active == True).all()
                unrealized_pnl = sum(pos.unrealized_pnl for pos in active_positions if pos.unrealized_pnl)
                
                # Get recent performance metrics
                recent_metrics = session.query(PerformanceMetric).filter(
                    PerformanceMetric.date >= cutoff_date
                ).order_by(desc(PerformanceMetric.date)).limit(days).all()
                
                return {
                    'period_days': days,
                    'total_trades': total_trades,
                    'filled_trades': len(filled_trades),
                    'realized_pnl': total_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl': total_pnl + unrealized_pnl,
                    'active_positions': len(active_positions),
                    'recent_performance': [
                        {
                            'date': m.date.strftime('%Y-%m-%d'),
                            'strategy': m.strategy_name,
                            'daily_pnl': m.daily_pnl,
                            'cumulative_pnl': m.cumulative_pnl
                        }
                        for m in recent_metrics
                    ]
                }
                
            except SQLAlchemyError as e:
                logger.error(f"Error getting trading summary: {e}")
                return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to prevent database bloat."""
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                
                # Delete old completed trades
                old_trades = session.query(Trade).filter(
                    and_(
                        Trade.created_at < cutoff_date,
                        Trade.status.in_(['filled', 'cancelled', 'rejected'])
                    )
                ).delete()
                
                # Delete old risk events
                old_events = session.query(RiskEvent).filter(
                    RiskEvent.created_at < cutoff_date
                ).delete()
                
                session.commit()
                
                if old_trades or old_events:
                    logger.info(f"Cleaned up old data: {old_trades} trades, {old_events} risk events")
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error cleaning up old data: {e}")
    
    def export_trades_csv(self, file_path: str, strategy_name: str = None, days: int = 30):
        """Export trades to CSV file."""
        import pandas as pd
        
        with self.get_session() as session:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query = session.query(Trade).filter(Trade.created_at >= cutoff_date)
                
                if strategy_name:
                    query = query.filter(Trade.strategy_name == strategy_name)
                
                trades = query.all()
                
                if trades:
                    data = []
                    for trade in trades:
                        data.append({
                            'order_id': trade.order_id,
                            'strategy': trade.strategy_name,
                            'symbol': trade.symbol,
                            'side': trade.side,
                            'quantity': trade.quantity,
                            'price': trade.price,
                            'filled_price': trade.filled_price,
                            'status': trade.status,
                            'created_at': trade.created_at,
                            'filled_at': trade.filled_at,
                            'realized_pnl': trade.realized_pnl
                        })
                    
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, index=False)
                    logger.info(f"Exported {len(trades)} trades to {file_path}")
                else:
                    logger.info("No trades to export")
                    
            except Exception as e:
                logger.error(f"Error exporting trades: {e}")
                raise