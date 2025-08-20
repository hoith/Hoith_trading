import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = None, 
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """Setup logging configuration for the trading system."""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    return logger


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)
    
    def log_entry(self, strategy: str, symbol: str, side: str, quantity: float, 
                  price: float, order_type: str = "market", **kwargs):
        """Log trade entry."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"ENTRY | {strategy} | {symbol} | {side} | {quantity} @ {price} | {order_type}"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_exit(self, strategy: str, symbol: str, side: str, quantity: float, 
                 price: float, pnl: float, **kwargs):
        """Log trade exit."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"EXIT | {strategy} | {symbol} | {side} | {quantity} @ {price} | PnL: {pnl:.2f}"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_rejection(self, strategy: str, symbol: str, reason: str, **kwargs):
        """Log order rejection."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.warning(
            f"REJECTION | {strategy} | {symbol} | {reason}"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_risk_breach(self, risk_type: str, current_value: float, 
                       limit: float, action: str, **kwargs):
        """Log risk management events."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.warning(
            f"RISK BREACH | {risk_type} | Current: {current_value} | Limit: {limit} | Action: {action}"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_strategy_performance(self, strategy: str, total_trades: int, 
                               win_rate: float, total_pnl: float, **kwargs):
        """Log strategy performance summary."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"PERFORMANCE | {strategy} | Trades: {total_trades} | Win Rate: {win_rate:.1f}% | PnL: {total_pnl:.2f}"
            + (f" | {extra_info}" if extra_info else "")
        )
    
    def log_market_event(self, event_type: str, description: str, **kwargs):
        """Log market-related events."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"MARKET | {event_type} | {description}"
            + (f" | {extra_info}" if extra_info else "")
        )


def get_daily_log_filename(base_name: str = "trading") -> str:
    """Generate daily log filename."""
    today = datetime.now().strftime("%Y%m%d")
    return f"{base_name}_{today}.log"


class StructuredLogger:
    """Logger that outputs structured data for analysis."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logger = logging.getLogger("structured")
        
        # Setup file handler for structured logs
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Simple formatter for structured data
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_trade(self, timestamp: datetime, strategy: str, symbol: str,
                  action: str, quantity: float, price: float, **kwargs):
        """Log structured trade data."""
        data = {
            'timestamp': timestamp.isoformat(),
            'strategy': strategy,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            **kwargs
        }
        
        # Convert to pipe-separated values for easy parsing
        values = [str(data[key]) for key in sorted(data.keys())]
        self.logger.info(" | ".join(values))
    
    def log_position(self, timestamp: datetime, symbol: str, quantity: float,
                    market_value: float, unrealized_pnl: float, **kwargs):
        """Log position data."""
        data = {
            'timestamp': timestamp.isoformat(),
            'type': 'position',
            'symbol': symbol,
            'quantity': quantity,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            **kwargs
        }
        
        values = [str(data[key]) for key in sorted(data.keys())]
        self.logger.info(" | ".join(values))