import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BreakerType(Enum):
    DAILY_LOSS = "daily_loss"
    TOTAL_LOSS = "total_loss"
    POSITION_LOSS = "position_loss"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"


class BreakerStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    DISABLED = "disabled"


@dataclass
class BreakerConfig:
    """Configuration for a risk breaker."""
    breaker_type: BreakerType
    threshold: float
    enabled: bool = True
    cooldown_minutes: int = 60
    action: str = "block_new_entries"  # "block_new_entries", "close_all_positions", "reduce_size"
    notification: bool = True


class RiskBreaker:
    """Risk management circuit breakers."""
    
    def __init__(self, account_equity: float):
        self.account_equity = account_equity
        self.breakers: Dict[BreakerType, BreakerConfig] = {}
        self.triggered_breakers: Dict[BreakerType, datetime] = {}
        self.daily_pnl = 0.0
        self.session_start_equity = account_equity
        self.session_peak_equity = account_equity
        self.consecutive_losses = 0
        self.last_trade_pnl = 0.0
        
        # Callbacks for breaker events
        self.on_breaker_triggered: List[Callable] = []
        self.on_breaker_reset: List[Callable] = []
        
        # Setup default breakers
        self._setup_default_breakers()
        
        logger.info(f"Initialized RiskBreaker with equity ${account_equity:,.2f}")
    
    def _setup_default_breakers(self):
        """Setup default risk breakers."""
        
        # Daily loss breaker (3% of equity)
        self.add_breaker(BreakerConfig(
            breaker_type=BreakerType.DAILY_LOSS,
            threshold=self.account_equity * 0.03,  # 3% of equity
            action="block_new_entries",
            cooldown_minutes=60
        ))
        
        # Maximum drawdown breaker (10% from peak)
        self.add_breaker(BreakerConfig(
            breaker_type=BreakerType.DRAWDOWN,
            threshold=0.10,  # 10% drawdown
            action="reduce_size",
            cooldown_minutes=120
        ))
        
        # Consecutive losses breaker
        self.add_breaker(BreakerConfig(
            breaker_type=BreakerType.CONSECUTIVE_LOSSES,
            threshold=5,  # 5 consecutive losses
            action="block_new_entries",
            cooldown_minutes=240  # 4 hours
        ))
        
        # Single position loss breaker (5% of equity)
        self.add_breaker(BreakerConfig(
            breaker_type=BreakerType.POSITION_LOSS,
            threshold=self.account_equity * 0.05,  # 5% of equity
            action="close_all_positions",
            cooldown_minutes=30
        ))
    
    def add_breaker(self, config: BreakerConfig):
        """Add a risk breaker."""
        self.breakers[config.breaker_type] = config
        logger.info(f"Added {config.breaker_type.value} breaker: threshold={config.threshold}")
    
    def remove_breaker(self, breaker_type: BreakerType):
        """Remove a risk breaker."""
        if breaker_type in self.breakers:
            del self.breakers[breaker_type]
            logger.info(f"Removed {breaker_type.value} breaker")
    
    def check_all_breakers(self, current_equity: float, 
                          current_positions: Dict[str, Any]) -> Dict[BreakerType, bool]:
        """Check all breakers and return status."""
        
        breaker_status = {}
        
        for breaker_type, config in self.breakers.items():
            if not config.enabled:
                continue
            
            # Skip if breaker is in cooldown
            if self._is_in_cooldown(breaker_type):
                continue
            
            triggered = self._check_individual_breaker(
                breaker_type, config, current_equity, current_positions
            )
            
            breaker_status[breaker_type] = triggered
            
            if triggered:
                self._trigger_breaker(breaker_type, config)
        
        return breaker_status
    
    def _check_individual_breaker(self, breaker_type: BreakerType, 
                                 config: BreakerConfig, current_equity: float,
                                 current_positions: Dict[str, Any]) -> bool:
        """Check individual breaker condition."""
        
        if breaker_type == BreakerType.DAILY_LOSS:
            return self._check_daily_loss_breaker(config, current_equity)
        
        elif breaker_type == BreakerType.DRAWDOWN:
            return self._check_drawdown_breaker(config, current_equity)
        
        elif breaker_type == BreakerType.CONSECUTIVE_LOSSES:
            return self._check_consecutive_losses_breaker(config)
        
        elif breaker_type == BreakerType.POSITION_LOSS:
            return self._check_position_loss_breaker(config, current_positions)
        
        elif breaker_type == BreakerType.TOTAL_LOSS:
            return self._check_total_loss_breaker(config, current_equity)
        
        elif breaker_type == BreakerType.VOLATILITY:
            return self._check_volatility_breaker(config, current_equity)
        
        return False
    
    def _check_daily_loss_breaker(self, config: BreakerConfig, current_equity: float) -> bool:
        """Check daily loss breaker."""
        daily_loss = self.session_start_equity - current_equity
        return daily_loss > config.threshold
    
    def _check_drawdown_breaker(self, config: BreakerConfig, current_equity: float) -> bool:
        """Check drawdown breaker."""
        drawdown = (self.session_peak_equity - current_equity) / self.session_peak_equity
        return drawdown > config.threshold
    
    def _check_consecutive_losses_breaker(self, config: BreakerConfig) -> bool:
        """Check consecutive losses breaker."""
        return self.consecutive_losses >= config.threshold
    
    def _check_position_loss_breaker(self, config: BreakerConfig, 
                                   current_positions: Dict[str, Any]) -> bool:
        """Check if any single position has excessive loss."""
        
        for symbol, position in current_positions.items():
            unrealized_loss = position.get('unrealized_pl', 0)
            if unrealized_loss < -config.threshold:  # Negative P&L
                logger.warning(f"Position {symbol} loss ${abs(unrealized_loss):,.2f} exceeds threshold")
                return True
        
        return False
    
    def _check_total_loss_breaker(self, config: BreakerConfig, current_equity: float) -> bool:
        """Check total loss from starting equity."""
        total_loss = self.session_start_equity - current_equity
        return total_loss > config.threshold
    
    def _check_volatility_breaker(self, config: BreakerConfig, current_equity: float) -> bool:
        """Check volatility-based breaker (placeholder for future implementation)."""
        # This could track equity volatility over time
        return False
    
    def _trigger_breaker(self, breaker_type: BreakerType, config: BreakerConfig):
        """Trigger a risk breaker."""
        
        self.triggered_breakers[breaker_type] = datetime.now()
        
        logger.critical(f"RISK BREAKER TRIGGERED: {breaker_type.value} - Action: {config.action}")
        
        # Execute callbacks
        for callback in self.on_breaker_triggered:
            try:
                callback(breaker_type, config)
            except Exception as e:
                logger.error(f"Error in breaker callback: {e}")
        
        # Send notification if enabled
        if config.notification:
            self._send_notification(breaker_type, config)
    
    def _send_notification(self, breaker_type: BreakerType, config: BreakerConfig):
        """Send notification about breaker trigger."""
        message = (
            f"🚨 RISK BREAKER TRIGGERED\n"
            f"Type: {breaker_type.value}\n"
            f"Threshold: {config.threshold}\n"
            f"Action: {config.action}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        # This would integrate with notification system
        logger.critical(message)
    
    def _is_in_cooldown(self, breaker_type: BreakerType) -> bool:
        """Check if breaker is in cooldown period."""
        
        if breaker_type not in self.triggered_breakers:
            return False
        
        trigger_time = self.triggered_breakers[breaker_type]
        config = self.breakers[breaker_type]
        cooldown_period = timedelta(minutes=config.cooldown_minutes)
        
        return datetime.now() - trigger_time < cooldown_period
    
    def is_breaker_active(self, breaker_type: BreakerType) -> bool:
        """Check if a specific breaker is currently active (triggered and not in cooldown)."""
        
        if breaker_type not in self.triggered_breakers:
            return False
        
        return not self._is_in_cooldown(breaker_type)
    
    def can_enter_new_position(self) -> bool:
        """Check if new positions can be entered (no blocking breakers active)."""
        
        for breaker_type, config in self.breakers.items():
            if (breaker_type in self.triggered_breakers and 
                not self._is_in_cooldown(breaker_type) and
                config.action == "block_new_entries"):
                return False
        
        return True
    
    def should_close_all_positions(self) -> bool:
        """Check if all positions should be closed."""
        
        for breaker_type, config in self.breakers.items():
            if (breaker_type in self.triggered_breakers and 
                not self._is_in_cooldown(breaker_type) and
                config.action == "close_all_positions"):
                return True
        
        return False
    
    def get_size_reduction_factor(self) -> float:
        """Get position size reduction factor (1.0 = no reduction)."""
        
        reduction_factor = 1.0
        
        for breaker_type, config in self.breakers.items():
            if (breaker_type in self.triggered_breakers and 
                not self._is_in_cooldown(breaker_type) and
                config.action == "reduce_size"):
                
                # Reduce size by 50% for each active size reduction breaker
                reduction_factor *= 0.5
        
        return max(reduction_factor, 0.1)  # Minimum 10% of original size
    
    def update_trade_result(self, pnl: float):
        """Update breaker state with trade result."""
        
        self.last_trade_pnl = pnl
        
        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.debug(f"Trade PnL: ${pnl:.2f}, Consecutive losses: {self.consecutive_losses}")
    
    def update_equity(self, current_equity: float):
        """Update equity tracking for breakers."""
        
        # Update peak equity
        if current_equity > self.session_peak_equity:
            self.session_peak_equity = current_equity
        
        # Update daily P&L
        self.daily_pnl = current_equity - self.session_start_equity
        
        # Update account equity
        self.account_equity = current_equity
    
    def reset_daily_metrics(self):
        """Reset daily metrics at start of new trading session."""
        
        self.session_start_equity = self.account_equity
        self.session_peak_equity = self.account_equity
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        
        # Clear expired triggers
        self._clear_expired_triggers()
        
        logger.info("Reset daily risk metrics")
    
    def _clear_expired_triggers(self):
        """Clear triggers that have passed their cooldown period."""
        
        current_time = datetime.now()
        expired_breakers = []
        
        for breaker_type, trigger_time in self.triggered_breakers.items():
            config = self.breakers.get(breaker_type)
            if config:
                cooldown_period = timedelta(minutes=config.cooldown_minutes)
                if current_time - trigger_time > cooldown_period:
                    expired_breakers.append(breaker_type)
        
        for breaker_type in expired_breakers:
            del self.triggered_breakers[breaker_type]
            logger.info(f"Reset {breaker_type.value} breaker after cooldown")
            
            # Execute reset callbacks
            for callback in self.on_breaker_reset:
                try:
                    callback(breaker_type)
                except Exception as e:
                    logger.error(f"Error in breaker reset callback: {e}")
    
    def force_reset_breaker(self, breaker_type: BreakerType):
        """Manually reset a specific breaker."""
        
        if breaker_type in self.triggered_breakers:
            del self.triggered_breakers[breaker_type]
            logger.info(f"Manually reset {breaker_type.value} breaker")
    
    def get_breaker_status(self) -> Dict[str, Any]:
        """Get current status of all breakers."""
        
        status = {
            'account_equity': self.account_equity,
            'session_start_equity': self.session_start_equity,
            'session_peak_equity': self.session_peak_equity,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'can_enter_positions': self.can_enter_new_position(),
            'should_close_all': self.should_close_all_positions(),
            'size_reduction_factor': self.get_size_reduction_factor(),
            'active_breakers': [],
            'cooldown_breakers': []
        }
        
        current_time = datetime.now()
        
        for breaker_type, trigger_time in self.triggered_breakers.items():
            config = self.breakers.get(breaker_type)
            if config:
                time_since_trigger = current_time - trigger_time
                cooldown_period = timedelta(minutes=config.cooldown_minutes)
                
                breaker_info = {
                    'type': breaker_type.value,
                    'triggered_at': trigger_time.isoformat(),
                    'action': config.action,
                    'threshold': config.threshold
                }
                
                if time_since_trigger < cooldown_period:
                    remaining_cooldown = cooldown_period - time_since_trigger
                    breaker_info['cooldown_remaining_minutes'] = remaining_cooldown.total_seconds() / 60
                    status['cooldown_breakers'].append(breaker_info)
                else:
                    status['active_breakers'].append(breaker_info)
        
        return status
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for breaker events."""
        
        if event_type == "triggered":
            self.on_breaker_triggered.append(callback)
        elif event_type == "reset":
            self.on_breaker_reset.append(callback)
        else:
            logger.error(f"Unknown callback event type: {event_type}")
    
    def update_breaker_threshold(self, breaker_type: BreakerType, new_threshold: float):
        """Update breaker threshold."""
        
        if breaker_type in self.breakers:
            old_threshold = self.breakers[breaker_type].threshold
            self.breakers[breaker_type].threshold = new_threshold
            logger.info(f"Updated {breaker_type.value} threshold: {old_threshold} -> {new_threshold}")
        else:
            logger.error(f"Breaker {breaker_type.value} not found")