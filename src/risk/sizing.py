import logging
import math
from typing import Dict, Optional, Any
from enum import Enum

from strategies.base import StrategySignal

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"


class PositionSizer:
    """Position sizing and risk management calculations."""
    
    def __init__(self, account_equity: float, max_position_value: float = None,
                 max_positions: int = 10, risk_per_trade_pct: float = 1.0):
        self.account_equity = account_equity
        self.max_position_value = max_position_value or account_equity * 0.2  # 20% max per position
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        
        logger.info(f"Initialized PositionSizer: equity=${account_equity:,.2f}, "
                   f"max_position=${self.max_position_value:,.2f}, max_positions={max_positions}")
    
    def calculate_position_size(self, signal: StrategySignal, 
                               current_positions: int = 0,
                               method: SizingMethod = SizingMethod.PERCENT_EQUITY,
                               **kwargs) -> Dict[str, Any]:
        """Calculate optimal position size for a signal."""
        
        if current_positions >= self.max_positions:
            logger.warning(f"Maximum positions reached: {current_positions}")
            return self._zero_position("max_positions_reached")
        
        # Get signal metadata
        entry_price = signal.price or kwargs.get('entry_price', 0)
        stop_loss = signal.stop_loss or kwargs.get('stop_loss')
        
        if not entry_price or entry_price <= 0:
            logger.error(f"Invalid entry price for {signal.symbol}: {entry_price}")
            return self._zero_position("invalid_entry_price")
        
        try:
            if method == SizingMethod.FIXED_DOLLAR:
                return self._fixed_dollar_sizing(signal, entry_price, **kwargs)
            
            elif method == SizingMethod.PERCENT_EQUITY:
                return self._percent_equity_sizing(signal, entry_price, stop_loss, **kwargs)
            
            elif method == SizingMethod.VOLATILITY_ADJUSTED:
                return self._volatility_adjusted_sizing(signal, entry_price, stop_loss, **kwargs)
            
            elif method == SizingMethod.KELLY_CRITERION:
                return self._kelly_criterion_sizing(signal, entry_price, stop_loss, **kwargs)
            
            else:
                logger.error(f"Unknown sizing method: {method}")
                return self._zero_position("unknown_method")
                
        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return self._zero_position("calculation_error")
    
    def _fixed_dollar_sizing(self, signal: StrategySignal, entry_price: float,
                            fixed_amount: float = None, **kwargs) -> Dict[str, Any]:
        """Fixed dollar amount position sizing."""
        
        amount = fixed_amount or signal.metadata.get('position_size_usd', 1000)
        amount = min(amount, self.max_position_value)
        
        quantity = amount / entry_price
        
        # For fractional shares, keep precision
        if signal.metadata.get('strategy') == 'fractional_breakout':
            quantity = round(quantity, 6)
        else:
            quantity = math.floor(quantity)  # Whole shares only
        
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'method': 'fixed_dollar',
            'target_amount': amount,
            'valid': quantity > 0
        }
    
    def _percent_equity_sizing(self, signal: StrategySignal, entry_price: float,
                              stop_loss: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Percentage of equity position sizing."""
        
        # Calculate risk amount
        risk_amount = self.account_equity * (self.risk_per_trade_pct / 100)
        
        if stop_loss and stop_loss > 0:
            # Risk-based sizing: risk_amount / risk_per_share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
            else:
                # Fallback to 2% risk assumption
                quantity = risk_amount / (entry_price * 0.02)
        else:
            # No stop loss - use fixed percentage of equity
            target_value = min(risk_amount * 10, self.max_position_value)  # 10x leverage assumption
            quantity = target_value / entry_price
        
        # Apply constraints
        max_quantity_by_value = self.max_position_value / entry_price
        quantity = min(quantity, max_quantity_by_value)
        
        # Round appropriately
        if signal.metadata.get('strategy') == 'fractional_breakout':
            quantity = round(quantity, 6)
        else:
            quantity = math.floor(quantity)
        
        position_value = quantity * entry_price
        actual_risk = quantity * abs(entry_price - (stop_loss or entry_price * 0.98))
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'actual_risk': actual_risk,
            'risk_pct': (actual_risk / self.account_equity) * 100,
            'method': 'percent_equity',
            'valid': quantity > 0 and actual_risk <= risk_amount * 1.1  # 10% tolerance
        }
    
    def _volatility_adjusted_sizing(self, signal: StrategySignal, entry_price: float,
                                   stop_loss: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Volatility-adjusted position sizing."""
        
        # Get volatility measure (ATR or implied volatility)
        volatility = signal.metadata.get('atr', 0)
        if volatility <= 0:
            volatility = entry_price * 0.02  # 2% default volatility
        
        # Normalize volatility to percentage
        volatility_pct = volatility / entry_price
        
        # Base risk amount
        base_risk = self.account_equity * (self.risk_per_trade_pct / 100)
        
        # Adjust for volatility - higher volatility = smaller position
        volatility_adjustment = min(2.0, max(0.5, 0.02 / volatility_pct))  # Target 2% volatility
        adjusted_risk = base_risk * volatility_adjustment
        
        # Calculate position size
        if stop_loss and stop_loss > 0:
            risk_per_share = abs(entry_price - stop_loss)
            quantity = adjusted_risk / risk_per_share if risk_per_share > 0 else 0
        else:
            # Use volatility as risk measure
            quantity = adjusted_risk / volatility
        
        # Apply constraints
        max_quantity = self.max_position_value / entry_price
        quantity = min(quantity, max_quantity)
        
        # Round appropriately
        if signal.metadata.get('strategy') == 'fractional_breakout':
            quantity = round(quantity, 6)
        else:
            quantity = math.floor(quantity)
        
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'volatility': volatility,
            'volatility_pct': volatility_pct,
            'volatility_adjustment': volatility_adjustment,
            'method': 'volatility_adjusted',
            'valid': quantity > 0
        }
    
    def _kelly_criterion_sizing(self, signal: StrategySignal, entry_price: float,
                               stop_loss: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Kelly Criterion position sizing (requires win rate and avg win/loss)."""
        
        # Get strategy performance metrics
        win_rate = kwargs.get('win_rate', signal.metadata.get('win_rate', 0.5))  # 50% default
        avg_win = kwargs.get('avg_win', signal.metadata.get('avg_win', 0.1))  # 10% default
        avg_loss = kwargs.get('avg_loss', signal.metadata.get('avg_loss', 0.05))  # 5% default
        
        if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
            logger.warning(f"Insufficient data for Kelly sizing for {signal.symbol}")
            return self._percent_equity_sizing(signal, entry_price, stop_loss, **kwargs)
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = 1-p
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly fraction to prevent over-leverage
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% of equity
        
        # Calculate position size
        kelly_amount = self.account_equity * kelly_fraction
        kelly_amount = min(kelly_amount, self.max_position_value)
        
        quantity = kelly_amount / entry_price
        
        # Round appropriately
        if signal.metadata.get('strategy') == 'fractional_breakout':
            quantity = round(quantity, 6)
        else:
            quantity = math.floor(quantity)
        
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'kelly_fraction': kelly_fraction,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'method': 'kelly_criterion',
            'valid': quantity > 0 and kelly_fraction > 0
        }
    
    def _zero_position(self, reason: str) -> Dict[str, Any]:
        """Return zero position with reason."""
        return {
            'quantity': 0,
            'position_value': 0,
            'valid': False,
            'reason': reason,
            'method': 'none'
        }
    
    def validate_position_size(self, symbol: str, quantity: float, price: float,
                              current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position size against various constraints."""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        position_value = quantity * price
        
        # Check maximum position value
        if position_value > self.max_position_value:
            validation_results['errors'].append(
                f"Position value ${position_value:,.2f} exceeds maximum ${self.max_position_value:,.2f}"
            )
            validation_results['valid'] = False
        
        # Check maximum number of positions
        if len(current_positions) >= self.max_positions:
            validation_results['errors'].append(
                f"Already at maximum positions: {len(current_positions)}"
            )
            validation_results['valid'] = False
        
        # Check concentration risk (max 30% in single position)
        concentration_limit = self.account_equity * 0.3
        if position_value > concentration_limit:
            validation_results['warnings'].append(
                f"Position represents {(position_value/self.account_equity)*100:.1f}% of equity"
            )
        
        # Check total exposure
        total_exposure = sum(pos.get('market_value', 0) for pos in current_positions.values())
        total_exposure += position_value
        
        max_total_exposure = self.account_equity * 2.0  # 200% maximum exposure
        if total_exposure > max_total_exposure:
            validation_results['errors'].append(
                f"Total exposure ${total_exposure:,.2f} would exceed maximum ${max_total_exposure:,.2f}"
            )
            validation_results['valid'] = False
        
        return validation_results
    
    def update_equity(self, new_equity: float):
        """Update account equity for position sizing calculations."""
        old_equity = self.account_equity
        self.account_equity = new_equity
        self.max_position_value = new_equity * 0.2  # Update max position value
        
        logger.info(f"Updated equity: ${old_equity:,.2f} -> ${new_equity:,.2f}")
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get current sizing parameters summary."""
        return {
            'account_equity': self.account_equity,
            'max_position_value': self.max_position_value,
            'max_positions': self.max_positions,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'max_total_exposure': self.account_equity * 2.0,
            'position_value_pct': (self.max_position_value / self.account_equity) * 100
        }