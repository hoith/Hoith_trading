import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from strategies.base import StrategySignal, OrderType
from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


class EquityOrderManager:
    """Manager for equity order execution."""
    
    def __init__(self, data_client: AlpacaDataClient):
        self.data_client = data_client
        
    def execute_order(self, signal: StrategySignal) -> Dict[str, Any]:
        """Execute an equity order from a signal."""
        
        try:
            # Validate the signal
            if not self._validate_signal(signal):
                return {
                    'success': False,
                    'error': 'Signal validation failed',
                    'signal': signal.symbol
                }
            
            # Create order request
            order_request = self._create_order_request(signal)
            if not order_request:
                return {
                    'success': False,
                    'error': 'Failed to create order request',
                    'signal': signal.symbol
                }
            
            # Submit order
            order = self.data_client.trading_client.submit_order(order_request)
            
            # Log successful submission
            logger.info(f"Submitted order: {signal.action} {signal.quantity} {signal.symbol}")
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': signal.quantity,
                'order_type': signal.order_type.value if signal.order_type else 'market',
                'status': order.status.value,
                'timestamp': datetime.now().isoformat(),
                'strategy': signal.metadata.get('strategy', 'unknown')
            }
            
        except APIError as e:
            logger.error(f"Alpaca API error executing order for {signal.symbol}: {e}")
            return {
                'success': False,
                'error': f"API Error: {str(e)}",
                'signal': signal.symbol,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error executing order for {signal.symbol}: {e}")
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'signal': signal.symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_signal(self, signal: StrategySignal) -> bool:
        """Validate signal before execution."""
        
        if not signal.symbol:
            logger.error("Missing symbol in signal")
            return False
        
        if not signal.quantity or signal.quantity <= 0:
            logger.error(f"Invalid quantity for {signal.symbol}: {signal.quantity}")
            return False
        
        if signal.action not in ['buy', 'sell']:
            logger.error(f"Invalid action for equity order: {signal.action}")
            return False
        
        # Check if asset is tradable
        try:
            asset_info = self.data_client.get_asset_info(signal.symbol)
            if not asset_info.get('tradable', False):
                logger.error(f"Asset {signal.symbol} is not tradable")
                return False
        except Exception as e:
            logger.warning(f"Could not verify asset tradability for {signal.symbol}: {e}")
        
        return True
    
    def _create_order_request(self, signal: StrategySignal):
        """Create appropriate order request from signal."""
        
        # Determine order side
        side = OrderSide.BUY if signal.action == 'buy' else OrderSide.SELL
        
        # Handle fractional shares
        if signal.quantity != int(signal.quantity):
            # Fractional share order
            return self._create_notional_order(signal, side)
        
        # Determine order type
        if signal.order_type == OrderType.MARKET or signal.order_type is None:
            return MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
        
        elif signal.order_type == OrderType.LIMIT:
            if not signal.price:
                logger.error(f"Limit order requires price for {signal.symbol}")
                return None
            
            return LimitOrderRequest(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=signal.price
            )
        
        elif signal.order_type == OrderType.STOP:
            if not signal.stop_loss:
                logger.error(f"Stop order requires stop price for {signal.symbol}")
                return None
            
            return StopOrderRequest(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                stop_price=signal.stop_loss
            )
        
        else:
            logger.error(f"Unsupported order type: {signal.order_type}")
            return None
    
    def _create_notional_order(self, signal: StrategySignal, side: OrderSide):
        """Create notional (dollar-based) order for fractional shares."""
        
        try:
            # Get current price to calculate notional amount
            quotes = self.data_client.get_stock_quotes([signal.symbol])
            if signal.symbol not in quotes:
                logger.error(f"Could not get quote for fractional order: {signal.symbol}")
                return None
            
            current_price = (quotes[signal.symbol]['bid_price'] + quotes[signal.symbol]['ask_price']) / 2
            notional_amount = signal.quantity * current_price
            
            # Create notional market order
            return MarketOrderRequest(
                symbol=signal.symbol,
                notional=notional_amount,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
        except Exception as e:
            logger.error(f"Error creating notional order for {signal.symbol}: {e}")
            return None
    
    def create_bracket_order(self, signal: StrategySignal) -> Dict[str, Any]:
        """Create bracket order with stop loss and take profit."""
        
        try:
            if not signal.stop_loss and not signal.take_profit:
                return self.execute_order(signal)
            
            # For now, submit main order and manage stops separately
            # In production, you'd use Alpaca's bracket order functionality
            main_order_result = self.execute_order(signal)
            
            if main_order_result['success'] and (signal.stop_loss or signal.take_profit):
                logger.info(f"Would create bracket orders for {signal.symbol} "
                           f"(stop: {signal.stop_loss}, target: {signal.take_profit})")
            
            return main_order_result
            
        except Exception as e:
            logger.error(f"Error creating bracket order for {signal.symbol}: {e}")
            return {
                'success': False,
                'error': f"Bracket order error: {str(e)}",
                'signal': signal.symbol
            }
    
    def get_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol."""
        
        try:
            positions = self.data_client.get_positions()
            for position in positions:
                if position['symbol'] == symbol:
                    return position.get('market_value', 0)
            return 0
        except Exception as e:
            logger.error(f"Error getting position value for {symbol}: {e}")
            return 0
    
    def close_position(self, symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """Close a position (partial or full)."""
        
        try:
            # Get current position
            positions = self.data_client.get_positions()
            position = None
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    position = pos
                    break
            
            if not position:
                return {
                    'success': False,
                    'error': f'No position found for {symbol}',
                    'symbol': symbol
                }
            
            # Determine quantity to close
            current_qty = abs(float(position['qty']))
            close_qty = quantity if quantity and quantity <= current_qty else current_qty
            
            # Determine side (opposite of current position)
            position_side = position['side']
            close_side = OrderSide.SELL if position_side == 'long' else OrderSide.BUY
            
            # Create close order
            close_order = MarketOrderRequest(
                symbol=symbol,
                qty=close_qty,
                side=close_side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.data_client.trading_client.submit_order(close_order)
            
            logger.info(f"Submitted close order for {symbol}: {close_qty} shares")
            
            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'action': 'close',
                'quantity': close_qty,
                'status': order.status.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }