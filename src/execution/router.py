import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from strategies.base import StrategySignal, OrderType
from data.alpaca_client import AlpacaDataClient
from config.loader import get_alpaca_config
from .equity import EquityOrderManager
from .monitor import OrderMonitor

logger = logging.getLogger(__name__)


class OrderRouter:
    """Central order routing and execution management."""
    
    def __init__(self, data_client: AlpacaDataClient):
        self.data_client = data_client
        self.config = get_alpaca_config()
        self.dry_run = self.config.get('dry_run', False)
        
        # Order managers
        self.equity_manager = EquityOrderManager(data_client)
        self.order_monitor = OrderMonitor(data_client)
        
        # Track submitted orders
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized OrderRouter (dry_run={self.dry_run})")
    
    def execute_signal(self, signal: StrategySignal) -> Dict[str, Any]:
        """Execute a trading signal."""
        
        if self.dry_run:
            return self._simulate_order(signal)
        
        try:
            # Route to appropriate order manager
            if signal.action in ['buy', 'sell']:
                return self.equity_manager.execute_order(signal)
            elif 'spread' in signal.action or 'condor' in signal.action:
                return self._execute_options_strategy(signal)
            else:
                raise ValueError(f"Unknown signal action: {signal.action}")
                
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal': signal.symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _simulate_order(self, signal: StrategySignal) -> Dict[str, Any]:
        """Simulate order execution for dry run mode."""
        
        logger.info(f"[DRY RUN] Would execute {signal.action} {signal.quantity} {signal.symbol}")
        
        # Get current price for simulation
        try:
            quotes = self.data_client.get_stock_quotes([signal.symbol])
            if signal.symbol in quotes:
                current_price = (quotes[signal.symbol]['bid_price'] + quotes[signal.symbol]['ask_price']) / 2
            else:
                current_price = signal.price or 100.0  # Fallback price
        except:
            current_price = signal.price or 100.0
        
        # Simulate order details
        simulated_order = {
            'success': True,
            'dry_run': True,
            'order_id': f"DRY_{signal.symbol}_{datetime.now().strftime('%H%M%S')}",
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.quantity,
            'order_type': signal.order_type.value if signal.order_type else 'market',
            'price': current_price,
            'estimated_value': signal.quantity * current_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'strategy': signal.metadata.get('strategy', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'status': 'simulated'
        }
        
        # Log the simulated execution
        self.execution_log.append(simulated_order)
        
        logger.info(f"[DRY RUN] Simulated order: {simulated_order}")
        return simulated_order
    
    def _execute_options_strategy(self, signal: StrategySignal) -> Dict[str, Any]:
        """Execute options strategy (placeholder for complex options orders)."""
        
        logger.warning(f"Options strategy execution not fully implemented: {signal.action}")
        
        # For now, simulate options execution
        return {
            'success': False,
            'error': 'Options execution not implemented yet',
            'signal': signal.symbol,
            'action': signal.action,
            'timestamp': datetime.now().isoformat(),
            'note': 'Would execute options strategy in production'
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel order {order_id}")
            return {'success': True, 'dry_run': True, 'order_id': order_id}
        
        try:
            self.data_client.trading_client.cancel_order_by_id(order_id)
            
            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            logger.info(f"Cancelled order {order_id}")
            return {'success': True, 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {'success': False, 'error': str(e), 'order_id': order_id}
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an order."""
        
        if self.dry_run:
            # Check execution log for dry run orders
            for order in self.execution_log:
                if order.get('order_id') == order_id:
                    return order
            return None
        
        try:
            order = self.data_client.trading_client.get_order_by_id(order_id)
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'created_at': order.created_at.isoformat() if order.created_at else None
            }
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders."""
        
        if self.dry_run:
            return [order for order in self.execution_log if order.get('status') == 'simulated']
        
        try:
            orders = self.data_client.get_orders('open')
            return orders
        except Exception as e:
            logger.error(f"Error getting pending orders: {e}")
            return []
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        
        total_orders = len(self.execution_log)
        successful_orders = sum(1 for order in self.execution_log if order.get('success', False))
        
        return {
            'total_orders': total_orders,
            'successful_orders': successful_orders,
            'success_rate': (successful_orders / total_orders * 100) if total_orders > 0 else 0,
            'dry_run_mode': self.dry_run,
            'recent_orders': self.execution_log[-10:] if self.execution_log else []
        }