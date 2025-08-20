import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


class OrderMonitor:
    """Monitor order status and fills."""
    
    def __init__(self, data_client: AlpacaDataClient):
        self.data_client = data_client
        self.monitored_orders: Dict[str, Dict[str, Any]] = {}
        
    def add_order_to_monitor(self, order_id: str, metadata: Dict[str, Any] = None):
        """Add an order to monitoring."""
        self.monitored_orders[order_id] = {
            'order_id': order_id,
            'added_at': datetime.now(),
            'metadata': metadata or {},
            'last_status': 'pending'
        }
        logger.debug(f"Added order {order_id} to monitoring")
    
    def check_order_updates(self) -> List[Dict[str, Any]]:
        """Check for order status updates."""
        updates = []
        
        for order_id, order_info in self.monitored_orders.items():
            try:
                current_status = self._get_order_status(order_id)
                if current_status and current_status != order_info['last_status']:
                    update = {
                        'order_id': order_id,
                        'old_status': order_info['last_status'],
                        'new_status': current_status,
                        'timestamp': datetime.now(),
                        'metadata': order_info.get('metadata', {})
                    }
                    updates.append(update)
                    order_info['last_status'] = current_status
                    
                    logger.info(f"Order {order_id} status changed: {update['old_status']} -> {current_status}")
                    
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
        
        return updates
    
    def _get_order_status(self, order_id: str) -> Optional[str]:
        """Get current order status."""
        try:
            order = self.data_client.trading_client.get_order_by_id(order_id)
            return order.status.value
        except Exception as e:
            logger.debug(f"Could not get status for order {order_id}: {e}")
            return None
    
    def remove_completed_orders(self, max_age_hours: int = 24):
        """Remove old completed orders from monitoring."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        completed_statuses = ['filled', 'canceled', 'rejected']
        
        orders_to_remove = []
        for order_id, order_info in self.monitored_orders.items():
            if (order_info['last_status'] in completed_statuses and 
                order_info['added_at'] < cutoff_time):
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del self.monitored_orders[order_id]
            logger.debug(f"Removed completed order {order_id} from monitoring")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitored orders."""
        status_counts = {}
        for order_info in self.monitored_orders.values():
            status = order_info['last_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_orders': len(self.monitored_orders),
            'status_breakdown': status_counts,
            'orders': list(self.monitored_orders.values())
        }