import logging
import os
import json
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemAlert:
    """System alert data structure."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None


class NotificationManager:
    """Handle notifications via various channels."""
    
    def __init__(self, webhook_url: str = None, enabled: bool = True):
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.enabled = enabled and bool(self.webhook_url)
        self.alert_history: List[SystemAlert] = []
        self.max_history = 1000
        
        if self.enabled:
            logger.info("Notification manager initialized with webhook")
        else:
            logger.info("Notification manager initialized (notifications disabled)")
    
    def send_alert(self, alert: SystemAlert) -> bool:
        """Send an alert notification."""
        self.alert_history.append(alert)
        
        # Keep history size manageable
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        if not self.enabled:
            logger.debug(f"Alert (notifications disabled): {alert.title}")
            return True
        
        try:
            # Prepare message
            color = self._get_color_for_level(alert.level)
            
            payload = {
                "text": f"Trading System Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert.level.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Details",
                                "value": alert.message,
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            # Add metadata if present
            if alert.metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in alert.metadata.items()])
                payload["attachments"][0]["fields"].append({
                    "title": "Metadata",
                    "value": metadata_text,
                    "short": False
                })
            
            # Send to webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.debug(f"Alert sent successfully: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    def _get_color_for_level(self, level: AlertLevel) -> str:
        """Get color code for alert level."""
        color_map = {
            AlertLevel.INFO: "good",      # green
            AlertLevel.WARNING: "warning", # yellow
            AlertLevel.ERROR: "danger",    # red
            AlertLevel.CRITICAL: "danger"  # red
        }
        return color_map.get(level, "good")
    
    def send_trade_notification(self, trade_data: Dict[str, Any]):
        """Send trade execution notification."""
        symbol = trade_data.get('symbol', 'UNKNOWN')
        action = trade_data.get('action', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        strategy = trade_data.get('strategy', 'UNKNOWN')
        
        alert = SystemAlert(
            level=AlertLevel.INFO,
            title=f"Trade Executed: {action.upper()} {symbol}",
            message=f"Executed {action} {quantity} shares of {symbol} at ${price:.2f}",
            timestamp=datetime.now(),
            source=f"Strategy: {strategy}",
            metadata={
                'order_id': trade_data.get('order_id'),
                'estimated_value': trade_data.get('estimated_value'),
                'order_type': trade_data.get('order_type')
            }
        )
        
        self.send_alert(alert)
    
    def send_risk_breach_notification(self, breach_type: str, details: Dict[str, Any]):
        """Send risk management breach notification."""
        alert = SystemAlert(
            level=AlertLevel.CRITICAL,
            title=f"Risk Breach: {breach_type}",
            message=f"Risk management breach detected: {breach_type}",
            timestamp=datetime.now(),
            source="Risk Management System",
            metadata=details
        )
        
        self.send_alert(alert)
    
    def send_system_status(self, status_data: Dict[str, Any]):
        """Send daily system status summary."""
        alert = SystemAlert(
            level=AlertLevel.INFO,
            title="Daily Trading Summary",
            message="End of day system status and performance summary",
            timestamp=datetime.now(),
            source="System Monitor",
            metadata=status_data
        )
        
        self.send_alert(alert)
    
    def get_recent_alerts(self, hours: int = 24) -> List[SystemAlert]:
        """Get alerts from recent period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]


class SystemMonitor:
    """Monitor system health and performance."""
    
    def __init__(self, notification_manager: NotificationManager = None):
        self.notification_manager = notification_manager
        self.monitors: Dict[str, Callable] = {}
        self.last_check = {}
        self.health_status = {}
        
        # Register default monitors
        self._register_default_monitors()
    
    def _register_default_monitors(self):
        """Register default system monitors."""
        self.register_monitor("database_connection", self._check_database_connection)
        self.register_monitor("api_connection", self._check_api_connection)
        self.register_monitor("disk_space", self._check_disk_space)
        self.register_monitor("memory_usage", self._check_memory_usage)
    
    def register_monitor(self, name: str, check_function: Callable) -> None:
        """Register a new monitor."""
        self.monitors[name] = check_function
        logger.debug(f"Registered monitor: {name}")
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        results = {}
        
        for monitor_name, check_function in self.monitors.items():
            try:
                result = check_function()
                results[monitor_name] = {
                    'status': 'healthy' if result.get('healthy', True) else 'unhealthy',
                    'details': result,
                    'last_check': datetime.now()
                }
                
                # Store for trend analysis
                self.health_status[monitor_name] = results[monitor_name]
                
                # Send alerts for unhealthy systems
                if not result.get('healthy', True):
                    self._send_health_alert(monitor_name, result)
                    
            except Exception as e:
                logger.error(f"Health check failed for {monitor_name}: {e}")
                results[monitor_name] = {
                    'status': 'error',
                    'details': {'error': str(e)},
                    'last_check': datetime.now()
                }
        
        return results
    
    def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            from state.database import DatabaseManager
            db = DatabaseManager()
            
            with db.get_session() as session:
                # Simple query to test connection
                result = session.execute("SELECT 1").fetchone()
                
            return {
                'healthy': True,
                'message': 'Database connection successful',
                'response_time': 'fast'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Database connection failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_api_connection(self) -> Dict[str, Any]:
        """Check Alpaca API connectivity."""
        try:
            from data.alpaca_client import AlpacaDataClient
            client = AlpacaDataClient()
            
            # Test API connection
            account = client.get_account()
            market_open = client.is_market_open()
            
            return {
                'healthy': True,
                'message': 'API connection successful',
                'account_status': account.get('trading_blocked', False),
                'market_open': market_open
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'API connection failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            # Check disk space for database directory
            db_path = os.path.dirname(os.getenv('DATABASE_URL', 'trading_system.db'))
            if not db_path:
                db_path = '.'
            
            total, used, free = shutil.disk_usage(db_path)
            free_pct = (free / total) * 100
            
            return {
                'healthy': free_pct > 10,  # Alert if less than 10% free
                'message': f'Disk space: {free_pct:.1f}% free',
                'free_gb': free // (1024**3),
                'free_percent': free_pct,
                'total_gb': total // (1024**3)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Disk space check failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            
            # Get current process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            system_usage_pct = system_memory.percent
            
            return {
                'healthy': memory_mb < 1000 and system_usage_pct < 90,  # Alert if >1GB or system >90%
                'message': f'Memory usage: {memory_mb:.1f}MB (system: {system_usage_pct:.1f}%)',
                'process_memory_mb': memory_mb,
                'system_memory_percent': system_usage_pct,
                'system_available_gb': system_memory.available / (1024**3)
            }
            
        except ImportError:
            # psutil not available, skip memory check
            return {
                'healthy': True,
                'message': 'Memory check skipped (psutil not available)'
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Memory check failed: {str(e)}',
                'error': str(e)
            }
    
    def _send_health_alert(self, monitor_name: str, result: Dict[str, Any]):
        """Send health check alert."""
        if not self.notification_manager:
            return
        
        alert = SystemAlert(
            level=AlertLevel.WARNING,
            title=f"Health Check Failed: {monitor_name}",
            message=result.get('message', 'Health check failed'),
            timestamp=datetime.now(),
            source="System Monitor",
            metadata={'monitor': monitor_name, 'details': result}
        )
        
        self.notification_manager.send_alert(alert)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_results = self.run_health_checks()
        
        # Count healthy vs unhealthy checks
        healthy_count = sum(1 for result in health_results.values() if result['status'] == 'healthy')
        total_checks = len(health_results)
        
        overall_status = 'healthy' if healthy_count == total_checks else 'degraded'
        if healthy_count == 0:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now(),
            'health_score': (healthy_count / total_checks * 100) if total_checks > 0 else 0,
            'checks': health_results,
            'summary': {
                'total_checks': total_checks,
                'healthy_checks': healthy_count,
                'unhealthy_checks': total_checks - healthy_count
            }
        }
    
    def start_monitoring(self, interval_minutes: int = 15):
        """Start continuous monitoring (would run in background thread in production)."""
        logger.info(f"System monitoring started (interval: {interval_minutes} minutes)")
        # In a production system, this would run in a background thread or scheduler
        # For now, it's a placeholder for manual health checks