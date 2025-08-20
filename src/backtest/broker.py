"""
Simulated broker for backtesting.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class OrderStatus(Enum):
    """Order status in backtest."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestOrder:
    """Order in backtest simulation."""
    order_id: str
    symbol: str
    quantity: float
    price: float
    side: str  # buy/sell
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    created_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestPosition:
    """Position in backtest simulation."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BacktestBroker:
    """Simulated broker for backtesting."""
    
    def __init__(self, config):
        """Initialize backtest broker.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital
        self.equity = config.initial_capital
        
        # State
        self.current_date: Optional[datetime] = None
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: List[BacktestOrder] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0
        
    def set_current_date(self, date: datetime) -> None:
        """Set current simulation date.
        
        Args:
            date: Current date in simulation
        """
        self.current_date = date
        
    def get_account_equity(self) -> float:
        """Get current account equity.
        
        Returns:
            Total account equity
        """
        return self.equity
        
    def get_buying_power(self) -> float:
        """Get available buying power.
        
        Returns:
            Available cash for trading
        """
        return self.cash
        
    def get_positions(self) -> Dict[str, BacktestPosition]:
        """Get current positions.
        
        Returns:
            Dictionary of positions by symbol
        """
        return self.positions.copy()
        
    def get_position(self, symbol: str) -> Optional[BacktestPosition]:
        """Get position for specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)
        
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     side: str, commission: float = 0.0) -> bool:
        """Execute a trade immediately.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Execution price
            side: buy/sell
            commission: Trading commission
            
        Returns:
            True if trade executed successfully
        """
        try:
            trade_value = quantity * price
            total_cost = trade_value + commission
            
            if side.lower() == 'buy':
                # Check if we have enough cash
                if total_cost > self.cash:
                    return False
                
                # Update cash
                self.cash -= total_cost
                
                # Update or create position
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    # Calculate new average price
                    total_quantity = pos.quantity + quantity
                    total_value = (pos.quantity * pos.avg_price) + trade_value
                    new_avg_price = total_value / total_quantity
                    
                    pos.quantity = total_quantity
                    pos.avg_price = new_avg_price
                else:
                    self.positions[symbol] = BacktestPosition(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=price
                    )
                    
            elif side.lower() == 'sell':
                # Check if we have enough shares
                if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                    return False
                
                pos = self.positions[symbol]
                
                # Calculate realized P&L
                realized_pnl = (price - pos.avg_price) * quantity
                
                # Update cash
                self.cash += trade_value - commission
                
                # Update position
                pos.quantity -= quantity
                pos.realized_pnl += realized_pnl
                
                # Remove position if fully closed
                if pos.quantity <= 0:
                    del self.positions[symbol]
            
            # Update totals
            self.total_commission += commission
            self.trade_count += 1
            
            # Record trade
            trade_record = {
                'date': self.current_date,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'side': side,
                'commission': commission,
                'trade_value': trade_value
            }
            self.trade_history.append(trade_record)
            
            # Update equity
            self._update_equity()
            
            return True
            
        except Exception as e:
            print(f"Trade execution failed: {e}")
            return False
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update market prices for positions.
        
        Args:
            prices: Dictionary of symbol -> current price
        """
        total_market_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                total_market_value += position.market_value
        
        # Update total equity
        self.equity = self.cash + total_market_value
    
    def _update_equity(self) -> None:
        """Update total account equity."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        self.equity = self.cash + position_value
    
    def place_order(self, symbol: str, quantity: float, price: float, 
                   side: str, order_type: str = "market") -> str:
        """Place an order (for future implementation).
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Order price
            side: buy/sell
            order_type: market/limit
            
        Returns:
            Order ID
        """
        order_id = f"BO{len(self.orders) + 1:06d}"
        
        order = BacktestOrder(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            order_type=order_type,
            created_time=self.current_date
        )
        
        self.orders.append(order)
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: Order to cancel
            
        Returns:
            True if cancelled successfully
        """
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status if found
        """
        for order in self.orders:
            if order.order_id == order_id:
                return order.status
        return None
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary.
        
        Returns:
            Dictionary with account information
        """
        return {
            'equity': self.equity,
            'cash': self.cash,
            'buying_power': self.get_buying_power(),
            'positions': len(self.positions),
            'total_trades': self.trade_count,
            'total_commission': self.total_commission,
            'total_return_pct': (self.equity / self.initial_capital - 1) * 100
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'total_return_pct': (self.equity / self.initial_capital - 1) * 100,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'trade_count': self.trade_count
        }
        
        # Calculate average commission per trade
        if self.trade_count > 0:
            metrics['avg_commission_per_trade'] = self.total_commission / self.trade_count
        
        return metrics
    
    def reset(self) -> None:
        """Reset broker to initial state."""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trade_history.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0
        self.current_date = None