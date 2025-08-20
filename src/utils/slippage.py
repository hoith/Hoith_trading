import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AssetType(Enum):
    EQUITY = "equity"
    OPTION = "option"
    CRYPTO = "crypto"


class SlippageModel:
    """Model slippage for different asset types and order sizes."""
    
    def __init__(self, equity_bps: float = 5, options_cents: float = 0.02, 
                 options_pct: float = 1.0, crypto_bps: float = 10):
        self.equity_bps = equity_bps
        self.options_cents = options_cents
        self.options_pct = options_pct
        self.crypto_bps = crypto_bps
    
    def calculate_slippage(self, asset_type: AssetType, price: float, 
                          quantity: float, side: str = "buy",
                          market_impact_factor: float = 1.0) -> float:
        """Calculate expected slippage for an order."""
        
        if side.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        base_slippage = 0.0
        
        if asset_type == AssetType.EQUITY:
            # Basis points slippage for equities
            base_slippage = price * (self.equity_bps / 10000)
            
        elif asset_type == AssetType.OPTION:
            # Cents + percentage for options
            cents_slippage = self.options_cents
            pct_slippage = price * (self.options_pct / 100)
            base_slippage = max(cents_slippage, pct_slippage)
            
        elif asset_type == AssetType.CRYPTO:
            # Basis points for crypto
            base_slippage = price * (self.crypto_bps / 10000)
        
        # Apply market impact factor based on order size
        # Larger orders typically have higher slippage
        size_factor = self._calculate_size_factor(quantity, asset_type)
        adjusted_slippage = base_slippage * market_impact_factor * size_factor
        
        # Apply direction (buy orders typically have positive slippage, sell negative)
        direction_multiplier = 1 if side.lower() == 'buy' else -1
        
        return adjusted_slippage * direction_multiplier
    
    def _calculate_size_factor(self, quantity: float, asset_type: AssetType) -> float:
        """Calculate size-based market impact factor."""
        
        if asset_type == AssetType.EQUITY:
            # Assume normal size is around 100 shares
            if quantity <= 100:
                return 1.0
            elif quantity <= 500:
                return 1.2
            elif quantity <= 1000:
                return 1.5
            else:
                return 2.0
                
        elif asset_type == AssetType.OPTION:
            # Assume normal size is 1-5 contracts
            if quantity <= 5:
                return 1.0
            elif quantity <= 10:
                return 1.3
            elif quantity <= 20:
                return 1.6
            else:
                return 2.0
                
        elif asset_type == AssetType.CRYPTO:
            # Size factor for crypto depends on the specific asset
            # For now, use a simple scaling
            if quantity <= 1:
                return 1.0
            elif quantity <= 10:
                return 1.2
            else:
                return 1.5
        
        return 1.0
    
    def apply_slippage_to_price(self, price: float, asset_type: AssetType,
                               quantity: float, side: str = "buy") -> float:
        """Apply slippage to a given price."""
        
        slippage = self.calculate_slippage(asset_type, price, quantity, side)
        adjusted_price = price + slippage
        
        # Ensure price doesn't go negative
        return max(adjusted_price, 0.01)
    
    def estimate_fill_price(self, bid: float, ask: float, asset_type: AssetType,
                           quantity: float, side: str = "market") -> float:
        """Estimate fill price for market orders."""
        
        if side.lower() == "buy":
            # Market buy typically fills at ask + slippage
            base_price = ask
            slippage = self.calculate_slippage(asset_type, base_price, quantity, "buy")
            return base_price + slippage
            
        elif side.lower() == "sell":
            # Market sell typically fills at bid - slippage
            base_price = bid
            slippage = self.calculate_slippage(asset_type, base_price, quantity, "sell")
            return base_price + slippage  # Note: slippage is negative for sells
        
        else:
            # For limit orders, assume mid-price
            return (bid + ask) / 2
    
    def calculate_transaction_costs(self, price: float, quantity: float,
                                   asset_type: AssetType, commission: float = 0.0) -> Dict[str, float]:
        """Calculate total transaction costs including slippage and commissions."""
        
        slippage_cost = abs(self.calculate_slippage(asset_type, price, quantity))
        total_slippage = slippage_cost * quantity
        
        if asset_type == AssetType.OPTION:
            # Options commission is typically per contract
            commission_cost = commission * quantity
        else:
            # Equity commission might be per share or flat fee
            commission_cost = commission * quantity if commission < 1 else commission
        
        notional_value = price * quantity
        
        return {
            'slippage_per_unit': slippage_cost,
            'slippage_total': total_slippage,
            'commission_total': commission_cost,
            'total_cost': total_slippage + commission_cost,
            'cost_bps': (total_slippage + commission_cost) / notional_value * 10000,
            'notional_value': notional_value
        }
    
    def get_conservative_estimate(self, price: float, asset_type: AssetType,
                                 quantity: float) -> float:
        """Get conservative (pessimistic) slippage estimate."""
        
        base_slippage = self.calculate_slippage(asset_type, price, quantity)
        
        # Add 50% buffer for conservative estimate
        return abs(base_slippage) * 1.5
    
    def update_parameters(self, **kwargs):
        """Update slippage model parameters."""
        
        if 'equity_bps' in kwargs:
            self.equity_bps = kwargs['equity_bps']
        if 'options_cents' in kwargs:
            self.options_cents = kwargs['options_cents']
        if 'options_pct' in kwargs:
            self.options_pct = kwargs['options_pct']
        if 'crypto_bps' in kwargs:
            self.crypto_bps = kwargs['crypto_bps']
            
        logger.info(f"Updated slippage parameters: {kwargs}")


class MarketImpactModel:
    """Advanced market impact model based on order flow."""
    
    def __init__(self):
        self.impact_cache = {}
    
    def calculate_impact(self, symbol: str, quantity: float, 
                        avg_volume: float, volatility: float) -> float:
        """Calculate market impact based on order size relative to average volume."""
        
        # Participation rate (order size / daily volume)
        participation_rate = quantity / avg_volume if avg_volume > 0 else 0
        
        # Base impact (square root model)
        base_impact = participation_rate ** 0.5
        
        # Adjust for volatility
        volatility_adjustment = volatility / 0.2  # Normalize to 20% volatility
        
        # Final impact in basis points
        impact_bps = base_impact * volatility_adjustment * 10
        
        return min(impact_bps, 100)  # Cap at 100 bps
    
    def get_optimal_order_size(self, target_quantity: float, avg_volume: float,
                              max_participation: float = 0.1) -> int:
        """Get optimal order size to minimize market impact."""
        
        max_size = avg_volume * max_participation
        
        if target_quantity <= max_size:
            return 1  # Single order
        
        # Calculate number of orders needed
        num_orders = int(target_quantity / max_size) + 1
        
        return num_orders