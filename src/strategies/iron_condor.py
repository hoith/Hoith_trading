import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base import Strategy, StrategySignal, StrategyPosition, PositionSide, OrderType
from signals.iv_rank import IVRankCalculator
from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


class IronCondorStrategy(Strategy):
    """Iron Condor options strategy implementation."""
    
    def __init__(self, config: Dict[str, Any], data_client: AlpacaDataClient):
        super().__init__("iron_condor", config)
        self.data_client = data_client
        self.iv_calculator = IVRankCalculator(data_client)
        
        # Strategy parameters
        self.universe = config.get('universe', [])
        self.iv_lookback = config.get('iv_lookback', 20)
        self.delta_short = config.get('delta_short', 15)
        self.delta_long = config.get('delta_long', 10)
        self.spread_width = config.get('spread_width', 1.0)
        self.min_credit = config.get('min_credit', 0.20)
        self.max_credit = config.get('max_credit', 0.40)
        self.profit_target = config.get('profit_target', 0.5)
        self.max_loss_multiplier = config.get('max_loss_multiplier', 1.3)
        self.max_dte = config.get('max_dte', 2)
        
        logger.info(f"Initialized Iron Condor strategy with {len(self.universe)} symbols")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        current_positions: Dict[str, Any]) -> List[StrategySignal]:
        """Generate iron condor entry signals."""
        signals = []
        
        if not self.enabled:
            return signals
        
        for symbol in self.universe:
            try:
                # Skip if we already have a position
                if self.has_position(symbol):
                    continue
                
                # Check IV rank
                iv_data = self.iv_calculator.calculate_iv_rank(symbol, self.iv_lookback)
                if not iv_data or not iv_data.get('iv_elevated', False):
                    logger.debug(f"IV not elevated for {symbol}: {iv_data.get('iv_rank', 0):.1f}%")
                    continue
                
                # Get options chain
                options_chain = self.data_client.get_options_chain(symbol)
                if not options_chain:
                    logger.debug(f"No options chain for {symbol}")
                    continue
                
                # Get current stock price
                quotes = self.data_client.get_stock_quotes([symbol])
                if symbol not in quotes:
                    continue
                
                current_price = (quotes[symbol]['bid_price'] + quotes[symbol]['ask_price']) / 2
                
                # Find suitable iron condor structure
                condor_setup = self._find_iron_condor_setup(options_chain, current_price)
                
                if condor_setup:
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type='entry',
                        action='sell_iron_condor',
                        confidence=self._calculate_signal_confidence(iv_data, condor_setup),
                        quantity=1,  # Number of spreads
                        order_type=OrderType.LIMIT,
                        metadata={
                            'strategy': 'iron_condor',
                            'iv_rank': iv_data.get('iv_rank', 0),
                            'current_iv': iv_data.get('current_iv', 0),
                            'expected_credit': condor_setup['credit'],
                            'condor_structure': condor_setup,
                            'current_price': current_price,
                            'max_profit': condor_setup['credit'],
                            'max_loss': self.spread_width - condor_setup['credit'],
                            'expiration_date': condor_setup['expiration']
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"Generated iron condor signal for {symbol}: credit=${condor_setup['credit']:.2f}")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _find_iron_condor_setup(self, options_chain: List[Dict], 
                               current_price: float) -> Optional[Dict[str, Any]]:
        """Find the best iron condor setup from the options chain."""
        
        # Filter options by expiration (within max_dte days)
        max_expiry = datetime.now() + timedelta(days=self.max_dte)
        valid_options = [
            opt for opt in options_chain
            if opt.get('expiration_date') and 
            datetime.strptime(str(opt['expiration_date']), '%Y-%m-%d') <= max_expiry
        ]
        
        if not valid_options:
            return None
        
        # Group by expiration
        by_expiry = {}
        for opt in valid_options:
            expiry = opt['expiration_date']
            if expiry not in by_expiry:
                by_expiry[expiry] = {'calls': [], 'puts': []}
            
            if opt['option_type'] == 'call':
                by_expiry[expiry]['calls'].append(opt)
            else:
                by_expiry[expiry]['puts'].append(opt)
        
        best_setup = None
        best_credit = 0
        
        for expiry, opts in by_expiry.items():
            calls = sorted(opts['calls'], key=lambda x: x['strike_price'])
            puts = sorted(opts['puts'], key=lambda x: x['strike_price'])
            
            if len(calls) < 4 or len(puts) < 4:  # Need at least 4 strikes each side
                continue
            
            # Find strikes around current price
            setup = self._find_strikes_for_condor(calls, puts, current_price)
            
            if setup and setup['credit'] >= self.min_credit:
                if setup['credit'] > best_credit:
                    best_credit = setup['credit']
                    best_setup = setup
                    best_setup['expiration'] = expiry
        
        return best_setup
    
    def _find_strikes_for_condor(self, calls: List[Dict], puts: List[Dict], 
                                current_price: float) -> Optional[Dict[str, Any]]:
        """Find optimal strikes for iron condor."""
        
        # Find call strikes (above current price)
        call_strikes = [c for c in calls if c['strike_price'] > current_price]
        put_strikes = [p for p in puts if p['strike_price'] < current_price]
        
        if len(call_strikes) < 2 or len(put_strikes) < 2:
            return None
        
        # Sort by strike price
        call_strikes.sort(key=lambda x: x['strike_price'])
        put_strikes.sort(key=lambda x: x['strike_price'], reverse=True)
        
        # Try to find strikes that match our criteria
        for i in range(len(call_strikes) - 1):
            for j in range(len(put_strikes) - 1):
                
                # Call spread: sell lower strike, buy higher strike
                call_short = call_strikes[i]
                call_long = call_strikes[i + 1]
                
                # Put spread: sell higher strike, buy lower strike  
                put_short = put_strikes[j]
                put_long = put_strikes[j + 1]
                
                # Check spread width
                call_width = call_long['strike_price'] - call_short['strike_price']
                put_width = put_short['strike_price'] - put_long['strike_price']
                
                if abs(call_width - self.spread_width) > 0.01 or abs(put_width - self.spread_width) > 0.01:
                    continue
                
                # Calculate approximate delta (simplified)
                call_short_delta = self._estimate_delta(call_short['strike_price'], current_price, 'call')
                put_short_delta = self._estimate_delta(put_short['strike_price'], current_price, 'put')
                
                # Check if deltas are in target range
                if not (self.delta_short - 5 <= abs(call_short_delta) <= self.delta_short + 5):
                    continue
                if not (self.delta_short - 5 <= abs(put_short_delta) <= self.delta_short + 5):
                    continue
                
                # Calculate credit (simplified - would need real option prices)
                credit = self._estimate_credit(call_short, call_long, put_short, put_long)
                
                if self.min_credit <= credit <= self.max_credit:
                    return {
                        'call_short_strike': call_short['strike_price'],
                        'call_long_strike': call_long['strike_price'],
                        'put_short_strike': put_short['strike_price'],
                        'put_long_strike': put_long['strike_price'],
                        'credit': credit,
                        'call_short_symbol': call_short['symbol'],
                        'call_long_symbol': call_long['symbol'],
                        'put_short_symbol': put_short['symbol'],
                        'put_long_symbol': put_long['symbol'],
                        'width': self.spread_width
                    }
        
        return None
    
    def _estimate_delta(self, strike: float, spot: float, option_type: str) -> float:
        """Estimate option delta (simplified Black-Scholes approximation)."""
        moneyness = strike / spot
        
        if option_type == 'call':
            if moneyness < 0.95:
                return 80  # Deep ITM
            elif moneyness < 1.0:
                return 60  # ITM
            elif moneyness < 1.05:
                return 40  # ATM
            elif moneyness < 1.10:
                return 20  # OTM
            else:
                return 10  # Deep OTM
        else:  # put
            if moneyness > 1.05:
                return -80  # Deep ITM
            elif moneyness > 1.0:
                return -60  # ITM
            elif moneyness > 0.95:
                return -40  # ATM
            elif moneyness > 0.90:
                return -20  # OTM
            else:
                return -10  # Deep OTM
    
    def _estimate_credit(self, call_short: Dict, call_long: Dict, 
                        put_short: Dict, put_long: Dict) -> float:
        """Estimate credit received for iron condor."""
        
        # Use bid/ask if available, otherwise estimate
        call_spread_credit = 0
        put_spread_credit = 0
        
        # Call spread credit = short bid - long ask
        if call_short.get('bid_price') and call_long.get('ask_price'):
            call_spread_credit = call_short['bid_price'] - call_long['ask_price']
        else:
            # Estimate based on moneyness
            call_spread_credit = max(0.1, self.spread_width * 0.3)
        
        # Put spread credit = short bid - long ask
        if put_short.get('bid_price') and put_long.get('ask_price'):
            put_spread_credit = put_short['bid_price'] - put_long['ask_price']
        else:
            # Estimate based on moneyness
            put_spread_credit = max(0.1, self.spread_width * 0.3)
        
        total_credit = call_spread_credit + put_spread_credit
        return max(total_credit, 0.1)  # Minimum credit
    
    def _calculate_signal_confidence(self, iv_data: Dict[str, float], 
                                   condor_setup: Dict[str, Any]) -> float:
        """Calculate signal confidence based on IV and setup quality."""
        confidence = 50.0  # Base confidence
        
        # IV rank contribution
        iv_rank = iv_data.get('iv_rank', 50)
        confidence += (iv_rank - 50) * 0.5  # Higher IV = higher confidence
        
        # Credit quality contribution
        credit_ratio = condor_setup['credit'] / self.spread_width
        confidence += credit_ratio * 30  # Better credit = higher confidence
        
        # Risk/reward contribution
        max_profit = condor_setup['credit']
        max_loss = self.spread_width - condor_setup['credit']
        risk_reward = max_profit / max_loss if max_loss > 0 else 0
        confidence += min(risk_reward * 10, 20)  # Cap at 20 points
        
        return min(confidence, 95.0)  # Cap at 95%
    
    def validate_signal(self, signal: StrategySignal, 
                       market_data: Dict[str, Any]) -> bool:
        """Validate iron condor signal."""
        
        # Check if signal is for this strategy
        if signal.metadata.get('strategy') != 'iron_condor':
            return False
        
        # Check minimum credit
        expected_credit = signal.metadata.get('expected_credit', 0)
        if expected_credit < self.min_credit:
            logger.warning(f"Credit too low for {signal.symbol}: {expected_credit}")
            return False
        
        # Check risk/reward ratio
        max_loss = signal.metadata.get('max_loss', 0)
        if max_loss > 0 and expected_credit / max_loss < 0.3:  # Minimum 30% return on risk
            logger.warning(f"Poor risk/reward for {signal.symbol}")
            return False
        
        # Check option liquidity (bid-ask spread)
        condor_structure = signal.metadata.get('condor_structure', {})
        if not self._validate_option_liquidity(condor_structure):
            logger.warning(f"Poor option liquidity for {signal.symbol}")
            return False
        
        return True
    
    def _validate_option_liquidity(self, condor_structure: Dict[str, Any]) -> bool:
        """Validate that options have adequate liquidity."""
        
        # This would normally check bid-ask spreads, volume, open interest
        # For now, assume basic validation
        required_fields = ['call_short_symbol', 'call_long_symbol', 
                          'put_short_symbol', 'put_long_symbol']
        
        return all(field in condor_structure for field in required_fields)
    
    def calculate_position_size(self, signal: StrategySignal, 
                               account_equity: float, 
                               risk_per_trade: float) -> float:
        """Calculate position size (number of iron condors)."""
        
        max_loss = signal.metadata.get('max_loss', self.spread_width * 0.7)
        max_risk_dollar = account_equity * (risk_per_trade / 100)
        
        # Position size = max risk / max loss per spread
        position_size = max_risk_dollar / max_loss if max_loss > 0 else 1
        
        # Round down to whole number of spreads
        return max(1, int(position_size))
    
    def should_exit_position(self, position: StrategyPosition, 
                            market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Determine if iron condor position should be exited."""
        
        if position.strategy_name != self.name:
            return None
        
        # Get current option prices
        current_value = self._get_current_condor_value(position)
        
        # Profit target: 50% of credit received
        entry_credit = position.metadata.get('entry_credit', 0)
        profit_target_value = entry_credit * (1 - self.profit_target)
        
        # Check profit target
        if current_value <= profit_target_value:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='buy_to_close',
                confidence=90.0,
                quantity=position.quantity,
                metadata={'exit_reason': 'profit_target'}
            )
        
        # Check max loss
        max_loss = entry_credit * self.max_loss_multiplier
        if current_value >= max_loss:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='buy_to_close',
                confidence=95.0,
                quantity=position.quantity,
                metadata={'exit_reason': 'max_loss'}
            )
        
        # Check time-based exit (end of day before expiration)
        expiry_date = position.metadata.get('expiration_date')
        if expiry_date:
            try:
                expiry = datetime.strptime(str(expiry_date), '%Y-%m-%d')
                if datetime.now().date() >= expiry.date():
                    return StrategySignal(
                        symbol=position.symbol,
                        signal_type='exit',
                        action='buy_to_close',
                        confidence=100.0,
                        quantity=position.quantity,
                        metadata={'exit_reason': 'expiration'}
                    )
            except (ValueError, TypeError):
                logger.error(f"Invalid expiration date for {position.symbol}")
        
        return None
    
    def _get_current_condor_value(self, position: StrategyPosition) -> float:
        """Get current market value of iron condor position."""
        
        try:
            # Get current options chain
            options_chain = self.data_client.get_options_chain(position.symbol)
            
            if not options_chain:
                return position.entry_price  # Fallback to entry price
            
            # Extract option symbols from position metadata
            condor_symbols = position.metadata.get('condor_structure', {})
            
            total_value = 0
            option_count = 0
            
            for opt in options_chain:
                symbol = opt['symbol']
                
                # Check if this option is part of our condor
                if symbol == condor_symbols.get('call_short_symbol'):
                    total_value -= opt.get('bid_price', 0)  # We sold this
                    option_count += 1
                elif symbol == condor_symbols.get('call_long_symbol'):
                    total_value += opt.get('ask_price', 0)  # We bought this
                    option_count += 1
                elif symbol == condor_symbols.get('put_short_symbol'):
                    total_value -= opt.get('bid_price', 0)  # We sold this
                    option_count += 1
                elif symbol == condor_symbols.get('put_long_symbol'):
                    total_value += opt.get('ask_price', 0)  # We bought this
                    option_count += 1
            
            # If we found all 4 legs, return the value
            if option_count == 4:
                return abs(total_value)  # Cost to close position
            
        except Exception as e:
            logger.error(f"Error getting current condor value for {position.symbol}: {e}")
        
        # Fallback: estimate based on time decay and moneyness
        return position.entry_price * 0.8  # Assume some profit from time decay