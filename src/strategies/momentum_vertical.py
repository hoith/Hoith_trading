import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base import Strategy, StrategySignal, StrategyPosition, PositionSide, OrderType
from signals.momentum import MomentumSignals
from data.alpaca_client import AlpacaDataClient
from data.historical import HistoricalDataFetcher

logger = logging.getLogger(__name__)


class MomentumVerticalStrategy(Strategy):
    """Momentum-based vertical options spread strategy."""
    
    def __init__(self, config: Dict[str, Any], data_client: AlpacaDataClient):
        super().__init__("momentum_vertical", config)
        self.data_client = data_client
        self.historical_fetcher = HistoricalDataFetcher(data_client)
        self.momentum_signals = MomentumSignals()
        
        # Strategy parameters
        self.universe = config.get('universe', [])
        self.spread_width = config.get('spread_width', 1.0)
        self.max_debit = config.get('max_debit', 0.35)
        self.profit_target = config.get('profit_target', 0.8)  # 80% of max value
        self.breakout_lookback = config.get('breakout_lookback', 20)
        self.volume_threshold = config.get('volume_threshold', 1000000)
        
        logger.info(f"Initialized Momentum Vertical strategy with {len(self.universe)} symbols")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        current_positions: Dict[str, Any]) -> List[StrategySignal]:
        """Generate momentum vertical entry signals."""
        signals = []
        
        if not self.enabled:
            return signals
        
        for symbol in self.universe:
            try:
                # Skip if we already have a position
                if self.has_position(symbol):
                    continue
                
                # Get historical data for momentum analysis
                if symbol not in market_data or market_data[symbol].empty:
                    continue
                
                df = market_data[symbol]
                
                # Check momentum criteria
                momentum_data = self._analyze_momentum(df)
                if not momentum_data['has_breakout']:
                    continue
                
                # Check volume criteria
                volume_check = self._validate_volume(df)
                if not volume_check:
                    continue
                
                # Get options chain for the symbol
                options_chain = self.data_client.get_options_chain(symbol)
                if not options_chain:
                    logger.debug(f"No options chain for {symbol}")
                    continue
                
                # Get current stock price
                quotes = self.data_client.get_stock_quotes([symbol])
                if symbol not in quotes:
                    continue
                
                current_price = (quotes[symbol]['bid_price'] + quotes[symbol]['ask_price']) / 2
                
                # Determine direction based on momentum
                direction = momentum_data['direction']
                
                # Find suitable vertical spread
                vertical_setup = self._find_vertical_spread(
                    options_chain, current_price, direction
                )
                
                if vertical_setup:
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type='entry',
                        action=f'{direction}_vertical_spread',
                        confidence=self._calculate_signal_confidence(momentum_data, vertical_setup),
                        quantity=1,  # Number of spreads
                        order_type=OrderType.LIMIT,
                        metadata={
                            'strategy': 'momentum_vertical',
                            'direction': direction,
                            'momentum_score': momentum_data['momentum_score'],
                            'breakout_strength': momentum_data['breakout_strength'],
                            'volume_ratio': momentum_data['volume_ratio'],
                            'vertical_setup': vertical_setup,
                            'current_price': current_price,
                            'max_profit': vertical_setup['max_profit'],
                            'max_loss': vertical_setup['debit_paid'],
                            'expiration_date': vertical_setup['expiration']
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"Generated {direction} vertical signal for {symbol}: debit=${vertical_setup['debit_paid']:.2f}")
                
            except Exception as e:
                logger.error(f"Error generating momentum vertical signal for {symbol}: {e}")
        
        return signals
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum conditions for entry."""
        
        # Get breakout signals
        breakout_signals = self.momentum_signals.get_breakout_signals(
            df, self.breakout_lookback, volume_threshold=1.5
        )
        
        # Check for recent breakout
        has_breakout = False
        breakout_strength = 0
        direction = None
        
        if not breakout_signals.empty:
            recent_breakout = breakout_signals['breakout_signal'].tail(5).any()
            
            if recent_breakout:
                has_breakout = True
                # Get strength of most recent signal
                recent_signals = breakout_signals.tail(5)
                breakout_strength = recent_signals['signal_strength'].max()
                
                # Determine direction based on price momentum
                price_momentum = df['close'].pct_change(5).iloc[-1] * 100
                direction = 'bullish' if price_momentum > 0 else 'bearish'
        
        # Calculate momentum score
        momentum_score = self.momentum_signals.get_momentum_score(df).iloc[-1] if len(df) > 20 else 0
        
        # Calculate volume ratio
        volume_ratio = breakout_signals['volume_ratio'].iloc[-1] if not breakout_signals.empty else 1.0
        
        return {
            'has_breakout': has_breakout,
            'breakout_strength': breakout_strength,
            'momentum_score': momentum_score,
            'direction': direction,
            'volume_ratio': volume_ratio,
            'trend': self.momentum_signals.identify_trend_direction(df).iloc[-1]
        }
    
    def _validate_volume(self, df: pd.DataFrame) -> bool:
        """Validate volume criteria."""
        
        if len(df) < 20:
            return False
        
        # Check average volume
        avg_volume = df['volume'].tail(20).mean()
        if avg_volume < self.volume_threshold:
            logger.debug(f"Volume too low: {avg_volume}")
            return False
        
        # Check recent volume spike
        recent_volume = df['volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio < 1.1:  # Reduced from 1.2 to 1.1
            logger.debug(f"No volume spike: {volume_ratio}")
            return False
        
        return True
    
    def _find_vertical_spread(self, options_chain: List[Dict], 
                             current_price: float, direction: str) -> Optional[Dict[str, Any]]:
        """Find suitable vertical spread setup."""
        
        # Filter for near-term expirations (1-30 days)
        max_expiry = datetime.now() + timedelta(days=30)
        min_expiry = datetime.now() + timedelta(days=1)
        
        valid_options = [
            opt for opt in options_chain
            if opt.get('expiration_date') and
            min_expiry <= datetime.strptime(str(opt['expiration_date']), '%Y-%m-%d') <= max_expiry
        ]
        
        if not valid_options:
            return None
        
        # Group by expiration and option type
        by_expiry = {}
        for opt in valid_options:
            expiry = opt['expiration_date']
            opt_type = opt['option_type']
            
            if expiry not in by_expiry:
                by_expiry[expiry] = {'call': [], 'put': []}
            
            by_expiry[expiry][opt_type].append(opt)
        
        best_setup = None
        best_value = 0
        
        for expiry, opts in by_expiry.items():
            
            if direction == 'bullish':
                # Bull call spread: buy lower strike, sell higher strike
                setup = self._find_bull_call_spread(opts['call'], current_price)
            else:
                # Bear put spread: buy higher strike, sell lower strike
                setup = self._find_bear_put_spread(opts['put'], current_price)
            
            if setup and setup['debit_paid'] <= self.max_debit:
                # Evaluate setup quality
                risk_reward = setup['max_profit'] / setup['debit_paid'] if setup['debit_paid'] > 0 else 0
                
                if risk_reward > best_value:
                    best_value = risk_reward
                    best_setup = setup
                    best_setup['expiration'] = expiry
        
        return best_setup
    
    def _find_bull_call_spread(self, calls: List[Dict], 
                              current_price: float) -> Optional[Dict[str, Any]]:
        """Find bull call spread (buy lower strike, sell higher strike)."""
        
        # Filter calls and sort by strike
        valid_calls = [c for c in calls if c['strike_price'] >= current_price * 0.98]
        valid_calls.sort(key=lambda x: x['strike_price'])
        
        if len(valid_calls) < 2:
            return None
        
        # Try different combinations
        for i in range(len(valid_calls) - 1):
            buy_option = valid_calls[i]
            sell_option = valid_calls[i + 1]
            
            # Check spread width
            spread_width = sell_option['strike_price'] - buy_option['strike_price']
            if abs(spread_width - self.spread_width) > 0.01:
                continue
            
            # Calculate debit (buy ask - sell bid)
            buy_price = buy_option.get('ask_price', 0)
            sell_price = sell_option.get('bid_price', 0)
            
            if buy_price <= 0 or sell_price <= 0:
                continue
            
            debit_paid = buy_price - sell_price
            if debit_paid <= 0 or debit_paid > self.max_debit:
                continue
            
            # Calculate max profit
            max_profit = spread_width - debit_paid
            
            return {
                'type': 'bull_call_spread',
                'buy_strike': buy_option['strike_price'],
                'sell_strike': sell_option['strike_price'],
                'buy_symbol': buy_option['symbol'],
                'sell_symbol': sell_option['symbol'],
                'debit_paid': debit_paid,
                'max_profit': max_profit,
                'spread_width': spread_width,
                'breakeven': buy_option['strike_price'] + debit_paid
            }
        
        return None
    
    def _find_bear_put_spread(self, puts: List[Dict], 
                             current_price: float) -> Optional[Dict[str, Any]]:
        """Find bear put spread (buy higher strike, sell lower strike)."""
        
        # Filter puts and sort by strike (descending)
        valid_puts = [p for p in puts if p['strike_price'] <= current_price * 1.02]
        valid_puts.sort(key=lambda x: x['strike_price'], reverse=True)
        
        if len(valid_puts) < 2:
            return None
        
        # Try different combinations
        for i in range(len(valid_puts) - 1):
            buy_option = valid_puts[i]  # Higher strike
            sell_option = valid_puts[i + 1]  # Lower strike
            
            # Check spread width
            spread_width = buy_option['strike_price'] - sell_option['strike_price']
            if abs(spread_width - self.spread_width) > 0.01:
                continue
            
            # Calculate debit (buy ask - sell bid)
            buy_price = buy_option.get('ask_price', 0)
            sell_price = sell_option.get('bid_price', 0)
            
            if buy_price <= 0 or sell_price <= 0:
                continue
            
            debit_paid = buy_price - sell_price
            if debit_paid <= 0 or debit_paid > self.max_debit:
                continue
            
            # Calculate max profit
            max_profit = spread_width - debit_paid
            
            return {
                'type': 'bear_put_spread',
                'buy_strike': buy_option['strike_price'],
                'sell_strike': sell_option['strike_price'],
                'buy_symbol': buy_option['symbol'],
                'sell_symbol': sell_option['symbol'],
                'debit_paid': debit_paid,
                'max_profit': max_profit,
                'spread_width': spread_width,
                'breakeven': buy_option['strike_price'] - debit_paid
            }
        
        return None
    
    def _calculate_signal_confidence(self, momentum_data: Dict[str, float], 
                                   vertical_setup: Dict[str, Any]) -> float:
        """Calculate signal confidence."""
        confidence = 50.0
        
        # Momentum strength contribution
        momentum_score = momentum_data.get('momentum_score', 0)
        confidence += min(momentum_score * 2, 25)  # Cap at 25 points
        
        # Breakout strength contribution
        breakout_strength = momentum_data.get('breakout_strength', 0)
        confidence += min(breakout_strength * 0.3, 15)  # Cap at 15 points
        
        # Volume contribution
        volume_ratio = momentum_data.get('volume_ratio', 1)
        confidence += min((volume_ratio - 1) * 10, 10)  # Cap at 10 points
        
        # Risk/reward contribution
        risk_reward = vertical_setup['max_profit'] / vertical_setup['debit_paid']
        confidence += min(risk_reward * 5, 15)  # Cap at 15 points
        
        return min(confidence, 95.0)
    
    def validate_signal(self, signal: StrategySignal, 
                       market_data: Dict[str, Any]) -> bool:
        """Validate momentum vertical signal."""
        
        if signal.metadata.get('strategy') != 'momentum_vertical':
            return False
        
        # Check debit paid
        vertical_setup = signal.metadata.get('vertical_setup', {})
        debit_paid = vertical_setup.get('debit_paid', 0)
        
        if debit_paid > self.max_debit:
            logger.warning(f"Debit too high for {signal.symbol}: {debit_paid}")
            return False
        
        # Check risk/reward ratio
        max_profit = vertical_setup.get('max_profit', 0)
        if debit_paid > 0 and max_profit / debit_paid < 1.0:  # At least 1:1 reward/risk
            logger.warning(f"Poor risk/reward for {signal.symbol}")
            return False
        
        # Check momentum strength - lowered threshold
        momentum_score = signal.metadata.get('momentum_score', 0)
        if momentum_score < 1.5:  # Reduced from 3
            logger.warning(f"Weak momentum for {signal.symbol}: {momentum_score}")
            return False
        
        return True
    
    def calculate_position_size(self, signal: StrategySignal, 
                               account_equity: float, 
                               risk_per_trade: float) -> float:
        """Calculate position size for momentum vertical."""
        
        vertical_setup = signal.metadata.get('vertical_setup', {})
        max_loss = vertical_setup.get('debit_paid', self.max_debit)
        
        max_risk_dollar = account_equity * (risk_per_trade / 100)
        
        # Position size = max risk / max loss per spread
        position_size = max_risk_dollar / max_loss if max_loss > 0 else 1
        
        # Round down to whole number of spreads
        return max(1, int(position_size))
    
    def should_exit_position(self, position: StrategyPosition, 
                            market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Determine if momentum vertical should be exited."""
        
        if position.strategy_name != self.name:
            return None
        
        # Get current spread value
        current_value = self._get_current_spread_value(position)
        vertical_setup = position.metadata.get('vertical_setup', {})
        
        # Profit target: 70-80% of max profit
        max_profit = vertical_setup.get('max_profit', 0)
        profit_target_value = max_profit * self.profit_target
        
        # Check profit target
        if current_value >= profit_target_value:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='sell_to_close',
                confidence=85.0,
                quantity=position.quantity,
                metadata={'exit_reason': 'profit_target'}
            )
        
        # Check momentum invalidation
        if position.symbol in market_data:
            df = market_data[position.symbol]
            momentum_data = self._analyze_momentum(df)
            
            # Exit if momentum has reversed
            position_direction = position.metadata.get('direction')
            current_direction = momentum_data.get('direction')
            
            if position_direction != current_direction:
                return StrategySignal(
                    symbol=position.symbol,
                    signal_type='exit',
                    action='sell_to_close',
                    confidence=90.0,
                    quantity=position.quantity,
                    metadata={'exit_reason': 'momentum_reversal'}
                )
        
        # Check time-based exit (day before expiration)
        expiry_date = position.metadata.get('expiration_date')
        if expiry_date:
            try:
                expiry = datetime.strptime(str(expiry_date), '%Y-%m-%d')
                days_to_expiry = (expiry.date() - datetime.now().date()).days
                
                if days_to_expiry <= 1:
                    return StrategySignal(
                        symbol=position.symbol,
                        signal_type='exit',
                        action='sell_to_close',
                        confidence=100.0,
                        quantity=position.quantity,
                        metadata={'exit_reason': 'expiration_approaching'}
                    )
            except (ValueError, TypeError):
                logger.error(f"Invalid expiration date for {position.symbol}")
        
        return None
    
    def _get_current_spread_value(self, position: StrategyPosition) -> float:
        """Get current value of vertical spread."""
        
        try:
            # Get current options prices
            options_chain = self.data_client.get_options_chain(position.symbol)
            if not options_chain:
                return 0
            
            vertical_setup = position.metadata.get('vertical_setup', {})
            buy_symbol = vertical_setup.get('buy_symbol')
            sell_symbol = vertical_setup.get('sell_symbol')
            
            buy_value = 0
            sell_value = 0
            
            for opt in options_chain:
                if opt['symbol'] == buy_symbol:
                    buy_value = opt.get('bid_price', 0)  # We can sell to close
                elif opt['symbol'] == sell_symbol:
                    sell_value = opt.get('ask_price', 0)  # We need to buy to close
            
            # Current spread value = long value - short value
            current_spread_value = buy_value - sell_value
            return max(current_spread_value, 0)
            
        except Exception as e:
            logger.error(f"Error getting current spread value for {position.symbol}: {e}")
            return 0