import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base import Strategy, StrategySignal, StrategyPosition, PositionSide, OrderType
from signals.indicators import TechnicalIndicators
from signals.momentum import MomentumSignals
from data.alpaca_client import AlpacaDataClient
from data.historical import HistoricalDataFetcher

logger = logging.getLogger(__name__)


class FractionalBreakoutStrategy(Strategy):
    """Fractional equity breakout strategy using small position sizes."""
    
    def __init__(self, config: Dict[str, Any], data_client: AlpacaDataClient):
        super().__init__("fractional_breakout", config)
        self.data_client = data_client
        self.historical_fetcher = HistoricalDataFetcher(data_client)
        self.indicators = TechnicalIndicators()
        self.momentum_signals = MomentumSignals()
        
        # Strategy parameters
        self.universe = config.get('universe', [])
        self.position_size_usd = config.get('position_size_usd', 30)
        self.atr_window = config.get('atr_window', 14)
        self.atr_stop_multiplier = config.get('atr_stop_multiplier', 1.0)
        self.atr_target_multiplier = config.get('atr_target_multiplier', 2.0)
        self.breakout_lookback = config.get('breakout_lookback', 20)
        self.min_volume = config.get('min_volume', 500000)
        
        logger.info(f"Initialized Fractional Breakout strategy with {len(self.universe)} symbols")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        current_positions: Dict[str, Any]) -> List[StrategySignal]:
        """Generate fractional breakout entry signals."""
        signals = []
        
        if not self.enabled:
            return signals
        
        for symbol in self.universe:
            try:
                # Skip if we already have a position
                if self.has_position(symbol):
                    continue
                
                # Get historical data
                if symbol not in market_data or market_data[symbol].empty:
                    continue
                
                df = market_data[symbol]
                
                # Check if symbol supports fractional trading
                if not self._is_fractionable(symbol):
                    logger.debug(f"{symbol} not fractionable, skipping")
                    continue
                
                # Analyze breakout conditions
                breakout_data = self._analyze_breakout(df)
                if not breakout_data['has_breakout']:
                    continue
                
                # Validate volume and liquidity
                if not self._validate_volume_liquidity(df):
                    continue
                
                # Get current price and calculate position details
                current_price = df['close'].iloc[-1]
                position_details = self._calculate_position_details(df, current_price)
                
                if position_details:
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type='entry',
                        action='buy',
                        confidence=self._calculate_signal_confidence(breakout_data, position_details),
                        quantity=position_details['quantity'],
                        price=current_price,
                        order_type=OrderType.MARKET,
                        stop_loss=position_details['stop_loss'],
                        take_profit=position_details['take_profit'],
                        metadata={
                            'strategy': 'fractional_breakout',
                            'breakout_strength': breakout_data['breakout_strength'],
                            'volume_ratio': breakout_data['volume_ratio'],
                            'atr': position_details['atr'],
                            'position_size_usd': self.position_size_usd,
                            'support_level': breakout_data['support_level'],
                            'resistance_level': breakout_data['resistance_level'],
                            'risk_reward_ratio': position_details['risk_reward_ratio']
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"Generated fractional breakout signal for {symbol}: ${self.position_size_usd} position")
                
            except Exception as e:
                logger.error(f"Error generating fractional breakout signal for {symbol}: {e}")
        
        return signals
    
    def _is_fractionable(self, symbol: str) -> bool:
        """Check if symbol supports fractional trading."""
        try:
            asset_info = self.data_client.get_asset_info(symbol)
            return asset_info.get('fractionable', False)
        except Exception as e:
            logger.warning(f"Could not check fractionable status for {symbol}: {e}")
            # Assume fractionable for major stocks in our universe
            major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            return symbol in major_stocks
    
    def _analyze_breakout(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze breakout conditions."""
        
        # Add technical indicators
        df_with_indicators = self.indicators.calculate_all_indicators(df)
        
        # Get breakout signals
        breakout_signals = self.momentum_signals.get_breakout_signals(
            df_with_indicators, self.breakout_lookback, volume_threshold=1.3
        )
        
        # Check for recent breakout
        has_breakout = False
        breakout_strength = 0
        volume_ratio = 1.0
        
        if not breakout_signals.empty:
            # Look for breakout in last 3 bars
            recent_breakout = breakout_signals['breakout_signal'].tail(3).any()
            
            if recent_breakout:
                has_breakout = True
                recent_data = breakout_signals.tail(3)
                breakout_strength = recent_data['signal_strength'].max()
                volume_ratio = recent_data['volume_ratio'].iloc[-1]
        
        # Calculate support and resistance levels
        support_resistance = self.indicators.get_support_resistance_levels(df)
        
        # Check price position relative to breakout level
        current_price = df['close'].iloc[-1]
        resistance = support_resistance.get('resistance', current_price * 1.02)
        
        # Breakout confirmation: price should be above resistance with volume
        price_breakout = current_price > resistance * 0.999  # Allow small tolerance
        
        return {
            'has_breakout': has_breakout and price_breakout,
            'breakout_strength': breakout_strength,
            'volume_ratio': volume_ratio,
            'support_level': support_resistance.get('support', current_price * 0.95),
            'resistance_level': resistance,
            'current_price': current_price
        }
    
    def _validate_volume_liquidity(self, df: pd.DataFrame) -> bool:
        """Validate volume and liquidity criteria."""
        
        if len(df) < self.atr_window:
            return False
        
        # Check minimum volume
        recent_volume = df['volume'].tail(5).mean()
        if recent_volume < self.min_volume:
            logger.debug(f"Volume too low: {recent_volume}")
            return False
        
        # Check for volume spike
        avg_volume = df['volume'].tail(20).mean()
        volume_spike = recent_volume / avg_volume
        
        if volume_spike < 1.1:  # Reduced from 1.2 to 1.1
            logger.debug(f"No volume spike: {volume_spike}")
            return False
        
        # Check price volatility (ATR)
        atr = self.indicators.atr(df['high'], df['low'], df['close'], self.atr_window)
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        
        if current_atr < df['close'].iloc[-1] * 0.01:  # ATR should be at least 1% of price
            logger.debug(f"ATR too low: {current_atr}")
            return False
        
        return True
    
    def _calculate_position_details(self, df: pd.DataFrame, 
                                   current_price: float) -> Optional[Dict[str, Any]]:
        """Calculate position size and risk management levels."""
        
        # Calculate ATR for stop loss and take profit
        atr = self.indicators.atr(df['high'], df['low'], df['close'], self.atr_window)
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        
        # Calculate position size based on dollar amount
        notional_quantity = self.position_size_usd / current_price
        
        # For fractional shares, we can use exact notional amount
        # Round to 6 decimal places for precision
        quantity = round(notional_quantity, 6)
        
        if quantity <= 0:
            return None
        
        # Calculate stop loss and take profit based on ATR
        stop_loss = current_price - (current_atr * self.atr_stop_multiplier)
        take_profit = current_price + (current_atr * self.atr_target_multiplier)
        
        # Ensure stop loss is not too close to current price
        min_stop_distance = current_price * 0.02  # 2% minimum
        if current_price - stop_loss < min_stop_distance:
            stop_loss = current_price - min_stop_distance
        
        # Calculate risk/reward ratio
        risk_per_share = current_price - stop_loss
        reward_per_share = take_profit - current_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        # Only proceed if risk/reward is favorable - lowered threshold
        if risk_reward_ratio < 1.2:  # Reduced from 1.5:1 to 1.2:1
            logger.debug(f"Poor risk/reward ratio: {risk_reward_ratio}")
            return None
        
        return {
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': current_atr,
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'risk_reward_ratio': risk_reward_ratio,
            'total_risk': risk_per_share * quantity
        }
    
    def _calculate_signal_confidence(self, breakout_data: Dict[str, float], 
                                   position_details: Dict[str, Any]) -> float:
        """Calculate signal confidence."""
        confidence = 60.0  # Base confidence for breakout strategy
        
        # Breakout strength contribution
        breakout_strength = breakout_data.get('breakout_strength', 0)
        confidence += min(breakout_strength * 0.2, 15)  # Cap at 15 points
        
        # Volume contribution
        volume_ratio = breakout_data.get('volume_ratio', 1)
        confidence += min((volume_ratio - 1) * 15, 10)  # Cap at 10 points
        
        # Risk/reward contribution
        risk_reward = position_details.get('risk_reward_ratio', 1)
        confidence += min((risk_reward - 1) * 10, 15)  # Cap at 15 points
        
        # Price position contribution (how far above resistance)
        current_price = breakout_data.get('current_price', 0)
        resistance = breakout_data.get('resistance_level', current_price)
        
        if resistance > 0:
            breakout_margin = (current_price - resistance) / resistance
            confidence += min(breakout_margin * 200, 10)  # Cap at 10 points
        
        return min(confidence, 90.0)  # Cap total confidence at 90%
    
    def validate_signal(self, signal: StrategySignal, 
                       market_data: Dict[str, Any]) -> bool:
        """Validate fractional breakout signal."""
        
        if signal.metadata.get('strategy') != 'fractional_breakout':
            return False
        
        # Check position size
        if signal.quantity <= 0:
            logger.warning(f"Invalid quantity for {signal.symbol}: {signal.quantity}")
            return False
        
        # Check risk/reward ratio - lowered threshold
        risk_reward = signal.metadata.get('risk_reward_ratio', 0)
        if risk_reward < 1.2:  # Reduced from 1.5 to 1.2
            logger.warning(f"Poor risk/reward for {signal.symbol}: {risk_reward}")
            return False
        
        # Check stop loss and take profit levels
        if signal.stop_loss is None or signal.take_profit is None:
            logger.warning(f"Missing stop loss or take profit for {signal.symbol}")
            return False
        
        if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
            logger.warning(f"Invalid stop/target levels for {signal.symbol}")
            return False
        
        # Check breakout strength - lowered threshold
        breakout_strength = signal.metadata.get('breakout_strength', 0)
        if breakout_strength < 5:  # Reduced from 10 to 5
            logger.warning(f"Weak breakout for {signal.symbol}: {breakout_strength}")
            return False
        
        return True
    
    def calculate_position_size(self, signal: StrategySignal, 
                               account_equity: float, 
                               risk_per_trade: float) -> float:
        """Position size is already calculated in the signal."""
        return signal.quantity
    
    def should_exit_position(self, position: StrategyPosition, 
                            market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Determine if fractional breakout position should be exited."""
        
        if position.strategy_name != self.name:
            return None
        
        current_price = position.current_price
        
        # Check stop loss
        if position.should_exit_loss():
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='sell',
                confidence=100.0,
                quantity=position.quantity,
                price=current_price,
                order_type=OrderType.MARKET,
                metadata={'exit_reason': 'stop_loss'}
            )
        
        # Check take profit
        if position.should_exit_profit():
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='sell',
                confidence=95.0,
                quantity=position.quantity,
                price=current_price,
                order_type=OrderType.MARKET,
                metadata={'exit_reason': 'take_profit'}
            )
        
        # Check for trend reversal
        if position.symbol in market_data:
            reversal_signal = self._check_trend_reversal(position, market_data[position.symbol])
            if reversal_signal:
                return reversal_signal
        
        # Check time-based exit (hold for maximum 5 days)
        position_age = datetime.now() - position.entry_time
        if position_age.days >= 5:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                action='sell',
                confidence=80.0,
                quantity=position.quantity,
                price=current_price,
                order_type=OrderType.MARKET,
                metadata={'exit_reason': 'time_limit'}
            )
        
        return None
    
    def _check_trend_reversal(self, position: StrategyPosition, 
                            df: pd.DataFrame) -> Optional[StrategySignal]:
        """Check for trend reversal signals."""
        
        if len(df) < 20:
            return None
        
        try:
            # Calculate recent momentum
            recent_momentum = self.momentum_signals.get_momentum_score(df.tail(10))
            
            if not recent_momentum.empty:
                momentum_score = recent_momentum.iloc[-1]
                
                # Exit if momentum has turned significantly negative
                if momentum_score < -5:  # Strong negative momentum
                    return StrategySignal(
                        symbol=position.symbol,
                        signal_type='exit',
                        action='sell',
                        confidence=85.0,
                        quantity=position.quantity,
                        price=position.current_price,
                        order_type=OrderType.MARKET,
                        metadata={'exit_reason': 'momentum_reversal', 'momentum_score': momentum_score}
                    )
            
            # Check for breakdown below support
            support_resistance = self.indicators.get_support_resistance_levels(df)
            support_level = support_resistance.get('support', position.current_price * 0.95)
            
            if position.current_price < support_level * 0.995:  # Below support with tolerance
                return StrategySignal(
                    symbol=position.symbol,
                    signal_type='exit',
                    action='sell',
                    confidence=90.0,
                    quantity=position.quantity,
                    price=position.current_price,
                    order_type=OrderType.MARKET,
                    metadata={'exit_reason': 'support_breakdown', 'support_level': support_level}
                )
        
        except Exception as e:
            logger.error(f"Error checking trend reversal for {position.symbol}: {e}")
        
        return None
    
    def get_position_status(self, position: StrategyPosition) -> Dict[str, Any]:
        """Get detailed status of a fractional breakout position."""
        
        # Calculate current metrics
        current_risk = abs(position.current_price - position.stop_loss) * position.quantity
        current_reward = abs(position.take_profit - position.current_price) * position.quantity
        
        return {
            'symbol': position.symbol,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'current_risk': current_risk,
            'current_reward': current_reward,
            'days_held': (datetime.now() - position.entry_time).days,
            'distance_to_stop_pct': ((position.current_price - position.stop_loss) / position.current_price) * 100,
            'distance_to_target_pct': ((position.take_profit - position.current_price) / position.current_price) * 100
        }