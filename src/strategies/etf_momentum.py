import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base import Strategy, StrategySignal, StrategyPosition, PositionSide, OrderType
from signals.momentum import MomentumSignals
from signals.indicators import TechnicalIndicators
from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


class ETFMomentumStrategy(Strategy):
    """ETF momentum strategy for QQQ, SPY, and TQQQ with relative strength and volatility filtering."""
    
    def __init__(self, config: Dict[str, Any], data_client: AlpacaDataClient):
        super().__init__("etf_momentum", config)
        self.data_client = data_client
        self.momentum_signals = MomentumSignals()
        self.indicators = TechnicalIndicators()
        
        # Strategy parameters
        self.universe = config.get('universe', ['QQQ', 'SPY', 'TQQQ'])
        self.position_size_usd = config.get('position_size_usd', 100)
        self.momentum_lookback = config.get('momentum_lookback', 20)
        self.volatility_lookback = config.get('volatility_lookback', 14)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.min_volume = config.get('min_volume', 1000000)
        self.max_hold_days = config.get('max_hold_days', 10)
        
        # Risk management
        self.max_drawdown_pct = config.get('max_drawdown_pct', 3.0)
        self.profit_target_pct = config.get('profit_target_pct', 5.0)
        self.stop_loss_pct = config.get('stop_loss_pct', 2.0)
        
        logger.info(f"Initialized ETF Momentum strategy with {len(self.universe)} ETFs")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        current_positions: Dict[str, Any]) -> List[StrategySignal]:
        """Generate ETF momentum entry signals."""
        signals = []
        
        if not self.enabled:
            return signals
        
        # Calculate relative strength across ETFs
        relative_strengths = self._calculate_relative_strength(market_data)
        
        for symbol in self.universe:
            try:
                # Skip if we already have a position
                if self.has_position(symbol):
                    continue
                
                # Get historical data
                if symbol not in market_data or market_data[symbol].empty:
                    continue
                
                df = market_data[symbol]
                
                if len(df) < max(self.momentum_lookback, self.volatility_lookback, self.rsi_period):
                    continue
                
                # Analyze momentum and technical conditions
                analysis = self._analyze_etf_conditions(df, symbol, relative_strengths)
                
                if not analysis['is_valid']:
                    continue
                
                # Generate signal based on ETF type
                signal_type = self._determine_signal_type(symbol, analysis)
                
                if signal_type:
                    current_price = df['close'].iloc[-1]
                    
                    # Calculate position size and risk levels
                    position_details = self._calculate_position_details(df, current_price, symbol)
                    
                    if position_details:
                        signal = StrategySignal(
                            symbol=symbol,
                            signal_type='entry',
                            action='buy',
                            confidence=self._calculate_signal_confidence(analysis, symbol),
                            quantity=position_details['quantity'],
                            price=current_price,
                            order_type=OrderType.MARKET,
                            stop_loss=position_details['stop_loss'],
                            take_profit=position_details['take_profit'],
                            metadata={
                                'strategy': 'etf_momentum',
                                'etf_type': self._get_etf_type(symbol),
                                'momentum_score': analysis['momentum_score'],
                                'relative_strength': analysis['relative_strength'],
                                'rsi': analysis['rsi'],
                                'volatility': analysis['volatility'],
                                'volume_ratio': analysis['volume_ratio'],
                                'signal_type': signal_type,
                                'position_size_usd': self.position_size_usd,
                                'risk_reward_ratio': position_details['risk_reward_ratio']
                            }
                        )
                        
                        signals.append(signal)
                        logger.info(f"Generated {signal_type} signal for {symbol}: ${self.position_size_usd} position")
                
            except Exception as e:
                logger.error(f"Error generating ETF momentum signal for {symbol}: {e}")
        
        return signals
    
    def _calculate_relative_strength(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate relative strength of each ETF."""
        relative_strengths = {}
        
        # Calculate momentum for each ETF
        for symbol in self.universe:
            if symbol in market_data and not market_data[symbol].empty:
                df = market_data[symbol]
                if len(df) >= self.momentum_lookback:
                    # Calculate momentum as percentage change over lookback period
                    momentum = (df['close'].iloc[-1] / df['close'].iloc[-self.momentum_lookback] - 1) * 100
                    relative_strengths[symbol] = momentum
        
        return relative_strengths
    
    def _analyze_etf_conditions(self, df: pd.DataFrame, symbol: str, 
                               relative_strengths: Dict[str, float]) -> Dict[str, Any]:
        """Analyze technical and momentum conditions for ETF."""
        
        # Basic momentum score
        momentum_score = self.momentum_signals.get_momentum_score(df).iloc[-1] if len(df) > 20 else 0
        
        # RSI analysis
        rsi = self.indicators.rsi(df['close'], self.rsi_period).iloc[-1]
        
        # Volatility analysis
        volatility = df['close'].pct_change().rolling(self.volatility_lookback).std().iloc[-1] * np.sqrt(252) * 100
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Relative strength
        relative_strength = relative_strengths.get(symbol, 0)
        
        # Price action analysis
        price_change_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        
        # Determine if conditions are valid for entry
        is_valid = self._validate_etf_entry_conditions(
            symbol, momentum_score, rsi, volatility, volume_ratio, 
            relative_strength, price_change_5d, avg_volume
        )
        
        return {
            'is_valid': is_valid,
            'momentum_score': momentum_score,
            'rsi': rsi,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'relative_strength': relative_strength,
            'price_change_5d': price_change_5d,
            'avg_volume': avg_volume
        }
    
    def _validate_etf_entry_conditions(self, symbol: str, momentum_score: float, rsi: float, 
                                     volatility: float, volume_ratio: float, relative_strength: float,
                                     price_change_5d: float, avg_volume: float) -> bool:
        """Validate entry conditions based on ETF type."""
        
        # Basic volume check
        if avg_volume < self.min_volume:
            return False
        
        # ETF-specific logic
        etf_type = self._get_etf_type(symbol)
        
        if etf_type == 'broad_market':  # SPY
            # Conservative approach for broad market
            return (momentum_score > 2.0 and 
                   rsi < 60 and  # Not too overbought
                   volume_ratio > 1.2 and
                   price_change_5d > 1.0)  # Recent strength
        
        elif etf_type == 'tech_focused':  # QQQ
            # Moderate momentum requirements
            return (momentum_score > 3.0 and
                   rsi < 65 and
                   volume_ratio > 1.1 and
                   relative_strength > 0)  # Outperforming
        
        elif etf_type == 'leveraged':  # TQQQ
            # Higher requirements for leveraged ETF
            return (momentum_score > 5.0 and
                   rsi < 70 and
                   volume_ratio > 1.3 and
                   relative_strength > 2.0 and  # Strong relative performance
                   volatility < 50)  # Not too volatile
        
        return False
    
    def _determine_signal_type(self, symbol: str, analysis: Dict[str, Any]) -> Optional[str]:
        """Determine the type of signal to generate."""
        
        etf_type = self._get_etf_type(symbol)
        momentum_score = analysis['momentum_score']
        rsi = analysis['rsi']
        relative_strength = analysis['relative_strength']
        
        if etf_type == 'broad_market':
            # Conservative momentum signals
            if momentum_score > 3 and rsi < 55:
                return 'conservative_momentum'
        
        elif etf_type == 'tech_focused':
            # Tech momentum signals
            if momentum_score > 4 and relative_strength > 1:
                return 'tech_momentum'
        
        elif etf_type == 'leveraged':
            # Leveraged momentum (higher conviction required)
            if momentum_score > 6 and relative_strength > 3 and rsi < 65:
                return 'leveraged_momentum'
        
        return None
    
    def _get_etf_type(self, symbol: str) -> str:
        """Get ETF classification."""
        if symbol == 'SPY':
            return 'broad_market'
        elif symbol == 'QQQ':
            return 'tech_focused'
        elif symbol == 'TQQQ':
            return 'leveraged'
        else:
            return 'other'
    
    def _calculate_position_details(self, df: pd.DataFrame, current_price: float, 
                                   symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate position size and risk management levels."""
        
        # Calculate position size based on dollar amount
        quantity = self.position_size_usd / current_price
        
        # ETF-specific risk management
        etf_type = self._get_etf_type(symbol)
        
        if etf_type == 'broad_market':
            stop_pct = 1.5  # Conservative stop
            target_pct = 3.0  # Conservative target
        elif etf_type == 'tech_focused':
            stop_pct = 2.0  # Moderate stop
            target_pct = 4.0  # Moderate target
        elif etf_type == 'leveraged':
            stop_pct = 1.0  # Tight stop for leverage
            target_pct = 2.5  # Quick target for leverage
        else:
            stop_pct = 2.0
            target_pct = 4.0
        
        # Calculate stop loss and take profit
        stop_loss = current_price * (1 - stop_pct / 100)
        take_profit = current_price * (1 + target_pct / 100)
        
        # Calculate risk/reward ratio
        risk_per_share = current_price - stop_loss
        reward_per_share = take_profit - current_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        # Ensure minimum risk/reward ratio
        if risk_reward_ratio < 1.5:
            return None
        
        return {
            'quantity': round(quantity, 6),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'risk_reward_ratio': risk_reward_ratio,
            'total_risk': risk_per_share * quantity
        }
    
    def _calculate_signal_confidence(self, analysis: Dict[str, Any], symbol: str) -> float:
        """Calculate signal confidence based on analysis."""
        
        base_confidence = 60.0
        etf_type = self._get_etf_type(symbol)
        
        # Momentum contribution
        momentum_score = analysis.get('momentum_score', 0)
        base_confidence += min(momentum_score * 2, 20)
        
        # Relative strength contribution
        relative_strength = analysis.get('relative_strength', 0)
        base_confidence += min(relative_strength, 15)
        
        # Volume contribution
        volume_ratio = analysis.get('volume_ratio', 1)
        base_confidence += min((volume_ratio - 1) * 10, 10)
        
        # ETF-specific adjustments
        if etf_type == 'leveraged':
            # Higher volatility = lower confidence
            volatility = analysis.get('volatility', 20)
            if volatility > 40:
                base_confidence -= 10
        
        return min(base_confidence, 95.0)
    
    def validate_signal(self, signal: StrategySignal, 
                       market_data: Dict[str, Any]) -> bool:
        """Validate ETF momentum signal."""
        
        if signal.metadata.get('strategy') != 'etf_momentum':
            return False
        
        # Check position size
        if signal.quantity <= 0:
            logger.warning(f"Invalid quantity for {signal.symbol}: {signal.quantity}")
            return False
        
        # Check risk/reward ratio
        risk_reward = signal.metadata.get('risk_reward_ratio', 0)
        if risk_reward < 1.5:
            logger.warning(f"Poor risk/reward for {signal.symbol}: {risk_reward}")
            return False
        
        # Check stop loss and take profit levels
        if signal.stop_loss is None or signal.take_profit is None:
            logger.warning(f"Missing stop loss or take profit for {signal.symbol}")
            return False
        
        if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
            logger.warning(f"Invalid stop/target levels for {signal.symbol}")
            return False
        
        # Check momentum strength
        momentum_score = signal.metadata.get('momentum_score', 0)
        etf_type = signal.metadata.get('etf_type', 'other')
        
        min_momentum = {'broad_market': 2, 'tech_focused': 3, 'leveraged': 5}.get(etf_type, 3)
        if momentum_score < min_momentum:
            logger.warning(f"Weak momentum for {signal.symbol}: {momentum_score}")
            return False
        
        return True
    
    def calculate_position_size(self, signal: StrategySignal, 
                               account_equity: float, 
                               risk_per_trade: float) -> float:
        """Position size is already calculated in the signal."""
        return signal.quantity
    
    def should_exit_position(self, position: StrategyPosition, 
                            market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Determine if ETF position should be exited."""
        
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
        
        # Check time-based exit
        position_age = datetime.now() - position.entry_time
        if position_age.days >= self.max_hold_days:
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
        
        # Check momentum reversal for leveraged ETFs
        etf_type = position.metadata.get('etf_type', 'other')
        if etf_type == 'leveraged' and position.symbol in market_data:
            reversal_signal = self._check_leveraged_etf_reversal(position, market_data[position.symbol])
            if reversal_signal:
                return reversal_signal
        
        return None
    
    def _check_leveraged_etf_reversal(self, position: StrategyPosition, 
                                    df: pd.DataFrame) -> Optional[StrategySignal]:
        """Check for momentum reversal in leveraged ETFs (quicker exits)."""
        
        if len(df) < 10:
            return None
        
        try:
            # Quick momentum check for leveraged ETFs
            recent_momentum = self.momentum_signals.get_momentum_score(df.tail(5))
            
            if not recent_momentum.empty:
                momentum_score = recent_momentum.iloc[-1]
                
                # Exit if momentum turns negative quickly
                if momentum_score < -2:  # Quick reversal
                    return StrategySignal(
                        symbol=position.symbol,
                        signal_type='exit',
                        action='sell',
                        confidence=90.0,
                        quantity=position.quantity,
                        price=position.current_price,
                        order_type=OrderType.MARKET,
                        metadata={'exit_reason': 'momentum_reversal', 'momentum_score': momentum_score}
                    )
        
        except Exception as e:
            logger.error(f"Error checking leveraged ETF reversal for {position.symbol}: {e}")
        
        return None
    
    def get_position_status(self, position: StrategyPosition) -> Dict[str, Any]:
        """Get detailed status of an ETF momentum position."""
        
        current_risk = abs(position.current_price - position.stop_loss) * position.quantity
        current_reward = abs(position.take_profit - position.current_price) * position.quantity
        
        return {
            'symbol': position.symbol,
            'etf_type': position.metadata.get('etf_type', 'unknown'),
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
            'distance_to_target_pct': ((position.take_profit - position.current_price) / position.current_price) * 100,
            'signal_type': position.metadata.get('signal_type', 'unknown')
        }