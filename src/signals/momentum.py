import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumSignals:
    """Generate momentum-based trading signals."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def get_breakout_signals(self, df: pd.DataFrame, lookback: int = 20, 
                           volume_threshold: float = 1.5) -> pd.DataFrame:
        """Generate breakout signals based on price and volume."""
        if df.empty:
            return pd.DataFrame()
        
        try:
            signals = df.copy()
            
            # Price breakout above recent highs
            signals['resistance'] = df['high'].rolling(window=lookback).max().shift(1)
            signals['support'] = df['low'].rolling(window=lookback).min().shift(1)
            
            # Volume confirmation
            signals['avg_volume'] = df['volume'].rolling(window=lookback).mean().shift(1)
            signals['volume_ratio'] = df['volume'] / signals['avg_volume']
            
            # Breakout conditions
            signals['price_breakout'] = df['close'] > signals['resistance']
            signals['volume_breakout'] = signals['volume_ratio'] > volume_threshold
            
            # Combined signal
            signals['breakout_signal'] = signals['price_breakout'] & signals['volume_breakout']
            
            # Signal strength (0-100)
            price_strength = ((df['close'] - signals['resistance']) / signals['resistance'] * 100).clip(0, 100)
            volume_strength = ((signals['volume_ratio'] - 1) * 50).clip(0, 100)
            signals['signal_strength'] = (price_strength + volume_strength) / 2
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating breakout signals: {e}")
            return df
    
    def get_momentum_score(self, df: pd.DataFrame, short_window: int = 5, 
                          long_window: int = 20) -> pd.Series:
        """Calculate momentum score based on multiple factors."""
        if df.empty or len(df) < long_window:
            return pd.Series(dtype=float, index=df.index)
        
        try:
            # Price momentum
            price_change_short = df['close'].pct_change(short_window) * 100
            price_change_long = df['close'].pct_change(long_window) * 100
            
            # Volume momentum
            volume_ratio = df['volume'] / df['volume'].rolling(long_window).mean()
            
            # Technical momentum
            df_with_indicators = self.indicators.calculate_all_indicators(df)
            rsi_momentum = (df_with_indicators['rsi_14'] - 50) / 50  # Normalize RSI
            
            # Combine factors
            momentum_score = (
                price_change_short * 0.3 +
                price_change_long * 0.3 +
                (volume_ratio - 1) * 20 * 0.2 +  # Scale volume ratio
                rsi_momentum * 50 * 0.2  # Scale RSI momentum
            )
            
            return momentum_score.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return pd.Series(0, index=df.index)
    
    def identify_trend_direction(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Identify trend direction using multiple timeframes."""
        if df.empty:
            return pd.Series(dtype=str, index=df.index)
        
        try:
            # Short-term trend
            short_sma = df['close'].rolling(window=window//2).mean()
            medium_sma = df['close'].rolling(window=window).mean()
            long_sma = df['close'].rolling(window=window*2).mean()
            
            # Trend conditions
            uptrend = (df['close'] > short_sma) & (short_sma > medium_sma) & (medium_sma > long_sma)
            downtrend = (df['close'] < short_sma) & (short_sma < medium_sma) & (medium_sma < long_sma)
            
            # Assign trend labels
            trend = pd.Series('sideways', index=df.index)
            trend[uptrend] = 'uptrend'
            trend[downtrend] = 'downtrend'
            
            return trend
            
        except Exception as e:
            logger.error(f"Error identifying trend direction: {e}")
            return pd.Series('unknown', index=df.index)
    
    def get_reversal_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify potential reversal signals."""
        if df.empty:
            return pd.DataFrame()
        
        try:
            signals = df.copy()
            
            # Add technical indicators
            signals = self.indicators.calculate_all_indicators(signals)
            
            # Oversold/Overbought conditions
            signals['oversold'] = signals['rsi_14'] < 30
            signals['overbought'] = signals['rsi_14'] > 70
            
            # Bollinger Band reversals
            signals['bb_oversold'] = signals['close'] < signals['bb_lower']
            signals['bb_overbought'] = signals['close'] > signals['bb_upper']
            
            # Volume divergence
            price_change = signals['close'].pct_change(5)
            volume_change = signals['volume'].pct_change(5)
            signals['volume_divergence'] = (price_change > 0) & (volume_change < 0)
            
            # Combined reversal signals
            signals['bullish_reversal'] = (
                signals['oversold'] | signals['bb_oversold']
            ) & signals['volume_divergence']
            
            signals['bearish_reversal'] = (
                signals['overbought'] | signals['bb_overbought']
            ) & signals['volume_divergence']
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating reversal signals: {e}")
            return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels."""
        if df.empty:
            return {}
        
        try:
            recent_data = df.tail(window * 2)  # Use more data for better levels
            
            # Find local highs and lows
            highs = recent_data['high'].rolling(window=5, center=True).max()
            lows = recent_data['low'].rolling(window=5, center=True).min()
            
            # Identify significant levels
            resistance_levels = highs[highs == recent_data['high']].dropna()
            support_levels = lows[lows == recent_data['low']].dropna()
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            resistance = resistance_levels[resistance_levels > current_price].min()
            support = support_levels[support_levels < current_price].max()
            
            return {
                'resistance': resistance if not pd.isna(resistance) else current_price * 1.05,
                'support': support if not pd.isna(support) else current_price * 0.95,
                'current_price': current_price,
                'distance_to_resistance': (resistance / current_price - 1) * 100 if not pd.isna(resistance) else 5.0,
                'distance_to_support': (current_price / support - 1) * 100 if not pd.isna(support) else 5.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def get_entry_signals(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame],
                         signal_type: str = 'breakout') -> Dict[str, Dict]:
        """Get entry signals for multiple symbols."""
        signals = {}
        
        for symbol in symbols:
            if symbol not in data_dict or data_dict[symbol].empty:
                continue
            
            try:
                df = data_dict[symbol]
                
                if signal_type == 'breakout':
                    signal_data = self.get_breakout_signals(df)
                    
                    # Check for recent signal
                    if not signal_data.empty and signal_data['breakout_signal'].iloc[-1]:
                        signals[symbol] = {
                            'signal_type': 'breakout',
                            'strength': signal_data['signal_strength'].iloc[-1],
                            'entry_price': df['close'].iloc[-1],
                            'resistance': signal_data['resistance'].iloc[-1],
                            'support': signal_data['support'].iloc[-1],
                            'volume_ratio': signal_data['volume_ratio'].iloc[-1],
                            'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else datetime.now()
                        }
                
                elif signal_type == 'momentum':
                    momentum_score = self.get_momentum_score(df)
                    
                    # Check for strong momentum
                    if not momentum_score.empty and momentum_score.iloc[-1] > 5:  # Threshold for strong momentum
                        signals[symbol] = {
                            'signal_type': 'momentum',
                            'momentum_score': momentum_score.iloc[-1],
                            'entry_price': df['close'].iloc[-1],
                            'trend': self.identify_trend_direction(df).iloc[-1],
                            'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else datetime.now()
                        }
                
                elif signal_type == 'reversal':
                    reversal_data = self.get_reversal_signals(df)
                    
                    # Check for reversal signals
                    if not reversal_data.empty and (
                        reversal_data['bullish_reversal'].iloc[-1] or
                        reversal_data['bearish_reversal'].iloc[-1]
                    ):
                        signals[symbol] = {
                            'signal_type': 'reversal',
                            'bullish': reversal_data['bullish_reversal'].iloc[-1],
                            'bearish': reversal_data['bearish_reversal'].iloc[-1],
                            'entry_price': df['close'].iloc[-1],
                            'rsi': reversal_data['rsi_14'].iloc[-1],
                            'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else datetime.now()
                        }
                
            except Exception as e:
                logger.error(f"Error generating entry signal for {symbol}: {e}")
        
        return signals
    
    def validate_signal_quality(self, symbol: str, signal_data: Dict, 
                               min_volume: int = 100000) -> bool:
        """Validate signal quality before entry."""
        try:
            # Volume validation
            if 'volume_ratio' in signal_data and signal_data['volume_ratio'] < 1.2:
                logger.debug(f"Low volume ratio for {symbol}: {signal_data['volume_ratio']}")
                return False
            
            # Strength validation
            if 'strength' in signal_data and signal_data['strength'] < 20:
                logger.debug(f"Low signal strength for {symbol}: {signal_data['strength']}")
                return False
            
            # Momentum validation
            if 'momentum_score' in signal_data and signal_data['momentum_score'] < 3:
                logger.debug(f"Low momentum score for {symbol}: {signal_data['momentum_score']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal for {symbol}: {e}")
            return False