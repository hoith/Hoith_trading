import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import talib, fallback to manual calculations if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not available, using fallback calculations")


class TechnicalIndicators:
    """Technical analysis indicators for trading strategies."""
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if HAS_TALIB:
            try:
                return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=window), index=close.index)
            except Exception as e:
                logger.error(f"Error calculating ATR with talib: {e}")
        
        # Fallback calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if HAS_TALIB:
            try:
                return pd.Series(talib.RSI(close.values, timeperiod=window), index=close.index)
            except Exception as e:
                logger.error(f"Error calculating RSI with talib: {e}")
        
        # Fallback calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if HAS_TALIB:
            try:
                upper, middle, lower = talib.BBANDS(close.values, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev)
                return pd.Series(upper, index=close.index), pd.Series(middle, index=close.index), pd.Series(lower, index=close.index)
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")
        
        # Fallback calculation
        middle = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        if HAS_TALIB:
            try:
                macd, signal_line, histogram = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                return (pd.Series(macd, index=close.index), 
                       pd.Series(signal_line, index=close.index), 
                       pd.Series(histogram, index=close.index))
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")
        
        # Fallback calculation
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        if HAS_TALIB:
            try:
                k, d = talib.STOCH(high.values, low.values, close.values, 
                                  fastk_period=k_window, slowk_period=d_window, slowd_period=d_window)
                return pd.Series(k, index=close.index), pd.Series(d, index=close.index)
            except Exception as e:
                logger.error(f"Error calculating Stochastic: {e}")
        
        # Fallback calculation
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_window).mean()
        return k, d
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        if HAS_TALIB:
            try:
                return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=window), index=close.index)
            except Exception as e:
                logger.error(f"Error calculating Williams %R: {e}")
        
        # Fallback calculation
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volume simple moving average."""
        return volume.rolling(window=window).mean()
    
    @staticmethod
    def price_momentum(close: pd.Series, window: int = 10) -> pd.Series:
        """Calculate price momentum."""
        return close.pct_change(periods=window) * 100
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Donchian Channels."""
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        return upper, lower
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                        window: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        ema = close.ewm(span=window).mean()
        atr = TechnicalIndicators.atr(high, low, close, window)
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        return upper, ema, lower
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame."""
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.warning("Insufficient data for indicator calculation")
            return df
        
        try:
            result = df.copy()
            
            # Trend indicators
            result['atr_14'] = self.atr(df['high'], df['low'], df['close'], 14)
            result['atr_20'] = self.atr(df['high'], df['low'], df['close'], 20)
            
            # Momentum indicators
            result['rsi_14'] = self.rsi(df['close'], 14)
            result['momentum_10'] = self.price_momentum(df['close'], 10)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.bollinger_bands(df['close'], 20, 2)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            macd, signal, histogram = self.macd(df['close'])
            result['macd'] = macd
            result['macd_signal'] = signal
            result['macd_histogram'] = histogram
            
            # Stochastic
            stoch_k, stoch_d = self.stochastic(df['high'], df['low'], df['close'])
            result['stoch_k'] = stoch_k
            result['stoch_d'] = stoch_d
            
            # Volume indicators
            result['volume_sma_20'] = self.volume_sma(df['volume'], 20)
            result['volume_ratio'] = df['volume'] / result['volume_sma_20']
            
            # Support/Resistance levels
            donch_upper, donch_lower = self.donchian_channels(df['high'], df['low'], 20)
            result['donch_upper'] = donch_upper
            result['donch_lower'] = donch_lower
            
            # Price position in range
            result['price_position'] = (df['close'] - donch_lower) / (donch_upper - donch_lower)
            
            # Volatility measures
            result['price_volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            logger.debug(f"Calculated indicators for {len(result)} bars")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    @staticmethod
    def get_breakout_signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Identify breakout signals."""
        if df.empty or 'close' not in df.columns:
            return pd.Series(dtype=bool, index=df.index)
        
        try:
            # Price breakouts above recent highs
            recent_high = df['high'].rolling(window=lookback).max().shift(1)
            breakout_up = df['close'] > recent_high
            
            # Volume confirmation
            avg_volume = df['volume'].rolling(window=lookback).mean().shift(1)
            volume_confirm = df['volume'] > avg_volume * 1.5
            
            return breakout_up & volume_confirm
            
        except Exception as e:
            logger.error(f"Error calculating breakout signals: {e}")
            return pd.Series(False, index=df.index)
    
    @staticmethod
    def get_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels."""
        if df.empty:
            return {}
        
        try:
            recent_data = df.tail(window)
            
            return {
                'resistance': recent_data['high'].max(),
                'support': recent_data['low'].min(),
                'pivot': (recent_data['high'].max() + recent_data['low'].min() + recent_data['close'].iloc[-1]) / 3,
                'current_price': recent_data['close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}