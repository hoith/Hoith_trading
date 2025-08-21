#!/usr/bin/env python3
"""
Unified Strategy Core for Backtest <-> Live Parity
Single source of truth for features, signals, sizing and guards
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import logging
import importlib
import sys
import os

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Params:
    """Strategy parameters dataclass"""
    # Technical indicators
    lookback_sma: int = 50
    rsi_len: int = 14
    atr_len: int = 14
    
    # Trade sizing
    equity_notional: float = 20.0  # per-trade in thousands
    submit_bps: float = 6.0        # limit price offset in basis points
    
    # Liquidity guards
    min_volume_5m: float = 10_000  # minimum 5-bar average volume
    max_spread_bps: float = 10.0   # maximum bid-ask spread in basis points
    
    # Risk management
    max_position_pct: float = 40.0  # max position size as % of portfolio
    freshness_threshold_s: int = 180  # max bar age in seconds

def compute_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Compute technical features from OHLCV data
    
    Args:
        df: DataFrame with OHLCV columns
        p: Parameters object
        
    Returns:
        DataFrame with additional feature columns
    """
    try:
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        if len(df) < max(p.lookback_sma, p.rsi_len, p.atr_len):
            logger.warning(f"Insufficient data: {len(df)} bars, need {max(p.lookback_sma, p.rsi_len, p.atr_len)}")
            return df.copy()
        
        feat = df.copy()
        
        # Simple Moving Average
        feat['sma'] = feat['close'].rolling(window=p.lookback_sma).mean()
        
        # RSI
        delta = feat['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p.rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p.rsi_len).mean()
        rs = gain / loss
        feat['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = feat['high'] - feat['low']
        high_close = np.abs(feat['high'] - feat['close'].shift())
        low_close = np.abs(feat['low'] - feat['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        feat['atr'] = true_range.rolling(window=p.atr_len).mean()
        
        # Volume metrics
        feat['volume_5m'] = feat['volume'].rolling(window=5).mean()
        
        # Momentum indicators
        feat['momentum_10'] = feat['close'] / feat['close'].shift(10) - 1
        feat['momentum_20'] = feat['close'] / feat['close'].shift(20) - 1
        
        return feat
        
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        return df.copy()

def decide_entries(feat: pd.DataFrame, p: Params) -> pd.Series:
    """
    Generate entry signals (Signal at t-1 � execute at t open)
    
    Args:
        feat: DataFrame with features
        p: Parameters object
        
    Returns:
        Boolean Series indicating entry signals
    """
    try:
        # Initialize all as False
        entries = pd.Series(False, index=feat.index)
        
        if len(feat) < 2:
            return entries
        
        # Auto-hook user logic if available
        user_entries = _try_user_entries(feat, p)
        if user_entries is not None:
            return user_entries
        
        # Fallback: Simple SMA cross strategy
        # Signal: prev close <= prev SMA and current close > current SMA
        prev_close = feat['close'].shift(1)
        prev_sma = feat['sma'].shift(1)
        curr_close = feat['close']
        curr_sma = feat['sma']
        
        # SMA cross up condition
        sma_cross_up = (prev_close <= prev_sma) & (curr_close > curr_sma)
        
        # Additional momentum filter
        momentum_ok = feat['momentum_10'] > 0.01  # 1% minimum momentum
        
        # RSI not overbought
        rsi_ok = feat['rsi'] < 70
        
        # Volume confirmation
        volume_ok = feat['volume'] > feat['volume_5m'] * 1.2  # 20% above 5-bar average
        
        # Combine conditions
        entries = sma_cross_up & momentum_ok & rsi_ok & volume_ok
        
        return entries
        
    except Exception as e:
        logger.error(f"Error in decide_entries: {e}")
        return pd.Series(False, index=feat.index)

def decide_exits(feat: pd.DataFrame, p: Params) -> pd.Series:
    """
    Generate exit signals (default no-op)
    
    Args:
        feat: DataFrame with features
        p: Parameters object
        
    Returns:
        Boolean Series indicating exit signals
    """
    try:
        # Auto-hook user logic if available
        user_exits = _try_user_exits(feat, p)
        if user_exits is not None:
            return user_exits
        
        # Default: no automatic exits (rely on stops/targets)
        return pd.Series(False, index=feat.index)
        
    except Exception as e:
        logger.error(f"Error in decide_exits: {e}")
        return pd.Series(False, index=feat.index)

def size_order(price: float, p: Params, fractionable: bool = True) -> Union[float, int]:
    """
    Calculate order size based on price and parameters
    
    Args:
        price: Current price
        p: Parameters object
        fractionable: Whether asset supports fractional shares
        
    Returns:
        Order quantity (float if fractionable, int otherwise)
    """
    try:
        # Target dollar amount (convert from thousands)
        target_dollars = p.equity_notional * 1000
        
        # Calculate raw quantity
        raw_qty = target_dollars / price
        
        if fractionable:
            # Round to reasonable precision for fractional shares
            return round(raw_qty, 6)
        else:
            # Integer shares only, minimum 1
            return max(1, int(raw_qty))
        
    except Exception as e:
        logger.error(f"Error calculating order size: {e}")
        return 1 if not fractionable else 1.0

def volume5_ok(df: pd.DataFrame, p: Params) -> bool:
    """
    Check if 5-bar average volume meets minimum threshold
    
    Args:
        df: DataFrame with volume data
        p: Parameters object
        
    Returns:
        True if volume criteria met
    """
    try:
        if len(df) < 5:
            return False
        
        avg_volume_5 = df['volume'].tail(5).mean()
        return avg_volume_5 >= p.min_volume_5m
        
    except Exception as e:
        logger.error(f"Error checking volume: {e}")
        return False

def target_limit_from_last(last: float, side: str, p: Params) -> float:
    """
    Calculate limit price from last price with BPS offset
    
    Args:
        last: Last traded price
        side: "buy" or "sell"
        p: Parameters object
        
    Returns:
        Limit price
    """
    try:
        bps_offset = p.submit_bps / 10000.0  # Convert BPS to decimal
        
        if side.lower() == "buy":
            # Buy: add BPS to get filled (aggressive)
            return last * (1 + bps_offset)
        elif side.lower() == "sell":
            # Sell: subtract BPS to get filled (aggressive)  
            return last * (1 - bps_offset)
        else:
            logger.warning(f"Unknown side: {side}, returning last price")
            return last
            
    except Exception as e:
        logger.error(f"Error calculating limit price: {e}")
        return last

def _try_user_entries(feat: pd.DataFrame, p: Params) -> Optional[pd.Series]:
    """Try to import and use user's custom entry logic"""
    user_modules = [
        'final_2025_backtest',
        'fixed_2025_backtest', 
        'simple_backtest',
        'test_2025_backtest',
        'aggressive_high_return_strategy',
        'expanded_aggressive_backtest_2024'
    ]
    
    for module_name in user_modules:
        try:
            # Try to import from current directory
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                # Add parent directory to path temporarily
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                module = importlib.import_module(module_name)
            
            # Look for entry signal function
            if hasattr(module, 'compute_features') and hasattr(module, 'decide_entries'):
                logger.info(f"Using entry logic from {module_name}")
                return module.decide_entries(feat, p)
            elif hasattr(module, 'simple_momentum_strategy'):
                logger.info(f"Using momentum strategy from {module_name}")
                # Adapt to our interface
                signals = module.simple_momentum_strategy(feat)
                if isinstance(signals, list):
                    # Convert list of signals to boolean series
                    entries = pd.Series(False, index=feat.index)
                    for signal in signals:
                        if 'date' in signal and signal['date'] in feat.index:
                            entries.loc[signal['date']] = True
                    return entries
                return signals
                
        except Exception as e:
            logger.debug(f"Could not import {module_name}: {e}")
            continue
    
    return None

def _try_user_exits(feat: pd.DataFrame, p: Params) -> Optional[pd.Series]:
    """Try to import and use user's custom exit logic"""
    user_modules = [
        'final_2025_backtest',
        'fixed_2025_backtest',
        'simple_backtest', 
        'test_2025_backtest',
        'aggressive_high_return_strategy',
        'expanded_aggressive_backtest_2024'
    ]
    
    for module_name in user_modules:
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                    
                module = importlib.import_module(module_name)
            
            if hasattr(module, 'decide_exits'):
                logger.info(f"Using exit logic from {module_name}")
                return module.decide_exits(feat, p)
                
        except Exception as e:
            logger.debug(f"Could not import exit logic from {module_name}: {e}")
            continue
    
    return None

def get_bar_age_seconds(df: pd.DataFrame) -> int:
    """
    Calculate age of last bar in seconds
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Age in seconds
    """
    try:
        if df.empty:
            return 999999
        
        last_bar_time = df.index[-1]
        
        # Convert to UTC if timezone-aware
        if hasattr(last_bar_time, 'tz') and last_bar_time.tz is not None:
            now_utc = datetime.now(timezone.utc)
            if last_bar_time.tz != timezone.utc:
                last_bar_time = last_bar_time.astimezone(timezone.utc)
        else:
            now_utc = datetime.now()
            
        age = (now_utc - last_bar_time).total_seconds()
        return int(age)
        
    except Exception as e:
        logger.error(f"Error calculating bar age: {e}")
        return 999999

def log_bar_freshness(symbol: str, df: pd.DataFrame) -> None:
    """Log bar freshness information"""
    try:
        if df.empty:
            logger.warning(f"{symbol}: No data available")
            return
            
        last_bar_utc = df.index[-1]
        age_s = get_bar_age_seconds(df)
        rows = len(df)
        
        logger.info(f"{symbol} tf=1m last_bar_UTC={last_bar_utc} age_s={age_s} rows={rows}")
        
        if age_s > 300:  # 5 minutes
            logger.warning(f"{symbol}: Stale data, age={age_s}s")
            
    except Exception as e:
        logger.error(f"Error logging bar freshness for {symbol}: {e}")

def log_filter_stage(stage: str, count: int) -> None:
    """Log filter stage results"""
    logger.info(f"[FILTER] {stage} -> {count}")

def is_fresh_data(df: pd.DataFrame, p: Params) -> bool:
    """
    Check if data is fresh enough for trading
    
    Args:
        df: DataFrame with datetime index
        p: Parameters object
        
    Returns:
        True if data is fresh
    """
    age_s = get_bar_age_seconds(df)
    return age_s <= p.freshness_threshold_s