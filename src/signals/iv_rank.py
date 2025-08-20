import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from data.alpaca_client import AlpacaDataClient

logger = logging.getLogger(__name__)


class IVRankCalculator:
    """Calculate implied volatility rank for options strategies."""
    
    def __init__(self, data_client: AlpacaDataClient):
        self.data_client = data_client
        self._iv_cache = {}
    
    def calculate_iv_rank(self, symbol: str, lookback_days: int = 20) -> Dict[str, float]:
        """Calculate IV rank for a symbol."""
        try:
            # Get options chain
            chain = self.data_client.get_options_chain(symbol)
            
            if not chain:
                logger.warning(f"No options chain data for {symbol}")
                return {}
            
            # Filter for valid IV data
            valid_options = [opt for opt in chain if opt.get('implied_volatility') is not None]
            
            if not valid_options:
                logger.warning(f"No valid IV data for {symbol}")
                return {}
            
            # Get current stock price for ATM selection
            quotes = self.data_client.get_stock_quotes([symbol])
            if symbol not in quotes:
                logger.warning(f"No quote data for {symbol}")
                return {}
            
            current_price = (quotes[symbol]['bid_price'] + quotes[symbol]['ask_price']) / 2
            
            # Filter for ATM options (within 5% of current price)
            atm_options = [
                opt for opt in valid_options
                if abs(opt['strike_price'] - current_price) / current_price <= 0.05
            ]
            
            if not atm_options:
                logger.warning(f"No ATM options for {symbol}")
                return {}
            
            # Calculate current IV (average of ATM options)
            current_iv = np.mean([opt['implied_volatility'] for opt in atm_options])
            
            # Get historical IV data
            historical_iv = self._get_historical_iv(symbol, lookback_days)
            
            if not historical_iv:
                return {'current_iv': current_iv, 'iv_rank': 50.0}  # Default to 50% if no history
            
            # Calculate IV rank
            iv_values = list(historical_iv.values()) + [current_iv]
            iv_rank = (sum(1 for iv in iv_values if iv <= current_iv) / len(iv_values)) * 100
            
            # Calculate additional metrics
            iv_mean = np.mean(iv_values)
            iv_std = np.std(iv_values)
            iv_zscore = (current_iv - iv_mean) / iv_std if iv_std > 0 else 0
            
            return {
                'current_iv': current_iv,
                'iv_rank': iv_rank,
                'iv_percentile': iv_rank,  # Same as rank for our purposes
                'iv_mean': iv_mean,
                'iv_std': iv_std,
                'iv_zscore': iv_zscore,
                'iv_elevated': iv_rank > 70,  # Consider elevated if > 70th percentile
                'sample_size': len(iv_values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV rank for {symbol}: {e}")
            return {}
    
    def _get_historical_iv(self, symbol: str, days: int) -> Dict[str, float]:
        """Get historical IV data (simplified approximation)."""
        try:
            # In a production system, this would fetch historical options data
            # For now, we'll estimate IV from historical volatility
            
            # Get historical stock price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)  # Get more data for calculation
            
            from alpaca.data.timeframe import TimeFrame
            df = self.data_client.get_stock_bars([symbol], TimeFrame.Day, start_date, end_date)
            
            if df.empty:
                return {}
            
            # Extract symbol data
            if symbol in df.index.get_level_values(0):
                symbol_data = df.loc[symbol]
            else:
                symbol_data = df
            
            if len(symbol_data) < days:
                return {}
            
            # Calculate rolling historical volatility as IV proxy
            returns = symbol_data['close'].pct_change().dropna()
            
            # Calculate rolling volatility for each day
            historical_iv = {}
            for i in range(days, len(returns)):
                period_returns = returns.iloc[i-days:i]
                daily_vol = period_returns.std()
                annualized_vol = daily_vol * np.sqrt(252)  # Annualize
                
                # Convert to IV-like format (as percentage)
                historical_iv[symbol_data.index[i].strftime('%Y-%m-%d')] = annualized_vol * 100
            
            return historical_iv
            
        except Exception as e:
            logger.error(f"Error getting historical IV for {symbol}: {e}")
            return {}
    
    def get_iv_signals(self, symbols: List[str], 
                      min_iv_rank: float = 70.0) -> Dict[str, Dict[str, float]]:
        """Get IV signals for multiple symbols."""
        signals = {}
        
        for symbol in symbols:
            try:
                iv_data = self.calculate_iv_rank(symbol)
                
                if iv_data and iv_data.get('iv_rank', 0) >= min_iv_rank:
                    signals[symbol] = iv_data
                    logger.info(f"IV signal for {symbol}: rank={iv_data.get('iv_rank', 0):.1f}%")
                
            except Exception as e:
                logger.error(f"Error getting IV signal for {symbol}: {e}")
        
        return signals
    
    def is_iv_elevated(self, symbol: str, threshold: float = 70.0) -> bool:
        """Check if IV is elevated for a symbol."""
        try:
            iv_data = self.calculate_iv_rank(symbol)
            return iv_data.get('iv_rank', 0) > threshold
        except Exception as e:
            logger.error(f"Error checking IV elevation for {symbol}: {e}")
            return False
    
    def get_iv_term_structure(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get IV term structure across different expirations."""
        try:
            chain = self.data_client.get_options_chain(symbol)
            
            if not chain:
                return {}
            
            # Group by expiration date
            by_expiry = {}
            for option in chain:
                if option.get('implied_volatility') is None:
                    continue
                
                expiry = option['expiration_date']
                if expiry not in by_expiry:
                    by_expiry[expiry] = []
                
                by_expiry[expiry].append(option['implied_volatility'])
            
            # Calculate average IV for each expiration
            term_structure = {}
            for expiry, iv_list in by_expiry.items():
                if iv_list:
                    term_structure[expiry] = {
                        'avg_iv': np.mean(iv_list),
                        'min_iv': np.min(iv_list),
                        'max_iv': np.max(iv_list),
                        'count': len(iv_list)
                    }
            
            return term_structure
            
        except Exception as e:
            logger.error(f"Error getting IV term structure for {symbol}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the IV data cache."""
        self._iv_cache.clear()
        logger.info("IV cache cleared")