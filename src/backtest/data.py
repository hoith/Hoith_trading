"""
Data provider for backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import yfinance as yf
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class BacktestDataProvider:
    """Provides historical data for backtesting."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize data provider.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
    def get_historical_data(self, symbols: List[str], start_date: Union[str, datetime], 
                          end_date: Union[str, datetime], 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Multi-symbol DataFrame with OHLCV data
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}"
        
        if cache_key in self._data_cache:
            logger.info(f"Using cached data for {symbols}")
            return self._data_cache[cache_key]
        
        # Try to load from disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self._data_cache[cache_key] = data
                logger.info(f"Loaded cached data from {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        # Download data
        logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
        
        all_data = {}
        
        for symbol in symbols:
            try:
                # Use yfinance to download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),  # Include end date
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                
                # Ensure we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue
                
                all_data[symbol] = data[required_cols]
                
            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {e}")
                continue
        
        if not all_data:
            logger.error("No data downloaded for any symbols")
            return pd.DataFrame()
        
        # Combine all data into multi-index DataFrame
        combined_data = pd.concat(all_data, axis=1)
        
        # Forward fill missing values (up to 5 days)
        combined_data = combined_data.fillna(method='ffill', limit=5)
        
        # Cache the data
        self._data_cache[cache_key] = combined_data
        
        # Save to disk cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(combined_data, f)
            logger.info(f"Saved data to cache file {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
        
        return combined_data
    
    def get_symbol_data(self, symbol: str, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime], 
                       interval: str = "1d") -> pd.DataFrame:
        """Get historical data for a single symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        multi_data = self.get_historical_data([symbol], start_date, end_date, interval)
        
        if multi_data.empty:
            return pd.DataFrame()
        
        # Extract single symbol data
        if symbol in multi_data.columns.levels[0]:
            return multi_data[symbol]
        else:
            return pd.DataFrame()
    
    def get_benchmark_data(self, start_date: Union[str, datetime], 
                          end_date: Union[str, datetime],
                          benchmark: str = "SPY") -> pd.Series:
        """Get benchmark data for comparison.
        
        Args:
            benchmark: Benchmark symbol (default SPY)
            start_date: Start date
            end_date: End date
            
        Returns:
            Series with benchmark returns
        """
        data = self.get_symbol_data(benchmark, start_date, end_date)
        
        if data.empty:
            return pd.Series()
        
        # Calculate daily returns
        returns = data['close'].pct_change().dropna()
        return returns
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Validate data quality for a symbol.
        
        Args:
            data: Historical data
            symbol: Stock symbol
            
        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {
                'valid': False,
                'reason': 'No data available'
            }
        
        quality_metrics = {
            'valid': True,
            'total_rows': len(data),
            'date_range': (data.index.min(), data.index.max()),
            'missing_values': data.isnull().sum().to_dict(),
            'gaps': [],
            'anomalies': []
        }
        
        # Check for missing values
        missing_pct = data.isnull().sum() / len(data) * 100
        if missing_pct.max() > 10:  # More than 10% missing
            quality_metrics['valid'] = False
            quality_metrics['reason'] = f"Too many missing values: {missing_pct.max():.1f}%"
        
        # Check for large gaps in dates
        date_diffs = pd.Series(data.index).diff().dt.days
        large_gaps = date_diffs[date_diffs > 7]  # More than 7 days
        
        if len(large_gaps) > 0:
            quality_metrics['gaps'] = [
                {'start': data.index[i-1], 'end': data.index[i], 'days': gap}
                for i, gap in large_gaps.items()
            ]
        
        # Check for price anomalies
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            extreme_returns = daily_returns[abs(daily_returns) > 0.5]  # >50% daily return
            
            if len(extreme_returns) > 0:
                quality_metrics['anomalies'] = [
                    {'date': date, 'return': ret}
                    for date, ret in extreme_returns.items()
                ]
        
        # Check volume anomalies
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            volume_anomalies = data[data['volume'] > avg_volume * 10]  # 10x average volume
            
            if len(volume_anomalies) > 0:
                quality_metrics['volume_anomalies'] = len(volume_anomalies)
        
        return quality_metrics
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for backtesting.
        
        Args:
            data: Raw historical data
            
        Returns:
            Cleaned data
        """
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        
        # Remove rows with missing OHLC data
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in cleaned_data.columns]
        
        if available_ohlc:
            cleaned_data = cleaned_data.dropna(subset=available_ohlc)
        
        # Fill missing volume with 0
        if 'volume' in cleaned_data.columns:
            cleaned_data['volume'] = cleaned_data['volume'].fillna(0)
        
        # Ensure high >= low and high >= open, close
        if all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close']):
            # Fix obvious errors
            cleaned_data['high'] = cleaned_data[['open', 'high', 'low', 'close']].max(axis=1)
            cleaned_data['low'] = cleaned_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove extreme outliers (>3 standard deviations in daily returns)
        if 'close' in cleaned_data.columns and len(cleaned_data) > 30:
            returns = cleaned_data['close'].pct_change()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Identify outliers
            outliers = abs(returns - mean_return) > 3 * std_return
            
            if outliers.sum() > 0:
                logger.warning(f"Removing {outliers.sum()} outlier data points")
                cleaned_data = cleaned_data[~outliers]
        
        return cleaned_data
    
    def generate_synthetic_data(self, symbol: str, start_date: datetime, 
                               end_date: datetime, 
                               initial_price: float = 100.0,
                               volatility: float = 0.2,
                               drift: float = 0.05) -> pd.DataFrame:
        """Generate synthetic price data for testing.
        
        Args:
            symbol: Symbol name
            start_date: Start date
            end_date: End date
            initial_price: Starting price
            volatility: Annual volatility
            drift: Annual drift
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        num_days = len(dates)
        
        # Generate random walks
        np.random.seed(42)  # For reproducible data
        
        # Daily parameters
        dt = 1/252  # Daily time step
        daily_vol = volatility * np.sqrt(dt)
        daily_drift = drift * dt
        
        # Generate price path using geometric Brownian motion
        returns = np.random.normal(daily_drift, daily_vol, num_days)
        prices = [initial_price]
        
        for i in range(1, num_days):
            next_price = prices[-1] * np.exp(returns[i])
            prices.append(next_price)
        
        prices = np.array(prices)
        
        # Generate OHLC from close prices
        data = {
            'open': np.roll(prices, 1),  # Previous day's close as open
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, num_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, num_days))),
            'volume': np.random.randint(100000, 2000000, num_days)
        }
        
        # Fix first day's open
        data['open'][0] = initial_price
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        for i in range(num_days):
            max_price = max(data['open'][i], data['close'][i])
            min_price = min(data['open'][i], data['close'][i])
            
            data['high'][i] = max(data['high'][i], max_price)
            data['low'][i] = min(data['low'][i], min_price)
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._data_cache.clear()
        
        # Remove cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                logger.info(f"Removed cache file {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'cached_files': len(cache_files),
            'memory_cache_entries': len(self._data_cache),
            'total_disk_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }