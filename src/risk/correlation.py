import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CorrelationGroup:
    """Group of correlated symbols."""
    name: str
    symbols: Set[str]
    max_positions: int
    correlation_threshold: float = 0.7


class CorrelationManager:
    """Manage correlation-based position limits."""
    
    def __init__(self, max_total_positions: int = 10):
        self.max_total_positions = max_total_positions
        self.correlation_groups: Dict[str, CorrelationGroup] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[datetime] = None
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
        
        # Setup predefined correlation groups
        self._setup_predefined_groups()
        
        logger.info("Initialized CorrelationManager")
    
    def _setup_predefined_groups(self):
        """Setup predefined correlation groups based on market knowledge."""
        
        # Technology stocks
        self.add_correlation_group(CorrelationGroup(
            name="tech_mega_caps",
            symbols={"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"},
            max_positions=2,  # Max 2 tech positions
            correlation_threshold=0.6
        ))
        
        # Index ETFs
        self.add_correlation_group(CorrelationGroup(
            name="broad_index",
            symbols={"SPY", "QQQ", "IWM", "DIA"},
            max_positions=1,  # Only one index position
            correlation_threshold=0.8
        ))
        
        # Financial sector
        self.add_correlation_group(CorrelationGroup(
            name="financials",
            symbols={"JPM", "BAC", "WFC", "GS", "MS", "C"},
            max_positions=2,
            correlation_threshold=0.7
        ))
        
        # Energy sector
        self.add_correlation_group(CorrelationGroup(
            name="energy",
            symbols={"XOM", "CVX", "COP", "EOG", "SLB"},
            max_positions=1,
            correlation_threshold=0.8
        ))
    
    def add_correlation_group(self, group: CorrelationGroup):
        """Add a correlation group."""
        self.correlation_groups[group.name] = group
        logger.info(f"Added correlation group '{group.name}' with {len(group.symbols)} symbols")
    
    def remove_correlation_group(self, group_name: str):
        """Remove a correlation group."""
        if group_name in self.correlation_groups:
            del self.correlation_groups[group_name]
            logger.info(f"Removed correlation group '{group_name}'")
    
    def can_add_position(self, symbol: str, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a new position can be added considering correlation limits."""
        
        # Check total position limit
        if len(current_positions) >= self.max_total_positions:
            return False, f"Maximum total positions reached: {len(current_positions)}/{self.max_total_positions}"
        
        # Check correlation group limits
        for group_name, group in self.correlation_groups.items():
            if symbol in group.symbols:
                # Count current positions in this group
                current_group_positions = sum(
                    1 for pos_symbol in current_positions.keys() 
                    if pos_symbol in group.symbols
                )
                
                if current_group_positions >= group.max_positions:
                    return False, f"Maximum positions in {group_name} group reached: {current_group_positions}/{group.max_positions}"
        
        # Check dynamic correlation limits
        if self.correlation_matrix is not None:
            violation = self._check_dynamic_correlation(symbol, current_positions)
            if violation:
                return False, violation
        
        return True, "Position allowed"
    
    def _check_dynamic_correlation(self, new_symbol: str, 
                                 current_positions: Dict[str, Any]) -> Optional[str]:
        """Check dynamic correlation constraints."""
        
        if new_symbol not in self.correlation_matrix.index:
            return None  # No correlation data available
        
        high_correlation_threshold = 0.8
        max_high_correlation_positions = 2
        
        # Check correlation with existing positions
        high_correlation_count = 0
        highly_correlated_symbols = []
        
        for pos_symbol in current_positions.keys():
            if pos_symbol in self.correlation_matrix.columns:
                correlation = abs(self.correlation_matrix.loc[new_symbol, pos_symbol])
                
                if correlation > high_correlation_threshold:
                    high_correlation_count += 1
                    highly_correlated_symbols.append(pos_symbol)
        
        if high_correlation_count >= max_high_correlation_positions:
            return f"Too many highly correlated positions (>{high_correlation_threshold}): {highly_correlated_symbols}"
        
        return None
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.DataFrame], 
                                   lookback_days: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix from price data."""
        
        try:
            # Prepare return data
            returns_data = {}
            
            for symbol, df in price_data.items():
                if len(df) < lookback_days:
                    continue
                
                # Calculate returns
                returns = df['close'].pct_change().dropna()
                if len(returns) >= lookback_days:
                    returns_data[symbol] = returns.tail(lookback_days)
            
            if len(returns_data) < 2:
                logger.warning("Insufficient data for correlation calculation")
                return pd.DataFrame()
            
            # Align dates and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:  # Minimum 20 days of data
                logger.warning("Insufficient aligned data for correlation")
                return pd.DataFrame()
            
            correlation_matrix = returns_df.corr()
            
            # Cache the result
            self.correlation_matrix = correlation_matrix
            self.last_correlation_update = datetime.now()
            
            logger.info(f"Updated correlation matrix with {len(correlation_matrix)} symbols")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get correlation between two symbols."""
        
        # Check cache first
        cache_key = f"{min(symbol1, symbol2)}_{max(symbol1, symbol2)}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        # Check correlation matrix
        if (self.correlation_matrix is not None and 
            symbol1 in self.correlation_matrix.index and 
            symbol2 in self.correlation_matrix.columns):
            
            correlation = self.correlation_matrix.loc[symbol1, symbol2]
            self.correlation_cache[cache_key] = correlation
            return correlation
        
        return None
    
    def get_highly_correlated_symbols(self, symbol: str, 
                                    threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Get symbols highly correlated with the given symbol."""
        
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.index:
            return []
        
        correlations = self.correlation_matrix[symbol].abs().sort_values(ascending=False)
        
        # Exclude self-correlation and filter by threshold
        highly_correlated = [
            (corr_symbol, corr_value) 
            for corr_symbol, corr_value in correlations.items()
            if corr_symbol != symbol and corr_value >= threshold
        ]
        
        return highly_correlated
    
    def analyze_portfolio_correlation(self, current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation structure of current portfolio."""
        
        if not current_positions or self.correlation_matrix is None:
            return {}
        
        position_symbols = list(current_positions.keys())
        valid_symbols = [s for s in position_symbols if s in self.correlation_matrix.index]
        
        if len(valid_symbols) < 2:
            return {"error": "Insufficient symbols for correlation analysis"}
        
        # Get correlation submatrix for portfolio
        portfolio_corr = self.correlation_matrix.loc[valid_symbols, valid_symbols]
        
        # Calculate metrics
        avg_correlation = portfolio_corr.values[np.triu_indices_from(portfolio_corr.values, k=1)].mean()
        max_correlation = portfolio_corr.values[np.triu_indices_from(portfolio_corr.values, k=1)].max()
        min_correlation = portfolio_corr.values[np.triu_indices_from(portfolio_corr.values, k=1)].min()
        
        # Find highest correlated pair
        upper_tri = np.triu(portfolio_corr.values, k=1)
        max_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
        highest_pair = (portfolio_corr.index[max_idx[0]], portfolio_corr.columns[max_idx[1]])
        
        # Count high correlation pairs
        high_corr_threshold = 0.7
        high_corr_count = np.sum(upper_tri > high_corr_threshold)
        
        # Calculate diversification ratio (1 = perfectly diversified, 0 = perfectly correlated)
        n_assets = len(valid_symbols)
        diversification_ratio = (1 - avg_correlation) if n_assets > 1 else 0
        
        return {
            'portfolio_symbols': valid_symbols,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'highest_corr_pair': highest_pair,
            'highest_corr_value': portfolio_corr.loc[highest_pair[0], highest_pair[1]],
            'high_corr_pairs_count': high_corr_count,
            'diversification_ratio': diversification_ratio,
            'correlation_risk': 'HIGH' if avg_correlation > 0.7 else 'MEDIUM' if avg_correlation > 0.4 else 'LOW'
        }
    
    def suggest_position_rebalancing(self, current_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest position changes to improve diversification."""
        
        suggestions = []
        
        # Get portfolio correlation analysis
        portfolio_analysis = self.analyze_portfolio_correlation(current_positions)
        
        if 'error' in portfolio_analysis:
            return suggestions
        
        # Check for overly correlated positions
        if portfolio_analysis['avg_correlation'] > 0.8:
            suggestions.append({
                'type': 'reduce_correlation',
                'description': f"Portfolio average correlation is high ({portfolio_analysis['avg_correlation']:.2f})",
                'action': 'Consider reducing positions in highly correlated assets',
                'priority': 'HIGH'
            })
        
        # Check correlation groups
        for group_name, group in self.correlation_groups.items():
            group_positions = [s for s in current_positions.keys() if s in group.symbols]
            
            if len(group_positions) > group.max_positions:
                suggestions.append({
                    'type': 'group_limit_exceeded',
                    'description': f"Too many positions in {group_name} group",
                    'current_positions': group_positions,
                    'max_allowed': group.max_positions,
                    'action': f"Reduce {group_name} positions",
                    'priority': 'HIGH'
                })
        
        # Suggest diversification opportunities
        if len(current_positions) < self.max_total_positions and portfolio_analysis['diversification_ratio'] < 0.6:
            suggestions.append({
                'type': 'improve_diversification',
                'description': f"Diversification ratio is low ({portfolio_analysis['diversification_ratio']:.2f})",
                'action': 'Consider adding positions in uncorrelated sectors',
                'priority': 'MEDIUM'
            })
        
        return suggestions
    
    def get_uncorrelated_candidates(self, current_positions: Dict[str, Any], 
                                   candidate_symbols: List[str],
                                   max_correlation: float = 0.5) -> List[str]:
        """Get candidate symbols that are not highly correlated with current positions."""
        
        if not current_positions or self.correlation_matrix is None:
            return candidate_symbols
        
        uncorrelated_candidates = []
        
        for candidate in candidate_symbols:
            if candidate not in self.correlation_matrix.index:
                uncorrelated_candidates.append(candidate)  # No correlation data, assume uncorrelated
                continue
            
            is_uncorrelated = True
            
            for pos_symbol in current_positions.keys():
                if pos_symbol in self.correlation_matrix.columns:
                    correlation = abs(self.correlation_matrix.loc[candidate, pos_symbol])
                    
                    if correlation > max_correlation:
                        is_uncorrelated = False
                        break
            
            if is_uncorrelated:
                uncorrelated_candidates.append(candidate)
        
        return uncorrelated_candidates
    
    def update_correlation_cache(self, symbol1: str, symbol2: str, correlation: float):
        """Manually update correlation cache."""
        cache_key = f"{min(symbol1, symbol2)}_{max(symbol1, symbol2)}"
        self.correlation_cache[cache_key] = correlation
    
    def clear_correlation_cache(self):
        """Clear correlation cache."""
        self.correlation_cache.clear()
        logger.info("Cleared correlation cache")
    
    def get_correlation_status(self) -> Dict[str, Any]:
        """Get current correlation management status."""
        
        status = {
            'total_groups': len(self.correlation_groups),
            'max_total_positions': self.max_total_positions,
            'correlation_matrix_size': len(self.correlation_matrix) if self.correlation_matrix is not None else 0,
            'last_update': self.last_correlation_update.isoformat() if self.last_correlation_update else None,
            'cache_size': len(self.correlation_cache),
            'groups': {}
        }
        
        for group_name, group in self.correlation_groups.items():
            status['groups'][group_name] = {
                'symbols': list(group.symbols),
                'max_positions': group.max_positions,
                'correlation_threshold': group.correlation_threshold
            }
        
        return status
    
    def should_update_correlations(self, hours_threshold: int = 24) -> bool:
        """Check if correlation matrix should be updated."""
        
        if self.last_correlation_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_correlation_update
        return time_since_update > timedelta(hours=hours_threshold)
    
    def export_correlation_matrix(self, file_path: str):
        """Export correlation matrix to CSV file."""
        
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(file_path)
            logger.info(f"Exported correlation matrix to {file_path}")
        else:
            logger.warning("No correlation matrix to export")
    
    def import_correlation_matrix(self, file_path: str):
        """Import correlation matrix from CSV file."""
        
        try:
            self.correlation_matrix = pd.read_csv(file_path, index_col=0)
            self.last_correlation_update = datetime.now()
            logger.info(f"Imported correlation matrix from {file_path}")
        except Exception as e:
            logger.error(f"Error importing correlation matrix: {e}")