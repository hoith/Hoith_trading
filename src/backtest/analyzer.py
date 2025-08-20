"""
Advanced analysis and reporting for backtest results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

from backtest.engine import BacktestResults, BacktestTrade


class BacktestAnalyzer:
    """Advanced analysis of backtest results."""
    
    def __init__(self, results: BacktestResults):
        """Initialize analyzer with backtest results.
        
        Args:
            results: Backtest results to analyze
        """
        self.results = results
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Dictionary with detailed performance metrics
        """
        report = {
            'summary': self._generate_summary(),
            'returns': self._analyze_returns(),
            'risk': self._analyze_risk(),
            'trades': self._analyze_trades(),
            'drawdowns': self._analyze_drawdowns(),
            'monthly_returns': self._analyze_monthly_returns(),
            'strategy_breakdown': self._analyze_strategy_performance(),
            'benchmarks': self._compare_to_benchmarks()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_return_pct': round(self.results.total_return, 2),
            'annualized_return_pct': round(self.results.annualized_return, 2),
            'volatility_pct': round(self.results.volatility, 2),
            'sharpe_ratio': round(self.results.sharpe_ratio, 3),
            'max_drawdown_pct': round(self.results.max_drawdown, 2),
            'calmar_ratio': round(self.results.calmar_ratio, 3),
            'sortino_ratio': round(self.results.sortino_ratio, 3),
            'total_trades': self.results.total_trades,
            'win_rate_pct': round(self.results.win_rate, 2),
            'profit_factor': round(self.results.profit_factor, 3),
            'duration_days': self.results.duration_days,
            'start_date': self.results.start_date.strftime('%Y-%m-%d') if self.results.start_date else None,
            'end_date': self.results.end_date.strftime('%Y-%m-%d') if self.results.end_date else None
        }
    
    def _analyze_returns(self) -> Dict[str, Any]:
        """Analyze return characteristics."""
        if self.results.daily_returns.empty:
            return {}
        
        returns = self.results.daily_returns
        
        # Calculate various return metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        return {
            'mean_daily_return_pct': round(returns.mean() * 100, 4),
            'median_daily_return_pct': round(returns.median() * 100, 4),
            'std_daily_return_pct': round(returns.std() * 100, 4),
            'skewness': round(returns.skew(), 3),
            'kurtosis': round(returns.kurtosis(), 3),
            'positive_days': len(positive_returns),
            'negative_days': len(negative_returns),
            'flat_days': len(returns[returns == 0]),
            'best_day_pct': round(returns.max() * 100, 2),
            'worst_day_pct': round(returns.min() * 100, 2),
            'avg_positive_return_pct': round(positive_returns.mean() * 100, 4) if len(positive_returns) > 0 else 0,
            'avg_negative_return_pct': round(negative_returns.mean() * 100, 4) if len(negative_returns) > 0 else 0
        }
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk metrics."""
        if self.results.daily_returns.empty:
            return {}
        
        returns = self.results.daily_returns
        
        # Value at Risk calculations
        var_levels = [0.01, 0.05, 0.1]
        var_metrics = {}
        
        for level in var_levels:
            var_value = np.percentile(returns, level * 100)
            var_metrics[f'var_{int(level*100)}pct'] = round(var_value * 100, 3)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_value]
            if len(tail_returns) > 0:
                es_value = tail_returns.mean()
                var_metrics[f'es_{int(level*100)}pct'] = round(es_value * 100, 3)
        
        # Beta calculation (if benchmark available)
        beta = None
        if not self.results.benchmark_returns.empty and len(self.results.benchmark_returns) == len(returns):
            covariance = np.cov(returns, self.results.benchmark_returns)[0, 1]
            benchmark_variance = np.var(self.results.benchmark_returns)
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
        
        risk_metrics = {
            'var_es_metrics': var_metrics,
            'beta': round(beta, 3) if beta is not None else None,
            'tracking_error': None,  # To be implemented
            'information_ratio': None  # To be implemented
        }
        
        return risk_metrics
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze individual trades."""
        if not self.results.trades:
            return {}
        
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'strategy': t.strategy,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600 if t.exit_time and t.entry_time else 0,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'exit_reason': t.exit_reason
        } for t in self.results.trades])
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Trade duration analysis
        if not trades_df['duration_hours'].empty:
            avg_duration = trades_df['duration_hours'].mean()
            max_duration = trades_df['duration_hours'].max()
            min_duration = trades_df['duration_hours'].min()
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()] if not trades_df.empty else None
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()] if not trades_df.empty else None
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in trades_df.columns else {}
        
        return {
            'total_pnl': round(trades_df['pnl'].sum(), 2),
            'avg_trade_pnl': round(trades_df['pnl'].mean(), 2),
            'median_trade_pnl': round(trades_df['pnl'].median(), 2),
            'std_trade_pnl': round(trades_df['pnl'].std(), 2),
            'largest_win': round(winning_trades['pnl'].max(), 2) if not winning_trades.empty else 0,
            'largest_loss': round(losing_trades['pnl'].min(), 2) if not losing_trades.empty else 0,
            'avg_win': round(winning_trades['pnl'].mean(), 2) if not winning_trades.empty else 0,
            'avg_loss': round(losing_trades['pnl'].mean(), 2) if not losing_trades.empty else 0,
            'avg_duration_hours': round(avg_duration, 2),
            'max_duration_hours': round(max_duration, 2),
            'min_duration_hours': round(min_duration, 2),
            'best_trade': {
                'symbol': best_trade['symbol'],
                'pnl': round(best_trade['pnl'], 2),
                'pnl_pct': round(best_trade['pnl_pct'], 2)
            } if best_trade is not None else None,
            'worst_trade': {
                'symbol': worst_trade['symbol'],
                'pnl': round(worst_trade['pnl'], 2),
                'pnl_pct': round(worst_trade['pnl_pct'], 2)
            } if worst_trade is not None else None,
            'exit_reasons': exit_reasons,
            'symbols_traded': trades_df['symbol'].nunique() if not trades_df.empty else 0,
            'trades_per_symbol': trades_df['symbol'].value_counts().to_dict() if not trades_df.empty else {}
        }
    
    def _analyze_drawdowns(self) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        if self.results.equity_curve.empty:
            return {}
        
        equity = self.results.equity_curve['equity']
        
        # Calculate running maximum and drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        if in_drawdown.any():
            # Find start and end of each drawdown period
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
            
            start_dates = drawdown.index[drawdown_starts]
            end_dates = drawdown.index[drawdown_ends]
            
            # Handle case where backtest ends in drawdown
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.Index([drawdown.index[-1]]))
            
            for start, end in zip(start_dates, end_dates):
                period_drawdown = drawdown.loc[start:end]
                max_dd = period_drawdown.min()
                duration = (end - start).days
                
                drawdown_periods.append({
                    'start_date': start.strftime('%Y-%m-%d'),
                    'end_date': end.strftime('%Y-%m-%d'),
                    'duration_days': duration,
                    'max_drawdown_pct': round(max_dd, 2)
                })
        
        # Sort by severity
        drawdown_periods.sort(key=lambda x: x['max_drawdown_pct'])
        
        return {
            'max_drawdown_pct': round(drawdown.min(), 2),
            'avg_drawdown_pct': round(drawdown[drawdown < 0].mean(), 2) if (drawdown < 0).any() else 0,
            'drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration_days': round(np.mean([dp['duration_days'] for dp in drawdown_periods]), 1) if drawdown_periods else 0,
            'max_drawdown_duration_days': max([dp['duration_days'] for dp in drawdown_periods]) if drawdown_periods else 0,
            'top_5_drawdowns': drawdown_periods[:5],
            'recovery_factor': round(self.results.total_return / abs(self.results.max_drawdown), 2) if self.results.max_drawdown < 0 else None
        }
    
    def _analyze_monthly_returns(self) -> Dict[str, Any]:
        """Analyze monthly return patterns."""
        if self.results.daily_returns.empty:
            return {}
        
        returns = self.results.daily_returns
        
        # Resample to monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Monthly statistics
        monthly_stats = {
            'avg_monthly_return_pct': round(monthly_returns.mean() * 100, 2),
            'median_monthly_return_pct': round(monthly_returns.median() * 100, 2),
            'std_monthly_return_pct': round(monthly_returns.std() * 100, 2),
            'best_month_pct': round(monthly_returns.max() * 100, 2),
            'worst_month_pct': round(monthly_returns.min() * 100, 2),
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns < 0).sum(),
            'flat_months': (monthly_returns == 0).sum()
        }
        
        # Create monthly returns table
        monthly_table = {}
        for date, ret in monthly_returns.items():
            year = date.year
            month = date.strftime('%b')
            
            if year not in monthly_table:
                monthly_table[year] = {}
            
            monthly_table[year][month] = round(ret * 100, 2)
        
        return {
            'statistics': monthly_stats,
            'monthly_table': monthly_table
        }
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy."""
        return self.results.strategy_results
    
    def _compare_to_benchmarks(self) -> Dict[str, Any]:
        """Compare performance to benchmarks."""
        # Placeholder for benchmark comparison
        # Would need benchmark data to implement
        return {
            'benchmark_comparison': 'Not implemented - requires benchmark data'
        }
    
    def save_report(self, output_path: str, format: str = 'json') -> None:
        """Save performance report to file.
        
        Args:
            output_path: Path to save report
            format: Output format ('json', 'html', 'txt')
        """
        report = self.generate_performance_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file.with_suffix('.json'), 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format.lower() == 'txt':
            self._save_text_report(report, output_file.with_suffix('.txt'))
        
        elif format.lower() == 'html':
            self._save_html_report(report, output_file.with_suffix('.html'))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_text_report(self, report: Dict, output_file: Path) -> None:
        """Save report in text format."""
        with open(output_file, 'w') as f:
            f.write("BACKTEST PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = report['summary']
            for key, value in summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n")
            
            # Returns
            if 'returns' in report and report['returns']:
                f.write("RETURN ANALYSIS\n")
                f.write("-" * 20 + "\n")
                returns = report['returns']
                for key, value in returns.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            # Trades
            if 'trades' in report and report['trades']:
                f.write("TRADE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                trades = report['trades']
                for key, value in trades.items():
                    if not isinstance(value, dict):
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
    
    def _save_html_report(self, report: Dict, output_file: Path) -> None:
        """Save report in HTML format."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #333; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1 class="header">Backtest Performance Report</h1>
            
            <div class="section">
                <h2>Summary</h2>
                {self._dict_to_html_table(report['summary'])}
            </div>
            
            {f'<div class="section"><h2>Return Analysis</h2>{self._dict_to_html_table(report["returns"])}</div>' if 'returns' in report and report['returns'] else ''}
            
            {f'<div class="section"><h2>Trade Analysis</h2>{self._dict_to_html_table({k: v for k, v in report["trades"].items() if not isinstance(v, dict)})}</div>' if 'trades' in report and report['trades'] else ''}
            
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _dict_to_html_table(self, data: Dict) -> str:
        """Convert dictionary to HTML table."""
        html = "<table>"
        for key, value in data.items():
            key_formatted = key.replace('_', ' ').title()
            html += f"<tr><td><strong>{key_formatted}</strong></td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def create_plots(self, output_dir: str) -> None:
        """Create visualization plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Equity curve plot
        if not self.results.equity_curve.empty:
            self._plot_equity_curve(output_path / "equity_curve.png")
        
        # Drawdown plot
        if not self.results.equity_curve.empty:
            self._plot_drawdowns(output_path / "drawdowns.png")
        
        # Returns distribution
        if not self.results.daily_returns.empty:
            self._plot_returns_distribution(output_path / "returns_distribution.png")
        
        # Monthly returns heatmap
        if not self.results.daily_returns.empty:
            self._plot_monthly_heatmap(output_path / "monthly_returns.png")
    
    def _plot_equity_curve(self, output_file: Path) -> None:
        """Plot equity curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.equity_curve.index, self.results.equity_curve['equity'])
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdowns(self, output_file: Path) -> None:
        """Plot drawdown chart."""
        equity = self.results.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title('Portfolio Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_distribution(self, output_file: Path) -> None:
        """Plot returns distribution."""
        returns = self.results.daily_returns * 100
        
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.7, density=True, edgecolor='black')
        plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        plt.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.2f}%')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_heatmap(self, output_file: Path) -> None:
        """Plot monthly returns heatmap."""
        returns = self.results.daily_returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create pivot table for heatmap
        monthly_data = []
        for date, ret in monthly_returns.items():
            monthly_data.append({
                'Year': date.year,
                'Month': date.month,
                'Return': ret * 100
            })
        
        if monthly_data:
            df = pd.DataFrame(monthly_data)
            pivot = df.pivot(index='Month', columns='Year', values='Return')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Monthly Return (%)'})
            plt.title('Monthly Returns Heatmap')
            plt.ylabel('Month')
            plt.xlabel('Year')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()