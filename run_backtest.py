#!/usr/bin/env python3
"""
Main script to run backtests for trading strategies.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.loader import load_config
from backtest.engine import BacktestEngine, BacktestConfig
from backtest.analyzer import BacktestAnalyzer
from strategies.fractional_breakout import FractionalBreakoutStrategy


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backtest.log')
        ]
    )


def create_backtest_config(config_data: dict, start_date: str = None, 
                          end_date: str = None) -> BacktestConfig:
    """Create backtest configuration from trading config.
    
    Args:
        config_data: Trading configuration
        start_date: Override start date
        end_date: Override end date
        
    Returns:
        Backtest configuration
    """
    backtest_data = config_data.get('backtest', {})
    
    # Use provided dates or defaults from config
    start = start_date or backtest_data.get('start_date', '2023-01-01')
    end = end_date or backtest_data.get('end_date', '2024-12-31')
    
    config = BacktestConfig(
        start_date=start,
        end_date=end,
        initial_capital=backtest_data.get('initial_capital', 10000.0),
        commission_equity=backtest_data.get('commission_equity', 0.0),
        commission_options=backtest_data.get('commission_options', 0.65),
        slippage_equity_bps=backtest_data.get('slippage_equity_bps', 5),
        slippage_options_pct=backtest_data.get('slippage_options_pct', 1.0),
        benchmark=backtest_data.get('benchmark', 'SPY')
    )
    
    # Override with account settings
    account_config = config_data.get('account', {})
    if 'max_positions' in account_config:
        config.max_positions = account_config['max_positions']
    if 'per_trade_risk_pct' in account_config:
        config.risk_per_trade_pct = account_config['per_trade_risk_pct']
    if 'daily_drawdown_pct' in account_config:
        config.daily_loss_limit_pct = account_config['daily_drawdown_pct']
    
    return config


def create_strategies(config_data: dict) -> list:
    """Create strategy instances from configuration.
    
    Args:
        config_data: Trading configuration
        
    Returns:
        List of strategy instances
    """
    strategies = []
    strategy_configs = config_data.get('strategies', {})
    
    # Create fractional breakout strategy if enabled
    if 'fractional_breakout' in strategy_configs:
        fb_config = strategy_configs['fractional_breakout']
        if fb_config.get('enabled', False):
            try:
                # Create mock data client for backtesting
                class MockDataClient:
                    def get_asset_info(self, symbol):
                        return {'fractionable': True, 'tradable': True}
                    
                    def get_stock_quotes(self, symbols):
                        return {symbol: {'bid_price': 150.0, 'ask_price': 150.1} for symbol in symbols}
                
                strategy = FractionalBreakoutStrategy(fb_config, MockDataClient())
                strategies.append(strategy)
                print(f"Added Fractional Breakout strategy with universe: {fb_config.get('universe', [])}")
                
            except Exception as e:
                print(f"Failed to create Fractional Breakout strategy: {e}")
    
    return strategies


def run_backtest(config_file: str = None, start_date: str = None, 
                end_date: str = None, output_dir: str = None,
                strategies: list = None) -> None:
    """Run a complete backtest.
    
    Args:
        config_file: Path to configuration file
        start_date: Start date for backtest
        end_date: End date for backtest
        output_dir: Directory to save results
        strategies: List of specific strategies to test
    """
    print("Starting backtest...")
    
    # Load configuration
    config_file = config_file or 'config/config.yml'
    if not Path(config_file).exists():
        print(f"Configuration file not found: {config_file}")
        print("Using default configuration...")
        
        # Create default config
        config_data = {
            'account': {
                'starting_equity': 10000.0,
                'max_positions': 5,
                'per_trade_risk_pct': 1.0,
                'daily_drawdown_pct': 3.0
            },
            'strategies': {
                'fractional_breakout': {
                    'enabled': True,
                    'universe': ['AAPL', 'MSFT', 'GOOGL'],
                    'position_size_usd': 100,
                    'atr_window': 14,
                    'atr_stop_multiplier': 1.0,
                    'atr_target_multiplier': 2.0,
                    'breakout_lookback': 20,
                    'min_volume': 500000
                }
            },
            'backtest': {
                'start_date': '2023-01-01',
                'end_date': '2024-06-30',
                'initial_capital': 10000.0,
                'commission_equity': 0.0,
                'commission_options': 0.65,
                'slippage_equity_bps': 5,
                'slippage_options_pct': 1.0
            }
        }
    else:
        print(f"Loading configuration from {config_file}")
        config_data = load_config(config_file)
        if hasattr(config_data, '__dict__'):
            # Convert Pydantic model to dict
            config_data = config_data.__dict__
    
    # Create backtest configuration
    backtest_config = create_backtest_config(config_data, start_date, end_date)
    print(f"Backtesting period: {backtest_config.start_date} to {backtest_config.end_date}")
    print(f"Initial capital: ${backtest_config.initial_capital:,.2f}")
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config)
    
    # Create and add strategies
    strategy_instances = create_strategies(config_data)
    
    if not strategy_instances:
        print("No strategies found or enabled. Creating default strategy...")
        
        # Create default strategy
        class MockDataClient:
            def get_asset_info(self, symbol):
                return {'fractionable': True, 'tradable': True}
            
            def get_stock_quotes(self, symbols):
                return {symbol: {'bid_price': 150.0, 'ask_price': 150.1} for symbol in symbols}
        
        default_config = {
            'enabled': True,
            'universe': ['AAPL', 'MSFT'],
            'position_size_usd': 100,
            'atr_window': 14,
            'atr_stop_multiplier': 1.0,
            'atr_target_multiplier': 2.0,
            'breakout_lookback': 20,
            'min_volume': 500000
        }
        
        try:
            strategy = FractionalBreakoutStrategy(default_config, MockDataClient())
            strategy_instances.append(strategy)
            print("Added default Fractional Breakout strategy")
        except Exception as e:
            print(f"Failed to create default strategy: {e}")
            return
    
    # Add strategies to engine
    for strategy in strategy_instances:
        engine.add_strategy(strategy)
        print(f"Added strategy: {strategy.name}")
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        results = engine.run()
        print("Backtest completed successfully!")
        
        # Print quick results
        print(f"\nQuick Results:")
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Annualized Return: {results.annualized_return:.2f}%")
        print(f"Volatility: {results.volatility:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2f}%")
        
        # Analyze results
        print("\nGenerating detailed analysis...")
        analyzer = BacktestAnalyzer(results)
        
        # Set output directory
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"backtest_results_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save reports
        print(f"Saving results to {output_path}")
        analyzer.save_report(str(output_path / "backtest_report"), format='json')
        analyzer.save_report(str(output_path / "backtest_report"), format='txt')
        analyzer.save_report(str(output_path / "backtest_report"), format='html')
        
        # Create plots
        try:
            analyzer.create_plots(str(output_path / "plots"))
            print("Generated visualization plots")
        except Exception as e:
            print(f"Failed to create plots: {e}")
        
        print(f"\nBacktest complete! Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run trading strategy backtest')
    
    parser.add_argument('--config', '-c', type=str, default='config/config.yml',
                       help='Configuration file path')
    parser.add_argument('--start-date', '-s', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for results')
    parser.add_argument('--strategies', nargs='+',
                       help='Specific strategies to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run backtest
    run_backtest(
        config_file=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        strategies=args.strategies
    )


if __name__ == '__main__':
    main()