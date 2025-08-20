#!/usr/bin/env python3

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional
from datetime import datetime, time
from pathlib import Path
import argparse
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import load_config, get_alpaca_config
from data.alpaca_client import AlpacaDataClient
from data.historical import HistoricalDataFetcher
from execution.router import OrderRouter
from strategies.iron_condor import IronCondorStrategy
from strategies.momentum_vertical import MomentumVerticalStrategy
from strategies.fractional_breakout import FractionalBreakoutStrategy
from risk.sizing import PositionSizer
from risk.breakers import RiskBreaker
from risk.correlation import CorrelationManager
from state.database import DatabaseManager
from metrics.performance import PerformanceAnalyzer
from metrics.monitoring import SystemMonitor, NotificationManager
from utils.logging import setup_logging, TradingLogger
from utils.timezone import get_market_timezone, is_market_hours, get_current_market_time

logger = logging.getLogger(__name__)


class TradingSystemRunner:
    """Main orchestrator for the automated trading system."""
    
    def __init__(self, config_path: str = "config.yml", session_id: str = None):
        self.config_path = config_path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_running = False
        self.shutdown_requested = False
        
        # Initialize components
        self.config = None
        self.data_client = None
        self.historical_fetcher = None
        self.order_router = None
        self.strategies = {}
        self.position_sizer = None
        self.risk_breaker = None
        self.correlation_manager = None
        self.db_manager = None
        self.performance_analyzer = None
        self.system_monitor = None
        self.notification_manager = None
        self.trading_logger = None
        
        # Runtime state
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self):
        """Initialize all system components."""
        logger.info(f"Initializing trading system (session: {self.session_id})")
        
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            alpaca_config = get_alpaca_config()
            
            logger.info(f"Paper trading: {alpaca_config['paper']}")
            logger.info(f"Dry run: {alpaca_config['dry_run']}")
            
            # Initialize data clients
            self.data_client = AlpacaDataClient()
            self.historical_fetcher = HistoricalDataFetcher(self.data_client)
            
            # Get account info
            account = self.data_client.get_account()
            account_equity = float(account['equity'])
            logger.info(f"Account equity: ${account_equity:,.2f}")
            
            # Initialize database
            self.db_manager = DatabaseManager()
            
            # Initialize execution
            self.order_router = OrderRouter(self.data_client)
            
            # Initialize risk management
            self.position_sizer = PositionSizer(
                account_equity=account_equity,
                max_positions=self.config.account.max_positions,
                risk_per_trade_pct=self.config.account.per_trade_risk_pct
            )
            
            self.risk_breaker = RiskBreaker(account_equity)
            self.correlation_manager = CorrelationManager(
                max_total_positions=self.config.account.max_positions
            )
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Initialize monitoring
            self.notification_manager = NotificationManager(
                webhook_url=self.config.notifications.webhook_url,
                enabled=self.config.notifications.enabled
            )
            
            self.system_monitor = SystemMonitor(self.notification_manager)
            self.performance_analyzer = PerformanceAnalyzer(self.db_manager)
            self.trading_logger = TradingLogger("trading.runner")
            
            # Start strategy run tracking
            self._start_strategy_runs()
            
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize enabled trading strategies."""
        self.strategies = {}
        
        # Iron Condor Strategy
        if self.config.strategies.iron_condor.enabled:
            self.strategies['iron_condor'] = IronCondorStrategy(
                self.config.strategies.iron_condor.dict(),
                self.data_client
            )
            logger.info("Initialized Iron Condor strategy")
        
        # Momentum Vertical Strategy
        if self.config.strategies.momentum_vertical.enabled:
            self.strategies['momentum_vertical'] = MomentumVerticalStrategy(
                self.config.strategies.momentum_vertical.dict(),
                self.data_client
            )
            logger.info("Initialized Momentum Vertical strategy")
        
        # Fractional Breakout Strategy
        if self.config.strategies.fractional_breakout.enabled:
            self.strategies['fractional_breakout'] = FractionalBreakoutStrategy(
                self.config.strategies.fractional_breakout.dict(),
                self.data_client
            )
            logger.info("Initialized Fractional Breakout strategy")
        
        if not self.strategies:
            logger.warning("No strategies enabled!")
    
    def _start_strategy_runs(self):
        """Start database tracking for strategy runs."""
        account = self.data_client.get_account()
        starting_equity = float(account['equity'])
        
        for strategy_name in self.strategies.keys():
            self.db_manager.start_strategy_run(
                strategy_name=strategy_name,
                session_id=self.session_id,
                starting_equity=starting_equity,
                config=self.config.dict()
            )
    
    async def run_once(self) -> Dict[str, any]:
        """Run one iteration of the trading loop."""
        logger.info("Starting trading loop iteration")
        
        try:
            # Check system health
            health_status = self.system_monitor.get_system_status()
            if health_status['overall_status'] != 'healthy':
                logger.warning(f"System health degraded: {health_status['health_score']:.1f}%")
            
            # Update account info and risk management
            account = self.data_client.get_account()
            current_equity = float(account['equity'])
            self.position_sizer.update_equity(current_equity)
            self.risk_breaker.update_equity(current_equity)
            
            # Check risk breakers
            active_positions = self.data_client.get_positions()
            risk_status = self.risk_breaker.check_all_breakers(current_equity, active_positions)
            
            # Handle risk breaker actions
            if self.risk_breaker.should_close_all_positions():
                logger.critical("Risk breaker triggered - closing all positions")
                await self._close_all_positions()
                return {'status': 'risk_breaker_triggered', 'action': 'closed_all_positions'}
            
            if not self.risk_breaker.can_enter_new_position():
                logger.warning("Risk breaker active - no new entries allowed")
                return {'status': 'risk_breaker_active', 'action': 'monitoring_only'}
            
            # Get market data for all strategy universes
            all_symbols = set()
            for strategy in self.strategies.values():
                if hasattr(strategy, 'universe'):
                    all_symbols.update(strategy.universe)
            
            market_data = {}
            if all_symbols:
                market_data = self._fetch_market_data(list(all_symbols))
            
            # Generate signals from all strategies
            all_signals = []
            for strategy_name, strategy in self.strategies.items():
                if not strategy.enabled:
                    continue
                
                try:
                    signals = strategy.generate_signals(market_data, self.current_positions)
                    for signal in signals:
                        signal.metadata['strategy'] = strategy_name
                    all_signals.extend(signals)
                    
                    logger.debug(f"{strategy_name} generated {len(signals)} signals")
                    
                except Exception as e:
                    logger.error(f"Error generating signals for {strategy_name}: {e}")
            
            # Filter and prioritize signals
            valid_signals = self._filter_signals(all_signals)
            logger.info(f"Generated {len(all_signals)} signals, {len(valid_signals)} valid")
            
            # Execute valid signals
            execution_results = []
            for signal in valid_signals:
                try:
                    result = await self._execute_signal(signal)
                    execution_results.append(result)
                    
                    if result.get('success'):
                        self.trade_count += 1
                        # Record trade in database
                        self.db_manager.record_trade(result)
                        
                        # Send notification
                        if self.config.notifications.enabled:
                            self.notification_manager.send_trade_notification(result)
                    
                except Exception as e:
                    logger.error(f"Error executing signal for {signal.symbol}: {e}")
            
            # Check exit conditions for existing positions
            exit_signals = self._check_exit_conditions(market_data)
            for exit_signal in exit_signals:
                try:
                    result = await self._execute_signal(exit_signal)
                    execution_results.append(result)
                    
                    if result.get('success'):
                        # Update position status
                        strategy_name = exit_signal.metadata.get('strategy')
                        self.db_manager.close_position(strategy_name, exit_signal.symbol)
                    
                except Exception as e:
                    logger.error(f"Error executing exit signal for {exit_signal.symbol}: {e}")
            
            # Update performance metrics
            self._update_performance_metrics()
            
            return {
                'status': 'completed',
                'signals_generated': len(all_signals),
                'signals_executed': len([r for r in execution_results if r.get('success')]),
                'current_equity': current_equity,
                'active_positions': len(active_positions),
                'trade_count': self.trade_count
            }
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _fetch_market_data(self, symbols: List[str]) -> Dict[str, any]:
        """Fetch market data for symbols."""
        try:
            # Get historical data (last 60 days for technical analysis)
            market_data = {}
            
            for symbol in symbols:
                try:
                    df = self.historical_fetcher.get_stock_data([symbol], days=60)
                    if not df.empty:
                        market_data[symbol] = df.loc[symbol] if symbol in df.index.get_level_values(0) else df
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {e}")
            
            logger.debug(f"Fetched market data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _filter_signals(self, signals: List) -> List:
        """Filter and validate signals before execution."""
        valid_signals = []
        
        for signal in signals:
            try:
                # Get strategy instance
                strategy_name = signal.metadata.get('strategy')
                strategy = self.strategies.get(strategy_name)
                
                if not strategy:
                    logger.warning(f"Unknown strategy in signal: {strategy_name}")
                    continue
                
                # Strategy-specific validation
                if not strategy.validate_signal(signal, {}):
                    logger.debug(f"Signal validation failed for {signal.symbol}")
                    continue
                
                # Check correlation limits
                correlated_symbols = self._get_correlated_symbols(signal.symbol)
                if not self.correlation_manager.can_add_position(signal.symbol, self.current_positions)[0]:
                    logger.info(f"Correlation limit exceeded for {signal.symbol}")
                    continue
                
                # Position sizing
                position_size_result = self.position_sizer.calculate_position_size(
                    signal,
                    current_positions=len(self.current_positions)
                )
                
                if not position_size_result.get('valid', False):
                    logger.debug(f"Position sizing failed for {signal.symbol}")
                    continue
                
                # Update signal with calculated position size
                signal.quantity = position_size_result['quantity']
                
                valid_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error filtering signal for {signal.symbol}: {e}")
        
        return valid_signals
    
    def _get_correlated_symbols(self, symbol: str) -> List[str]:
        """Get symbols correlated with the given symbol."""
        # Simplified correlation check
        index_etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
        
        if symbol in index_etfs:
            return index_etfs
        elif symbol in tech_stocks:
            return tech_stocks
        else:
            return [symbol]
    
    async def _execute_signal(self, signal) -> Dict[str, any]:
        """Execute a trading signal."""
        try:
            # Check final risk controls
            size_reduction = self.risk_breaker.get_size_reduction_factor()
            if size_reduction < 1.0:
                signal.quantity *= size_reduction
                logger.info(f"Applied risk reduction factor {size_reduction} to {signal.symbol}")
            
            # Execute through order router
            result = self.order_router.execute_signal(signal)
            
            if result.get('success'):
                self.trading_logger.log_entry(
                    strategy=signal.metadata.get('strategy', 'unknown'),
                    symbol=signal.symbol,
                    side=signal.action,
                    quantity=signal.quantity,
                    price=result.get('price', 0),
                    order_type=result.get('order_type', 'market')
                )
            else:
                self.trading_logger.log_rejection(
                    strategy=signal.metadata.get('strategy', 'unknown'),
                    symbol=signal.symbol,
                    reason=result.get('error', 'unknown')
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return {'success': False, 'error': str(e), 'symbol': signal.symbol}
    
    def _check_exit_conditions(self, market_data: Dict) -> List:
        """Check exit conditions for existing positions."""
        exit_signals = []
        
        try:
            # Get current positions
            positions = self.data_client.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                
                # Update position prices if market data available
                if symbol in market_data:
                    current_price = market_data[symbol]['close'].iloc[-1]
                    # Here you would update the position object with current price
                
                # Check each strategy for exit signals
                for strategy_name, strategy in self.strategies.items():
                    if strategy.has_position(symbol):
                        strategy_position = strategy.get_position(symbol)
                        
                        exit_signal = strategy.should_exit_position(strategy_position, market_data)
                        if exit_signal:
                            exit_signals.append(exit_signal)
                            logger.info(f"Exit signal generated for {symbol}: {exit_signal.metadata.get('exit_reason')}")
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
        
        return exit_signals
    
    async def _close_all_positions(self):
        """Close all open positions."""
        try:
            positions = self.data_client.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                quantity = abs(float(position['qty']))
                
                # Create market order to close position
                from strategies.base import StrategySignal, OrderType
                close_signal = StrategySignal(
                    symbol=symbol,
                    signal_type='exit',
                    action='sell' if position['side'] == 'long' else 'buy',
                    confidence=100.0,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    metadata={'exit_reason': 'risk_breaker', 'emergency_close': True}
                )
                
                result = await self._execute_signal(close_signal)
                
                if result.get('success'):
                    logger.info(f"Emergency closed position: {symbol}")
                else:
                    logger.error(f"Failed to close position {symbol}: {result.get('error')}")
                    
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _update_performance_metrics(self):
        """Update daily performance metrics."""
        try:
            account = self.data_client.get_account()
            current_equity = float(account['equity'])
            
            # Calculate daily P&L
            daily_pnl = current_equity - self.risk_breaker.session_start_equity
            
            # Update risk breaker
            self.risk_breaker.daily_pnl = daily_pnl
            
            # Record performance metrics for each strategy
            for strategy_name in self.strategies.keys():
                metric_data = {
                    'date': datetime.now().date(),
                    'strategy_name': strategy_name,
                    'daily_pnl': daily_pnl / len(self.strategies),  # Split across strategies
                    'daily_trades': self.trade_count,
                    'account_equity': current_equity,
                    'active_positions': len(self.current_positions)
                }
                
                self.db_manager.record_performance_metric(metric_data)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown_requested = True
    
    def _should_run_now(self) -> bool:
        """Check if trading should run based on schedule and market hours."""
        if self.shutdown_requested:
            return False
        
        # Check if market is open
        if not self.data_client.is_market_open():
            logger.debug("Market is closed")
            return False
        
        # Check schedule
        current_time = get_current_market_time().time()
        
        # Parse schedule from config
        entry_time = time.fromisoformat(self.config.scheduling.entry_time)
        monitor_end = time.fromisoformat(self.config.scheduling.monitor_end)
        
        # Only run during market hours within our schedule
        return entry_time <= current_time <= monitor_end
    
    async def run_continuous(self, interval_minutes: int = 15):
        """Run the trading system continuously."""
        logger.info(f"Starting continuous trading (interval: {interval_minutes} minutes)")
        self.is_running = True
        
        try:
            while not self.shutdown_requested:
                if self._should_run_now():
                    result = await self.run_once()
                    logger.debug(f"Trading loop result: {result}")
                else:
                    logger.debug("Outside trading hours, waiting...")
                
                # Wait for next iteration
                await asyncio.sleep(interval_minutes * 60)
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error in continuous trading loop: {e}")
            raise
        finally:
            self.is_running = False
    
    def run_daily_summary(self):
        """Generate and send daily summary."""
        try:
            # Get trading summary
            summary = self.db_manager.get_trading_summary(days=1)
            
            # Get system status
            system_status = self.system_monitor.get_system_status()
            
            # Combine into daily summary
            daily_summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'session_id': self.session_id,
                'trading_summary': summary,
                'system_health': system_status,
                'enabled_strategies': list(self.strategies.keys())
            }
            
            # Send notification
            if self.notification_manager:
                self.notification_manager.send_system_status(daily_summary)
            
            logger.info("Daily summary generated and sent")
            return daily_summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            return {}


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Trading System")
    parser.add_argument('--config', default='config.yml', help='Configuration file path')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--paper', action='store_true', help='Force paper trading mode')
    parser.add_argument('--interval', type=int, default=15, help='Run interval in minutes')
    parser.add_argument('--summary', action='store_true', help='Generate daily summary and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "INFO"
    setup_logging(log_level, "trading_system.log")
    
    logger.info("="*80)
    logger.info("AUTOMATED TRADING SYSTEM STARTING")
    logger.info("="*80)
    
    # Force paper trading if requested
    if args.paper:
        import os
        os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
        logger.info("Forced paper trading mode")
    
    try:
        # Initialize system
        runner = TradingSystemRunner(config_path=args.config)
        runner.initialize()
        
        if args.summary:
            # Generate daily summary and exit
            summary = runner.run_daily_summary()
            print(json.dumps(summary, indent=2, default=str))
            return
        
        if args.once:
            # Run once and exit
            logger.info("Running single iteration")
            result = await runner.run_once()
            logger.info(f"Single run completed: {result}")
            print(json.dumps(result, indent=2, default=str))
        else:
            # Run continuously
            await runner.run_continuous(interval_minutes=args.interval)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Trading system shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())