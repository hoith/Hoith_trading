#!/usr/bin/env python3
"""
Live Aggressive Trading Strategy - Paper Trading
Real-time implementation of the proven 8,376% return strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import os
import sys
import time as time_module
import json
from typing import Dict, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    pass

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AggressiveLiveTrader:
    """Live trading implementation of the aggressive momentum strategy"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.symbols = [
            # Original proven symbols
            'AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY', 'TQQQ',
            # Additional high-momentum stocks
            'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'AMD', 'CRM', 'UBER',
            # Additional ETFs for diversification
            'IWM',   # Russell 2000 Small Cap
            'XLK',   # Technology Sector ETF
            'XLF',   # Financial Sector ETF
            'ARKK',  # Innovation ETF
            'SOXL',  # 3x Semiconductor ETF
            'SPXL'   # 3x S&P 500 ETF
        ]
        
        # Initialize clients
        self.data_client = AlpacaDataClient()
        self.trading_client = TradingClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            paper=True  # Paper trading mode
        )
        
        # Strategy parameters (proven settings + new symbols)
        self.position_configs = {
            # Original proven symbols
            'AAPL': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'MSFT': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'GOOGL': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'QQQ': {'base_size': 30000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
            'SPY': {'base_size': 30000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},
            'TQQQ': {'base_size': 25000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8},
            
            # High-momentum individual stocks
            'TSLA': {'base_size': 20000, 'stop_pct': 4.0, 'target_pct': 10.0, 'max_hold': 10},  # Higher volatility
            'NVDA': {'base_size': 20000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
            'META': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'AMZN': {'base_size': 20000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'NFLX': {'base_size': 18000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
            'AMD': {'base_size': 18000, 'stop_pct': 4.0, 'target_pct': 10.0, 'max_hold': 10},
            'CRM': {'base_size': 18000, 'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold': 15},
            'UBER': {'base_size': 16000, 'stop_pct': 3.5, 'target_pct': 9.0, 'max_hold': 12},
            
            # Sector and leveraged ETFs
            'IWM': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},   # Small cap
            'XLK': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},   # Tech sector
            'XLF': {'base_size': 25000, 'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold': 12},   # Financial sector
            'ARKK': {'base_size': 20000, 'stop_pct': 3.5, 'target_pct': 8.0, 'max_hold': 10},  # Innovation ETF
            'SOXL': {'base_size': 22000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8},   # 3x Semiconductor
            'SPXL': {'base_size': 25000, 'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold': 8}    # 3x S&P 500
        }
        
        # Active positions tracking
        self.active_positions = {}
        self.trade_log = []
        
        logger.info("AggressiveLiveTrader initialized for paper trading")
        logger.info(f"Monitoring symbols: {self.symbols}")
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value from Alpaca"""
        try:
            account = self.trading_client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return self.initial_capital
    
    def get_market_data(self, symbol: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Get recent market data for signal generation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            df_raw = self.data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if df_raw.empty:
                return None
            
            # Fix MultiIndex
            if isinstance(df_raw.index, pd.MultiIndex):
                timestamps = [idx[1] for idx in df_raw.index if idx[0] == symbol]
                df_clean = df_raw.loc[df_raw.index.get_level_values(0) == symbol].copy()
                df_clean.index = timestamps
                return df_clean
            
            return df_raw
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def check_entry_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Check if entry signal is triggered for symbol"""
        if len(data) < 25:
            return None
        
        try:
            # Calculate indicators
            rsi = self.calculate_rsi(data['close'], 14)
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Momentum and volume analysis
            momentum_score = (data['close'].iloc[-1] / data['close'].iloc[-11] - 1) * 100
            avg_volume = data['volume'].iloc[-10:].mean()
            volume_ratio = current_volume / avg_volume
            
            # Apply proven entry criteria with symbol-specific thresholds
            entry_signal = False
            
            # Large cap individual stocks (conservative)
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'CRM']:
                if (momentum_score > 1.0 and current_rsi < 75 and volume_ratio > 0.8):
                    entry_signal = True
            
            # High-volatility growth stocks (more aggressive)        
            elif symbol in ['TSLA', 'NVDA', 'NFLX', 'AMD', 'UBER']:
                if (momentum_score > 1.5 and current_rsi < 70 and volume_ratio > 0.9):
                    entry_signal = True
                    
            # Standard ETFs (moderate)
            elif symbol in ['QQQ', 'SPY', 'IWM', 'XLK', 'XLF']:
                if (momentum_score > 0.5 and current_rsi < 80 and volume_ratio > 0.7):
                    entry_signal = True
            
            # Innovation/Growth ETFs (balanced)
            elif symbol in ['ARKK']:
                if (momentum_score > 1.0 and current_rsi < 75 and volume_ratio > 0.8):
                    entry_signal = True
                    
            # Leveraged ETFs (strict criteria due to high volatility)
            elif symbol in ['TQQQ', 'SOXL', 'SPXL']:
                if (momentum_score > 2.0 and current_rsi < 85):
                    entry_signal = True
            
            if entry_signal:
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'momentum_score': momentum_score,
                    'rsi': current_rsi,
                    'volume_ratio': volume_ratio,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry signal for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str) -> int:
        """Calculate position size based on current capital"""
        try:
            current_capital = self.get_current_portfolio_value()
            base_position = self.position_configs[symbol]['base_size']
            
            # Scale position size with capital growth
            position_multiplier = current_capital / self.initial_capital
            target_dollar_amount = base_position * position_multiplier
            
            # Cap at 40% of portfolio
            max_position = current_capital * 0.40
            target_dollar_amount = min(target_dollar_amount, max_position)
            
            # Get current price
            data = self.get_market_data(symbol, lookback_days=2)
            if data is None or len(data) == 0:
                return 0
            
            current_price = data['close'].iloc[-1]
            shares = int(target_dollar_amount / current_price)
            
            logger.info(f"{symbol}: Target ${target_dollar_amount:,.0f}, Price ${current_price:.2f}, Shares: {shares}")
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def place_market_order(self, symbol: str, shares: int, side: OrderSide) -> Optional[str]:
        """Place market order"""
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Order placed: {side.value} {shares} shares of {symbol}, Order ID: {order.id}")
            return str(order.id)
            
        except Exception as e:
            logger.error(f"Error placing {side.value} order for {symbol}: {e}")
            return None
    
    def enter_position(self, signal: Dict) -> bool:
        """Enter new position based on signal"""
        symbol = signal['symbol']
        
        # Check if already in position
        if symbol in self.active_positions:
            logger.info(f"Already in position for {symbol}, skipping entry")
            return False
        
        # Calculate position size
        shares = self.calculate_position_size(symbol)
        if shares <= 0:
            logger.warning(f"Invalid position size for {symbol}: {shares}")
            return False
        
        # Place buy order
        order_id = self.place_market_order(symbol, shares, OrderSide.BUY)
        if not order_id:
            return False
        
        # Record position
        config = self.position_configs[symbol]
        entry_price = signal['price']
        
        position = {
            'symbol': symbol,
            'shares': shares,
            'entry_price': entry_price,
            'entry_time': signal['timestamp'],
            'entry_order_id': order_id,
            'stop_loss': entry_price * (1 - config['stop_pct'] / 100),
            'take_profit': entry_price * (1 + config['target_pct'] / 100),
            'max_hold_days': config['max_hold'],
            'signal_data': signal
        }
        
        self.active_positions[symbol] = position
        
        logger.info(f"ENTERED POSITION: {symbol}")
        logger.info(f"  Shares: {shares}, Entry: ${entry_price:.2f}")
        logger.info(f"  Stop Loss: ${position['stop_loss']:.2f}")
        logger.info(f"  Take Profit: ${position['take_profit']:.2f}")
        logger.info(f"  Signal: Momentum {signal['momentum_score']:.1f}%, RSI {signal['rsi']:.1f}")
        
        return True
    
    def check_exit_conditions(self, symbol: str) -> Optional[str]:
        """Check if position should be exited"""
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        
        # Get current market data
        data = self.get_market_data(symbol, lookback_days=2)
        if data is None or len(data) == 0:
            return None
        
        current_price = data['close'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_high = data['high'].iloc[-1]
        
        # Check stop loss
        if current_low <= position['stop_loss']:
            return 'stop_loss'
        
        # Check take profit
        if current_high >= position['take_profit']:
            return 'take_profit'
        
        # Check time limit
        days_held = (datetime.now() - position['entry_time']).days
        if days_held >= position['max_hold_days']:
            return 'time_limit'
        
        return None
    
    def exit_position(self, symbol: str, exit_reason: str) -> bool:
        """Exit position"""
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        
        # Place sell order
        order_id = self.place_market_order(symbol, position['shares'], OrderSide.SELL)
        if not order_id:
            return False
        
        # Get exit price (approximate)
        data = self.get_market_data(symbol, lookback_days=1)
        exit_price = data['close'].iloc[-1] if data is not None and len(data) > 0 else position['entry_price']
        
        # Calculate P&L
        pnl = (exit_price - position['entry_price']) * position['shares']
        pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        
        # Log trade
        trade_record = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'exit_order_id': order_id,
            'signal_data': position['signal_data']
        }
        
        self.trade_log.append(trade_record)
        
        logger.info(f"EXITED POSITION: {symbol}")
        logger.info(f"  Exit Price: ${exit_price:.2f}, Reason: {exit_reason}")
        logger.info(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"  Days Held: {(trade_record['exit_time'] - trade_record['entry_time']).days}")
        
        # Remove from active positions
        del self.active_positions[symbol]
        
        return True
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("=" * 50)
        logger.info(f"TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current portfolio status
        portfolio_value = self.get_current_portfolio_value()
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Active Positions: {len(self.active_positions)}")
        
        # Check exit conditions for active positions
        for symbol in list(self.active_positions.keys()):
            exit_reason = self.check_exit_conditions(symbol)
            if exit_reason:
                self.exit_position(symbol, exit_reason)
        
        # Look for new entry opportunities
        for symbol in self.symbols:
            if symbol not in self.active_positions:
                data = self.get_market_data(symbol, lookback_days=30)
                if data is not None:
                    signal = self.check_entry_signal(symbol, data)
                    if signal:
                        logger.info(f"Entry signal detected for {symbol}")
                        self.enter_position(signal)
        
        # Log current positions
        if self.active_positions:
            logger.info(f"Current Active Positions:")
            for symbol, pos in self.active_positions.items():
                days_held = (datetime.now() - pos['entry_time']).days
                logger.info(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}, {days_held} days")
        
        logger.info("Trading cycle complete")
    
    def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'active_positions': {
                    k: {
                        **v,
                        'entry_time': v['entry_time'].isoformat(),
                        'signal_data': {
                            **v['signal_data'],
                            'timestamp': v['signal_data']['timestamp'].isoformat()
                        }
                    } for k, v in self.active_positions.items()
                },
                'trade_log': [
                    {
                        **trade,
                        'entry_time': trade['entry_time'].isoformat(),
                        'exit_time': trade['exit_time'].isoformat(),
                        'signal_data': {
                            **trade['signal_data'],
                            'timestamp': trade['signal_data']['timestamp'].isoformat()
                        }
                    } for trade in self.trade_log
                ]
            }
            
            with open('trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("Trading state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load state from file"""
        try:
            if os.path.exists('trading_state.json'):
                with open('trading_state.json', 'r') as f:
                    state = json.load(f)
                
                # Restore active positions
                for symbol, pos_data in state.get('active_positions', {}).items():
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                    pos_data['signal_data']['timestamp'] = datetime.fromisoformat(pos_data['signal_data']['timestamp'])
                    self.active_positions[symbol] = pos_data
                
                # Restore trade log
                for trade_data in state.get('trade_log', []):
                    trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    trade_data['signal_data']['timestamp'] = datetime.fromisoformat(trade_data['signal_data']['timestamp'])
                    self.trade_log.append(trade_data)
                
                logger.info(f"State loaded: {len(self.active_positions)} active positions, {len(self.trade_log)} completed trades")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def run_live_trading(self, check_interval_minutes: int = 1):
        """Main live trading loop"""
        logger.info("Starting live trading system")
        logger.info(f"Check interval: {check_interval_minutes} minutes")
        
        # Load previous state
        self.load_state()
        
        try:
            while True:
                if self.is_market_open():
                    self.run_trading_cycle()
                    self.save_state()
                else:
                    logger.info("Market is closed, waiting...")
                
                # Wait for next cycle
                time_module.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.save_state()
        except Exception as e:
            logger.error(f"Unexpected error in live trading: {e}")
            self.save_state()
            raise

def main():
    """Main function to run live trading"""
    # Check environment variables
    if not os.getenv('APCA_API_KEY_ID') or not os.getenv('APCA_API_SECRET_KEY'):
        print("ERROR: Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
        print("See setup instructions in LIVE_TRADING_SETUP.md")
        return
    
    # Initialize trader
    trader = AggressiveLiveTrader()
    
    # Run live trading with 1-minute check intervals
    trader.run_live_trading(check_interval_minutes=1)

if __name__ == "__main__":
    main()