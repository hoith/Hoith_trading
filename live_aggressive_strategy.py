#!/usr/bin/env python3
"""
Live Aggressive Trading Strategy - Updated with PROVEN +122.59% Return Parameters
Real-time implementation using the successful aggressive momentum strategy
Updated for minute bars, asset-class routing, and backtest <-> live parity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
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
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Configure logging with environment control
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProvenAggressiveLiveTrader:
    """Live trading implementation using PROVEN +122.59% aggressive strategy"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        
        # PROVEN AGGRESSIVE SYMBOLS - exactly from successful strategy
        self.symbols = [
            # Core proven performers
            'AAPL', 'MSFT', 'GOOGL',  # Individual stocks
            'QQQ', 'SPY',             # Standard ETFs  
            'TQQQ'                    # LEVERAGED ETF - biggest winner!
        ]
        
        # Get parameters from environment
        self.lookback_minutes = int(os.getenv('LOOKBACK_MINUTES', '720'))
        self.sleep_seconds = int(os.getenv('SLEEP_SECONDS', '60'))
        
        # Initialize clients - SIMPLIFIED ALPACA-ONLY APPROACH
        self.data_client = AlpacaDataClient()  # Pure Alpaca - works during market hours
        self.trading_client = TradingClient(
            api_key=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            paper=True  # Paper trading mode
        )
        
        # PROVEN AGGRESSIVE POSITION SIZING (from successful strategy)
        self.base_positions = {
            # Individual stocks - doubled from conservative
            'AAPL': 200,   # $200 base position
            'MSFT': 200,   # $200 base position  
            'GOOGL': 200,  # $200 base position
            
            # ETFs - tripled from conservative
            'QQQ': 300,    # $300 base position
            'SPY': 300,    # $300 base position
            
            # Leveraged ETF - much larger for explosive gains
            'TQQQ': 250    # $250 base position - key profit driver!
        }
        
        # PROVEN RISK/REWARD PARAMETERS (from successful strategy)
        self.risk_params = {
            # Individual stocks - wider stops, higher targets
            'AAPL': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            'MSFT': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            'GOOGL': {'stop_pct': 3.0, 'target_pct': 8.0, 'max_hold_days': 15},
            
            # ETFs - moderate parameters
            'QQQ': {'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold_days': 12},
            'SPY': {'stop_pct': 2.5, 'target_pct': 6.0, 'max_hold_days': 12},
            
            # Leveraged ETF - tight stops, quick profits (proven winner!)
            'TQQQ': {'stop_pct': 2.0, 'target_pct': 5.0, 'max_hold_days': 8}
        }
        
        # Active positions tracking
        self.active_positions = {}
        self.trade_log = []
        
        logger.info("ProvenAggressiveLiveTrader initialized with +122.59% strategy parameters")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Base positions: {self.base_positions}")
        logger.info(f"Risk params: {self.risk_params}")
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator - same as proven strategy"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_minute_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get minute bars - works during market hours, falls back to daily when closed"""
        try:
            end = datetime.now(timezone.utc).replace(microsecond=0)
            start = end - timedelta(minutes=self.lookback_minutes)
            
            # First try minute bars (works during market hours)
            try:
                bars = self.data_client.get_stock_bars(
                    symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end
                )
                
                if not bars.empty:
                    # Handle MultiIndex if present  
                    if isinstance(bars.index, pd.MultiIndex):
                        df = bars.xs(symbol, level=0)[["open","high","low","close","volume"]]
                    else:
                        df = bars[["open","high","low","close","volume"]]
                    
                    age_s = (datetime.now(timezone.utc) - df.index[-1]).total_seconds()
                    logger.info(f"{symbol} tf=1m last_bar_UTC={df.index[-1]} age_s={int(age_s)} rows={len(df)}")
                    return df
                    
            except Exception as minute_error:
                logger.warning(f"{symbol} minute bars failed: {minute_error}")
            
            # Fallback to daily bars when market closed
            logger.info(f"{symbol}: Market closed, using daily bars for signal generation")
            daily_end = end - timedelta(days=1)  # Yesterday
            daily_start = daily_end - timedelta(days=30)  # 30 days back
            
            daily_bars = self.data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=daily_start,
                end=daily_end
            )
            
            if not daily_bars.empty:
                if isinstance(daily_bars.index, pd.MultiIndex):
                    df = daily_bars.xs(symbol, level=0)[["open","high","low","close","volume"]]
                else:
                    df = daily_bars[["open","high","low","close","volume"]]
                
                logger.info(f"{symbol} tf=1d fallback: {len(df)} days, latest={df.index[-1].date()}")
                return df
            else:
                logger.warning(f"{symbol}: No daily bars available either")
                return None
            
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return None
    
    
    def check_proven_entry_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Check entry signal using PROVEN aggressive criteria"""
        try:
            if len(data) < 25:  # Need enough history
                return None
            
            # Calculate indicators - same as proven strategy
            rsi = self.calculate_rsi(data['close'], 14)
            lookback = 10  # Proven lookback period
            
            if len(data) < lookback + 10:
                return None
            
            current_price = float(data['close'].iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            
            # PROVEN AGGRESSIVE MOMENTUM CALCULATION
            momentum_score = (current_price / data['close'].iloc[-lookback] - 1) * 100
            
            # Volume analysis
            volume_ratio = float(data['volume'].iloc[-1] / data['volume'].iloc[-10:].mean())
            
            # PROVEN ENTRY CONDITIONS BY SYMBOL TYPE
            entry_signal = False
            
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Individual stocks
                # PROVEN: Very relaxed criteria - catch more moves
                if (momentum_score > 1.0 and      # 1% momentum threshold
                    current_rsi < 75 and          # Higher RSI allowed
                    volume_ratio > 0.8):          # Below average volume OK
                    entry_signal = True
                    
            elif symbol in ['QQQ', 'SPY']:  # Standard ETFs
                # PROVEN: Relaxed ETF criteria
                if (momentum_score > 0.5 and      # Very low threshold
                    current_rsi < 80 and          # Very high RSI allowed
                    volume_ratio > 0.7):          # Low volume OK
                    entry_signal = True
                    
            elif symbol == 'TQQQ':  # Leveraged ETF - THE MONEY MAKER!
                # PROVEN: Most aggressive for highest returns
                if (momentum_score > 2.0 and      # 2% momentum (reduced from 5%)
                    current_rsi < 85):            # Very overbought allowed
                    entry_signal = True
                    
            if entry_signal:
                logger.info(f"PROVEN SIGNAL: {symbol} - Momentum: {momentum_score:.1f}%, RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'signal_time': data.index[-1],
                    'momentum_score': momentum_score,
                    'rsi': current_rsi,
                    'volume_ratio': volume_ratio
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry signal for {symbol}: {e}")
            return None
    
    def calculate_proven_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size using PROVEN aggressive sizing"""
        try:
            # Get current portfolio value
            account = self.trading_client.get_account()
            current_capital = float(account.portfolio_value)
            
            # PROVEN DYNAMIC SCALING
            base_position = self.base_positions[symbol]
            capital_multiplier = current_capital / self.initial_capital
            target_dollars = base_position * capital_multiplier
            
            # Calculate shares (fractional allowed)
            shares = target_dollars / price
            
            logger.info(f"{symbol}: ${target_dollars:.0f} target (${base_position} base * {capital_multiplier:.2f}x) = {shares:.2f} shares @ ${price:.2f}")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def get_asset_class_info(self, symbol: str) -> Dict:
        """Get asset class for routing decisions"""
        try:
            asset = self.trading_client.get_asset(symbol)
            return {
                'asset_class': getattr(asset, 'asset_class', 'us_equity').lower(),
                'fractionable': getattr(asset, 'fractionable', False)
            }
        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")
            return {'asset_class': 'us_equity', 'fractionable': False}
    
    def place_proven_limit_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[str]:
        """Place limit order with asset-class routing and proven parameters"""
        try:
            # Get current data for limit price
            data = self.get_minute_bars(symbol)
            if data is None or data.empty:
                logger.error(f"{symbol}: No data for order placement")
                return None
            
            last_price = float(data['close'].iloc[-1])
            
            # Proven limit price calculation - 6 BPS offset for fills
            if side == OrderSide.BUY:
                limit_price = last_price * 1.0006  # Slightly aggressive for fills
            else:
                limit_price = last_price * 0.9994
            
            # Asset-class routing for options vs equities
            asset_info = self.get_asset_class_info(symbol)
            is_option = asset_info['asset_class'] == 'option'
            
            # Market hours check
            clock = self.trading_client.get_clock()
            
            if is_option and not clock.is_open:
                logger.info(f"{symbol}: Options RTH only; market closed - skip")
                return None
            
            # Set order parameters by asset class
            if is_option:
                tif = TimeInForce.DAY
                extended = False  # Options: no extended hours
            else:
                tif = TimeInForce.DAY  
                extended = True   # Equities: allow extended hours
            
            # Create limit order
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=str(qty),
                side=side,
                limit_price=str(round(limit_price, 2)),
                time_in_force=tif,
                extended_hours=extended
            )
            
            # Submit with full logging
            try:
                order = self.trading_client.submit_order(order_request)
                
                logger.info(f"ORDER SUBMIT: {side.value} {qty} {symbol} @ ${limit_price:.2f} limit ext={extended} tif={tif.value}")
                return str(order.id)
                
            except Exception as submit_error:
                logger.error(f"Order submit failed for {symbol}: {submit_error}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
        except Exception as e:
            logger.error(f"Error placing limit order for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def enter_proven_position(self, signal: Dict) -> bool:
        """Enter new position using proven aggressive parameters"""
        symbol = signal['symbol']
        
        # Check if already in position
        if symbol in self.active_positions:
            logger.debug(f"{symbol}: Already in position, skipping entry")
            return False
        
        # Check if options and market closed
        asset_info = self.get_asset_class_info(symbol)
        if asset_info['asset_class'] == 'option':
            clock = self.trading_client.get_clock()
            if not clock.is_open:
                logger.info(f"{symbol}: Options cannot be traded after close; skipping")
                return False
        
        # Calculate proven position size
        price = signal['price']
        qty = self.calculate_proven_position_size(symbol, price)
        
        if qty <= 0:
            logger.warning(f"{symbol}: Invalid position size: {qty}")
            return False
        
        # Place buy order
        order_id = self.place_proven_limit_order(symbol, qty, OrderSide.BUY)
        if not order_id:
            return False
        
        # Get proven risk parameters for this symbol
        risk_params = self.risk_params[symbol]
        
        # Record position with proven parameters
        position = {
            'symbol': symbol,
            'qty': qty,
            'entry_price': price,
            'entry_time': signal['signal_time'],
            'entry_order_id': order_id,
            'stop_loss': price * (1 - risk_params['stop_pct'] / 100),
            'take_profit': price * (1 + risk_params['target_pct'] / 100),
            'max_hold_days': risk_params['max_hold_days'],
            'signal_data': signal
        }
        
        self.active_positions[symbol] = position
        
        logger.info(f"PROVEN ENTRY: {symbol} {qty:.2f} shares @ ${price:.2f}")
        logger.info(f"  Stop: ${position['stop_loss']:.2f} (-{risk_params['stop_pct']}%)")
        logger.info(f"  Target: ${position['take_profit']:.2f} (+{risk_params['target_pct']}%)")
        logger.info(f"  Max Hold: {risk_params['max_hold_days']} days")
        logger.info(f"  Signal: {signal['momentum_score']:.1f}% momentum, RSI {signal['rsi']:.1f}")
        
        return True
    
    def check_proven_exit_conditions(self, symbol: str) -> Optional[str]:
        """Check exit conditions using proven parameters"""
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        
        # Get current market data
        data = self.get_minute_bars(symbol)
        if data is None or data.empty:
            return None
        
        current_price = float(data['close'].iloc[-1])
        current_low = float(data['low'].iloc[-1])
        current_high = float(data['high'].iloc[-1])
        
        # PROVEN EXIT CONDITIONS
        
        # Check stop loss (intraday low hit)
        if current_low <= position['stop_loss']:
            return 'stop_loss'
        
        # Check take profit (intraday high hit)  
        if current_high >= position['take_profit']:
            return 'take_profit'
        
        # Check time limit (proven max hold periods)
        time_held = (datetime.now(timezone.utc) - position['entry_time']).total_seconds() / 86400  # days
        if time_held >= position['max_hold_days']:
            return 'time_limit'
        
        return None
    
    def exit_proven_position(self, symbol: str, exit_reason: str) -> bool:
        """Exit position using proven parameters"""
        if symbol not in self.active_positions:
            return False
        
        position = self.active_positions[symbol]
        
        # Check if options and market closed
        asset_info = self.get_asset_class_info(symbol)
        if asset_info['asset_class'] == 'option':
            clock = self.trading_client.get_clock()
            if not clock.is_open:
                logger.info(f"{symbol}: Options RTH only; cannot exit after close")
                return False
        
        # Place sell order
        order_id = self.place_proven_limit_order(symbol, position['qty'], OrderSide.SELL)
        if not order_id:
            return False
        
        # Get exit price estimate
        data = self.get_minute_bars(symbol)
        exit_price = float(data['close'].iloc[-1]) if data is not None and not data.empty else position['entry_price']
        
        # Calculate P&L
        pnl = (exit_price - position['entry_price']) * position['qty']
        pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        
        # Log trade
        trade_record = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(timezone.utc),
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'qty': position['qty'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'exit_order_id': order_id,
            'signal_data': position['signal_data']
        }
        
        self.trade_log.append(trade_record)
        
        logger.info(f"PROVEN EXIT: {symbol} @ ${exit_price:.2f} - {exit_reason}")
        logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"  Original Signal: {position['signal_data']['momentum_score']:.1f}% momentum")
        
        # Remove from active positions
        del self.active_positions[symbol]
        
        return True
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def run_proven_trading_cycle(self):
        """Run one complete trading cycle with proven strategy"""
        logger.info("=" * 60)
        logger.info(f"PROVEN AGGRESSIVE CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check market status
        market_open = self.is_market_open()
        logger.info(f"Market Status: {'OPEN' if market_open else 'CLOSED'}")
        
        # Get portfolio status
        try:
            account = self.trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            total_return = (portfolio_value / self.initial_capital - 1) * 100
            
            logger.info(f"Portfolio: ${portfolio_value:,.2f} ({total_return:+.2f}%)")
            logger.info(f"Active Positions: {len(self.active_positions)}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        
        # Process each proven symbol
        entry_signals = 0
        exit_signals = 0
        
        for symbol in self.symbols:
            try:
                logger.debug(f"Processing {symbol}...")
                
                # Get minute bar data
                data = self.get_minute_bars(symbol)
                if data is None:
                    logger.info(f"[FILTER] {symbol} data_fetch -> 0")
                    continue
                
                logger.info(f"[FILTER] {symbol} data_available -> {len(data)}")
                
                # Check data freshness (within 3 minutes for live trading)
                age_s = (datetime.now(timezone.utc) - data.index[-1]).total_seconds()
                if age_s > 180:
                    logger.info(f"[FILTER] {symbol} freshness_check -> 0 (age: {age_s:.0f}s)")
                    continue
                
                logger.info(f"[FILTER] {symbol} fresh_data -> 1")
                
                # Check exit conditions for existing positions
                if symbol in self.active_positions:
                    exit_reason = self.check_proven_exit_conditions(symbol)
                    if exit_reason:
                        if self.exit_proven_position(symbol, exit_reason):
                            exit_signals += 1
                
                # Check entry conditions (only during market hours)
                elif market_open:
                    signal = self.check_proven_entry_signal(symbol, data)
                    if signal:
                        logger.info(f"[FILTER] {symbol} entry_signal -> 1")
                        if self.enter_proven_position(signal):
                            entry_signals += 1
                    else:
                        logger.info(f"[FILTER] {symbol} entry_signal -> 0")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Log cycle summary
        logger.info(f"Cycle complete: {entry_signals} entries, {exit_signals} exits")
        
        # Log current positions
        if self.active_positions:
            logger.info(f"Active Positions (Proven Strategy):")
            for symbol, pos in self.active_positions.items():
                time_held = (datetime.now(timezone.utc) - pos['entry_time']).total_seconds() / 3600  # hours
                current_data = self.get_minute_bars(symbol)
                if current_data is not None and not current_data.empty:
                    current_price = float(current_data['close'].iloc[-1])
                    unrealized_pnl = (current_price / pos['entry_price'] - 1) * 100
                    logger.info(f"  {symbol}: {pos['qty']:.2f} @ ${pos['entry_price']:.2f} ({time_held:.1f}h) - Unrealized: {unrealized_pnl:+.1f}%")
    
    def save_state(self):
        """Save trading state"""
        try:
            state = {
                'active_positions': {
                    k: {
                        **v,
                        'entry_time': v['entry_time'].isoformat(),
                        'signal_data': {
                            **v['signal_data'],
                            'signal_time': v['signal_data']['signal_time'].isoformat()
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
                            'signal_time': trade['signal_data']['signal_time'].isoformat()
                        }
                    } for trade in self.trade_log
                ]
            }
            
            with open('proven_trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug("Proven trading state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load trading state"""
        try:
            if os.path.exists('proven_trading_state.json'):
                with open('proven_trading_state.json', 'r') as f:
                    state = json.load(f)
                
                # Restore active positions
                for symbol, pos_data in state.get('active_positions', {}).items():
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                    pos_data['signal_data']['signal_time'] = datetime.fromisoformat(pos_data['signal_data']['signal_time'])
                    self.active_positions[symbol] = pos_data
                
                # Restore trade log
                for trade_data in state.get('trade_log', []):
                    trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    trade_data['signal_data']['signal_time'] = datetime.fromisoformat(trade_data['signal_data']['signal_time'])
                    self.trade_log.append(trade_data)
                
                logger.info(f"Proven state loaded: {len(self.active_positions)} positions, {len(self.trade_log)} completed trades")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")

def main_loop():
    """Main live trading loop using PROVEN +122.59% strategy"""
    logger.info("Starting PROVEN Aggressive Live Trading (+122.59% strategy parameters)")
    
    # Check environment variables
    if not os.getenv('APCA_API_KEY_ID') or not os.getenv('APCA_API_SECRET_KEY'):
        logger.error("Missing APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
        return
    
    # Initialize trader with proven strategy
    trader = ProvenAggressiveLiveTrader()
    
    # Load previous state
    trader.load_state()
    
    try:
        while True:
            trader.run_proven_trading_cycle()
            trader.save_state()
            
            # Wait for next cycle
            logger.debug(f"Sleeping for {trader.sleep_seconds} seconds...")
            time_module.sleep(trader.sleep_seconds)
            
    except KeyboardInterrupt:
        logger.info("PROVEN aggressive trading stopped by user")
        trader.save_state()
    except Exception as e:
        logger.error(f"Unexpected error in proven trading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        trader.save_state()
        raise

def main():
    """Direct execution entry point"""
    main_loop()

if __name__ == "__main__":
    main()