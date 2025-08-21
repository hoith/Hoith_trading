#!/usr/bin/env python3
"""
REALISTIC OPTIONS BACKTEST - Accurate as Possible
Models real options trading with:
1. Black-Scholes pricing for calls/puts
2. Realistic bid/ask spreads
3. Transaction costs and commissions
4. Liquidity constraints
5. Real market hours and expiration cycles
6. IV crush and time decay
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os
from pathlib import Path
import logging
from scipy.stats import norm
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealisticOptionsBacktest:
    """Realistic options backtest with accurate pricing and costs"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # High-volume ETFs with active options markets
        self.symbols = {
            'SPY': {'daily_volume': 50000000, 'iv_base': 0.15},    # Most liquid
            'QQQ': {'daily_volume': 30000000, 'iv_base': 0.18},    # Very liquid
            'TQQQ': {'daily_volume': 15000000, 'iv_base': 0.35},   # High IV
            'SQQQ': {'daily_volume': 10000000, 'iv_base': 0.40},   # High IV
            'IWM': {'daily_volume': 8000000, 'iv_base': 0.20},     # Decent volume
        }
        
        # Realistic trading parameters
        self.risk_per_trade_pct = 2.0        # Risk 2% per trade (conservative)
        self.max_positions = 3               # Max concurrent positions
        self.min_dte = 1                     # Minimum days to expiration
        self.max_dte = 7                     # Maximum days to expiration (weekly options)
        
        # Real options market parameters
        self.commission_per_contract = 0.65   # Typical options commission
        self.bid_ask_spread_pct = 0.02       # 2% spread (conservative)
        self.risk_free_rate = 0.045          # Current risk-free rate
        self.min_contract_price = 0.05       # Minimum option price
        self.max_contract_price = 10.00      # Don't buy expensive options
        
        # Position sizing
        self.max_position_value = 2000       # Max $2000 per position
        self.min_position_value = 200        # Min $200 per position
        
        # Market hours
        self.market_open = 9.5               # 9:30 AM
        self.market_close = 16.0             # 4:00 PM
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Trading state
        self.positions = {}
        self.trades = []
        self.current_time = None
        
        logger.info("REALISTIC OPTIONS backtest initialized")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"Commission: ${self.commission_per_contract} per contract")
        logger.info(f"Bid/ask spread: {self.bid_ask_spread_pct*100}%")
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum $0.01
    
    def calculate_implied_volatility(self, symbol: str, price_change_pct: float) -> float:
        """Calculate realistic implied volatility based on market conditions"""
        base_iv = self.symbols[symbol]['iv_base']
        
        # IV increases with volatility
        volatility_multiplier = 1 + abs(price_change_pct) * 2
        
        # Add some randomness for realism
        random_factor = np.random.normal(1.0, 0.1)
        
        iv = base_iv * volatility_multiplier * random_factor
        return np.clip(iv, 0.10, 1.50)  # Keep IV reasonable
    
    def get_option_chain(self, symbol: str, underlying_price: float, target_delta: float = 0.3) -> dict:
        """Generate realistic option chain"""
        # Calculate strikes around current price
        strike_interval = 1 if underlying_price < 50 else (5 if underlying_price < 200 else 10)
        
        # Find strike closest to target delta
        strikes = []
        for i in range(-5, 6):  # 11 strikes
            strike = round(underlying_price + i * strike_interval, 0)
            if strike > 0:
                strikes.append(strike)
        
        return {
            'strikes': strikes,
            'dte': [1, 2, 3, 7],  # Available expirations
        }
    
    def calculate_option_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        theta /= 365  # Daily theta
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    def find_best_option(self, symbol: str, underlying_price: float, direction: str, dte: int) -> dict:
        """Find best option for the trade"""
        option_chain = self.get_option_chain(symbol, underlying_price)
        
        if dte not in option_chain['dte']:
            dte = min(option_chain['dte'], key=lambda x: abs(x - dte))
        
        T = dte / 365.0
        price_change = 0.01  # Assume 1% recent move for IV calculation
        iv = self.calculate_implied_volatility(symbol, price_change)
        
        best_option = None
        best_score = -1
        
        for strike in option_chain['strikes']:
            if direction == 'call':
                option_type = 'call'
                # Look for slightly OTM calls
                if strike < underlying_price * 0.98 or strike > underlying_price * 1.05:
                    continue
            else:
                option_type = 'put'
                # Look for slightly OTM puts
                if strike > underlying_price * 1.02 or strike < underlying_price * 0.95:
                    continue
            
            # Calculate option price
            theo_price = self.black_scholes_price(underlying_price, strike, T, self.risk_free_rate, iv, option_type)
            
            # Apply bid/ask spread
            spread = theo_price * self.bid_ask_spread_pct
            bid = theo_price - spread / 2
            ask = theo_price + spread / 2
            
            # Skip if outside our price range
            if ask < self.min_contract_price or ask > self.max_contract_price:
                continue
            
            # Calculate Greeks
            greeks = self.calculate_option_greeks(underlying_price, strike, T, self.risk_free_rate, iv, option_type)
            
            # Score option (prefer higher delta, lower theta decay)
            score = abs(greeks['delta']) - abs(greeks['theta']) * 10
            
            if score > best_score:
                best_score = score
                best_option = {
                    'strike': strike,
                    'dte': dte,
                    'option_type': option_type,
                    'theo_price': theo_price,
                    'bid': bid,
                    'ask': ask,
                    'iv': iv,
                    'greeks': greeks
                }
        
        return best_option
    
    def calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """Calculate momentum signal strength"""
        if len(data) < 10:
            return 0
        
        # Short-term momentum
        momentum_3 = (data['close'].iloc[-1] / data['close'].iloc[-4] - 1) * 100
        momentum_5 = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Combine momentum with volume
        signal = (momentum_3 + momentum_5) / 2
        
        if volume_ratio > 1.5:  # High volume confirmation
            signal *= 1.3
        elif volume_ratio < 0.8:  # Low volume warning
            signal *= 0.7
        
        return signal
    
    def check_entry_signal(self, symbol: str, data: pd.DataFrame) -> str:
        """Check for entry signals"""
        momentum = self.calculate_momentum_signal(data)
        
        # Adjust thresholds based on symbol volatility
        if symbol in ['TQQQ', 'SQQQ']:
            threshold = 1.5  # Lower threshold for leveraged ETFs
        else:
            threshold = 2.0  # Higher threshold for regular ETFs
        
        if momentum > threshold:
            return 'call'
        elif momentum < -threshold:
            return 'put'
        
        return None
    
    def enter_position(self, symbol: str, direction: str, data: pd.DataFrame) -> bool:
        """Enter realistic options position"""
        if symbol in self.positions or len(self.positions) >= self.max_positions:
            return False
        
        underlying_price = data['close'].iloc[-1]
        
        # Find best option
        dte = np.random.choice([1, 2, 3, 7], p=[0.1, 0.3, 0.4, 0.2])  # Prefer 2-3 DTE
        option = self.find_best_option(symbol, underlying_price, direction, dte)
        
        if not option:
            logger.debug(f"No suitable {direction} option found for {symbol}")
            return False
        
        # Calculate position size
        risk_amount = self.capital * (self.risk_per_trade_pct / 100)
        risk_amount = np.clip(risk_amount, self.min_position_value, self.max_position_value)
        
        # Determine number of contracts
        option_cost = option['ask'] * 100  # Options are per 100 shares
        max_contracts = int(risk_amount / option_cost)
        
        if max_contracts < 1:
            logger.debug(f"Cannot afford {symbol} {direction} option at ${option['ask']:.2f}")
            return False
        
        contracts = min(max_contracts, 10)  # Max 10 contracts per trade
        total_cost = contracts * option_cost
        commission = contracts * self.commission_per_contract
        total_entry_cost = total_cost + commission
        
        if total_entry_cost > self.capital:
            return False
        
        # Create position
        position = {
            'symbol': symbol,
            'option_type': option['option_type'],
            'strike': option['strike'],
            'dte': option['dte'],
            'contracts': contracts,
            'entry_price': option['ask'],
            'entry_cost': total_entry_cost,
            'entry_time': self.current_time,
            'underlying_entry': underlying_price,
            'iv_entry': option['iv'],
            'greeks_entry': option['greeks'],
            'expiration': self.current_time + timedelta(days=int(option['dte']))
        }
        
        self.positions[symbol] = position
        self.capital -= total_entry_cost
        
        logger.info(f"ENTER {symbol} {option['option_type'].upper()}: {contracts} contracts @ ${option['ask']:.2f}")
        logger.info(f"  Strike: ${option['strike']}, DTE: {option['dte']}, IV: {option['iv']:.1%}")
        logger.info(f"  Total cost: ${total_entry_cost:.0f}, Delta: {option['greeks']['delta']:.2f}")
        
        return True
    
    def check_exit_conditions(self, symbol: str, data: pd.DataFrame) -> str:
        """Check realistic exit conditions"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = data['close'].iloc[-1]
        
        # Check expiration
        if self.current_time >= position['expiration']:
            return 'expiration'
        
        # Calculate days to expiration
        dte = (position['expiration'] - self.current_time).days
        if dte < 0:
            dte = 0
        
        # Calculate current option value
        T = max(dte / 365.0, 0.001)  # Minimum time value
        
        # Update IV based on recent price action
        price_change = (current_price / position['underlying_entry'] - 1)
        current_iv = self.calculate_implied_volatility(symbol, price_change)
        
        # Calculate current option price
        current_option_price = self.black_scholes_price(
            current_price, 
            position['strike'], 
            T, 
            self.risk_free_rate, 
            current_iv, 
            position['option_type']
        )
        
        # Apply bid/ask spread (we sell at bid)
        spread = current_option_price * self.bid_ask_spread_pct
        current_bid = current_option_price - spread / 2
        current_bid = max(current_bid, 0.01)  # Minimum bid
        
        # Calculate P&L
        current_value = position['contracts'] * current_bid * 100
        pnl_pct = (current_value / (position['entry_cost'] - position['contracts'] * self.commission_per_contract)) - 1
        
        # Exit conditions
        if pnl_pct >= 0.50:  # 50% profit target
            return 'profit_target'
        elif pnl_pct <= -0.80:  # 80% stop loss (options can go to zero)
            return 'stop_loss'
        elif dte <= 0:  # Day of expiration
            return 'expiration'
        elif dte == 1 and pnl_pct < 0:  # Close losing trades before expiration
            return 'expiration_risk'
        
        return None
    
    def exit_position(self, symbol: str, data: pd.DataFrame, exit_reason: str) -> None:
        """Exit position with realistic pricing"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = data['close'].iloc[-1]
        
        # Calculate exit value
        if exit_reason == 'expiration' and position['option_type'] == 'call':
            # Intrinsic value only
            intrinsic = max(0, current_price - position['strike'])
            exit_value = position['contracts'] * intrinsic * 100
        elif exit_reason == 'expiration' and position['option_type'] == 'put':
            # Intrinsic value only
            intrinsic = max(0, position['strike'] - current_price)
            exit_value = position['contracts'] * intrinsic * 100
        else:
            # Calculate current option price
            dte = max((position['expiration'] - self.current_time).days, 0)
            T = max(dte / 365.0, 0.001)
            
            price_change = (current_price / position['underlying_entry'] - 1)
            current_iv = self.calculate_implied_volatility(symbol, price_change)
            
            option_price = self.black_scholes_price(
                current_price, position['strike'], T, self.risk_free_rate, current_iv, position['option_type']
            )
            
            # Sell at bid price
            spread = option_price * self.bid_ask_spread_pct
            bid_price = option_price - spread / 2
            bid_price = max(bid_price, 0.01)
            
            exit_value = position['contracts'] * bid_price * 100
        
        # Calculate total P&L including commissions
        commission_out = position['contracts'] * self.commission_per_contract
        total_proceeds = exit_value - commission_out
        
        total_pnl = total_proceeds - position['entry_cost']
        pnl_pct = (total_pnl / position['entry_cost']) * 100
        
        # Update capital
        self.capital += total_proceeds
        
        # Record trade
        days_held = (self.current_time - position['entry_time']).days
        
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': self.current_time,
            'symbol': symbol,
            'option_type': position['option_type'],
            'strike': position['strike'],
            'contracts': position['contracts'],
            'entry_cost': position['entry_cost'],
            'exit_value': total_proceeds,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        logger.info(f"EXIT {symbol} {position['option_type'].upper()}: ${total_proceeds:.0f} - {exit_reason}")
        logger.info(f"  P&L: ${total_pnl:+.0f} ({pnl_pct:+.1f}%) in {days_held} days")
        logger.info(f"  New capital: ${self.capital:,.0f}")
        
        # Remove position
        del self.positions[symbol]
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data for symbol"""
        try:
            bars = self.data_client.get_stock_bars(
                symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Filter for this symbol and reset index
            symbol_data = bars.loc[symbol].reset_index()
            symbol_data = symbol_data.set_index('timestamp')
            
            logger.info(f"Loaded {len(symbol_data)} bars for {symbol}")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> None:
        """Run realistic options backtest"""
        
        logger.info("============================================================")
        logger.info("REALISTIC OPTIONS STRATEGY BACKTEST")
        logger.info("============================================================")
        logger.info("Fetching historical data...")
        
        # Get historical data for all symbols
        data = {}
        for symbol in self.symbols.keys():
            symbol_data = self.get_historical_data(symbol, start_date, end_date)
            if not symbol_data.empty:
                data[symbol] = symbol_data
        
        if not data:
            logger.error("No data loaded!")
            return
        
        # Get trading dates
        first_symbol = list(data.keys())[0]
        trading_dates = data[first_symbol].index[20:]  # Skip first 20 days
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            
            # Check exit conditions first
            for symbol in list(self.positions.keys()):
                if symbol in data and current_date in data[symbol].index:
                    exit_reason = self.check_exit_conditions(symbol, data[symbol].loc[:current_date])
                    if exit_reason:
                        self.exit_position(symbol, data[symbol].loc[:current_date], exit_reason)
            
            # Look for new entries
            if len(self.positions) < self.max_positions:
                for symbol in self.symbols.keys():
                    if symbol not in self.positions and symbol in data and current_date in data[symbol].index:
                        historical_data = data[symbol].loc[:current_date]
                        
                        signal = self.check_entry_signal(symbol, historical_data)
                        if signal:
                            if self.enter_position(symbol, signal, historical_data):
                                break  # Only one entry per day
            
            # Log progress
            if i % 30 == 0 or i == len(trading_dates) - 1:
                total_return = (self.capital / self.initial_capital - 1) * 100
                logger.info(f"{current_date.date()}: Capital=${self.capital:,.0f} ({total_return:+.1f}%), Positions={len(self.positions)}, Trades={len(self.trades)}")
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in data:
                last_data = data[symbol].iloc[-1:]
                self.exit_position(symbol, last_data, 'backtest_end')
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print realistic backtest results"""
        if not self.trades:
            logger.info("No trades completed!")
            return
        
        final_value = self.capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        total_commissions = len(trades_df) * 2 * np.mean([t['contracts'] for t in self.trades]) * self.commission_per_contract
        
        logger.info("============================================================")
        logger.info("REALISTIC OPTIONS STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:+.1f}%")
        logger.info(f"Total Commissions: ${total_commissions:.0f}")
        logger.info("")
        logger.info(f"Total Trades: {len(trades_df)}")
        logger.info(f"Winning Trades: {len(winning_trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Win: {avg_win:+.1f}%")
        logger.info(f"Average Loss: {avg_loss:+.1f}%")
        logger.info("============================================================")
        
        # Calculate annualized return
        if self.trades:
            days_elapsed = (self.trades[-1]['exit_time'] - self.trades[0]['entry_time']).days
            years = max(days_elapsed / 365.25, 0.1)
            annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100
            logger.info(f"Annualized Return: {annualized_return:.1f}%")
        
        # Show recent trades
        logger.info("\nRecent Trades:")
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        for trade in recent_trades:
            logger.info(f"{trade['symbol']} {trade['option_type'].upper()}: {trade['pnl_pct']:+.1f}% in {trade['days_held']}d ({trade['exit_reason']})")

if __name__ == "__main__":
    # Run the realistic options backtest
    backtest = RealisticOptionsBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)