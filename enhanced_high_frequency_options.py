#!/usr/bin/env python3
"""
ENHANCED HIGH-FREQUENCY OPTIONS STRATEGY
Target: 10+ trades daily with expanded ticker universe for maximum profit potential

Key Features:
- 25+ high-volume tickers across sectors
- Multiple timeframes for signal generation
- Aggressive position sizing for rapid account growth
- Realistic options pricing with Black-Scholes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os
from pathlib import Path
import logging
from scipy.stats import norm

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

class EnhancedHighFrequencyBacktest:
    """Enhanced high-frequency options strategy with expanded ticker universe"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # EXPANDED TICKER UNIVERSE - High volume, liquid options
        self.symbols = [
            # BROAD MARKET ETFS
            'SPY', 'QQQ', 'IWM', 'DIA',
            # LEVERAGED ETFS (3x)
            'TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'TNA', 'TZA',
            # SECTOR ETFS
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU',
            # MEGA CAP STOCKS
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            # HIGH BETA STOCKS
            'AMD', 'NFLX', 'META', 'BABA', 'COIN',
            # HIGH-VOLUME SMALL CAPS (based on options volume research)
            'AMC', 'PLTR', 'SOFI', 'GME', 'UPST', 'BB',
            # POPULAR SMALL CAPS WITH OPTIONS ACTIVITY
            'RIOT', 'MARA', 'TLRY', 'SNAP', 'UBER', 'LYFT',
            'ROKU', 'HOOD', 'AFRM', 'SQ', 'SHOP', 'RIVN',
            # BIOTECH/GROWTH SMALL CAPS
            'MRNA', 'BIIB', 'GILD', 'VRTX'
        ]
        
        # Strategy parameters - Ultra high frequency tuning
        self.risk_per_trade = 0.008  # 0.8% risk per trade (smaller but more frequent)
        self.profit_target = 0.75    # 75% profit target (faster exits)
        self.stop_loss = 0.35        # 35% stop loss (looser to avoid whipsaws)
        self.min_iv = 0.12          # Even lower minimum IV
        self.max_iv = 0.85          # Higher maximum IV for volatile stocks
        self.min_dte = 1            # 1-7 days to expiration
        self.max_dte = 7
        
        # Trading frequency parameters
        self.trades_per_day_target = 15  # Higher target
        self.max_positions = 25      # Allow even more concurrent positions
        self.min_signal_strength = 0.15  # Lower minimum signal strength
        self.aggressive_entry_mode = True  # Enable more aggressive entry conditions
        
        # Options pricing parameters
        self.commission = 0.65      # Per contract
        self.bid_ask_spread = 0.015 # 1.5% spread (tighter for more volume)
        self.risk_free_rate = 0.045 # 4.5% risk-free rate
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Portfolio state
        self.positions = []
        self.trades = []
        self.current_time = None
        
        logger.info("ENHANCED HIGH-FREQUENCY options strategy initialized")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"Target: {self.trades_per_day_target} trades/day")
        logger.info(f"Symbols: {len(self.symbols)} tickers")
        logger.info(f"Max positions: {self.max_positions}")
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if T <= 0:
            return 1.0 if (option_type == 'call' and S > K) else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)
    
    def get_trading_signals(self, data: pd.DataFrame) -> tuple:
        """Multi-signal approach for higher frequency trading"""
        if len(data) < 20:
            return None, 0.0
        
        current_price = data['close'].iloc[-1]
        signals = []
        
        # 1. MOMENTUM SIGNALS (lowered thresholds)
        # Short-term momentum (1-day)
        mom_1d = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
        # Medium-term momentum (3-day) 
        mom_3d = (current_price - data['close'].iloc[-4]) / data['close'].iloc[-4]
        
        # Momentum signals with ultra-low thresholds for high frequency
        if mom_1d > 0.001:  # 0.1% move (ultra sensitive)
            signals.append(('call', 'momentum_1d', abs(mom_1d) * 3))
        elif mom_1d < -0.001:
            signals.append(('put', 'momentum_1d', abs(mom_1d) * 3))
        
        if mom_3d > 0.002:  # 0.2% move over 3 days (more sensitive)
            signals.append(('call', 'momentum_3d', abs(mom_3d) * 2))
        elif mom_3d < -0.002:
            signals.append(('put', 'momentum_3d', abs(mom_3d) * 2))
        
        # 2. MEAN REVERSION SIGNALS
        # Bollinger Band-style mean reversion
        close_20 = data['close'].iloc[-20:]
        bb_mean = close_20.mean()
        bb_std = close_20.std()
        
        if bb_std > 0:
            bb_position = (current_price - bb_mean) / (2 * bb_std)
            
            # Oversold - expect bounce (more sensitive)
            if bb_position < -1.2:  # Below 1.2 std devs (more sensitive)
                signals.append(('call', 'mean_reversion', abs(bb_position) * 1.5))
            # Overbought - expect pullback  
            elif bb_position > 1.2:  # Above 1.2 std devs (more sensitive)
                signals.append(('put', 'mean_reversion', abs(bb_position) * 1.5))
        
        # 3. VOLUME BREAKOUT SIGNALS
        avg_volume = data['volume'].iloc[-10:].mean()
        recent_volume = data['volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5:  # 50% volume increase (more sensitive)
            # Volume breakout in direction of price move
            if mom_1d > 0.0005:  # 0.05% move with volume (ultra sensitive)
                signals.append(('call', 'volume_breakout', volume_ratio * abs(mom_1d) * 2))
            elif mom_1d < -0.0005:
                signals.append(('put', 'volume_breakout', volume_ratio * abs(mom_1d) * 2))
        
        # 4. GAP SIGNALS
        prev_close = data['close'].iloc[-2]
        today_open = data['open'].iloc[-1]
        gap = (today_open - prev_close) / prev_close
        
        # Gap up/down signals (more sensitive)
        if abs(gap) > 0.005:  # 0.5% gap (more sensitive)
            if gap > 0:
                signals.append(('call', 'gap_up', abs(gap) * 4))
            else:
                signals.append(('put', 'gap_down', abs(gap) * 4))
        
        # 5. VOLATILITY EXPANSION
        returns = data['close'].pct_change().dropna()
        vol_10d = returns.iloc[-10:].std()
        vol_3d = returns.iloc[-3:].std()
        
        if vol_10d > 0 and vol_3d / vol_10d > 1.3:  # Volatility expanding (more sensitive)
            # Trade direction of recent move with vol expansion
            if mom_1d > 0.001:  # Lower threshold
                signals.append(('call', 'vol_expansion', (vol_3d / vol_10d) * abs(mom_1d) * 2))
            elif mom_1d < -0.001:
                signals.append(('put', 'vol_expansion', (vol_3d / vol_10d) * abs(mom_1d) * 2))
        
        # 6. ADDITIONAL HIGH-FREQUENCY SIGNALS
        # Small price oscillations for scalping
        if len(data) >= 5:
            price_5d = data['close'].iloc[-5:]
            price_range = price_5d.max() - price_5d.min()
            current_position = (current_price - price_5d.min()) / price_range if price_range > 0 else 0.5
            
            # Buy near lows, sell near highs
            if current_position < 0.3:  # Near 5-day low
                signals.append(('call', 'range_low', (0.3 - current_position) * 2))
            elif current_position > 0.7:  # Near 5-day high
                signals.append(('put', 'range_high', (current_position - 0.7) * 2))
        
        # Return strongest signal
        if not signals:
            return None, 0.0
        
        # Sort by signal strength and return strongest
        signals.sort(key=lambda x: x[2], reverse=True)
        best_signal = signals[0]
        
        return best_signal[0], min(best_signal[2], 3.0)  # Cap signal strength
    
    def generate_strike_and_iv(self, symbol: str, current_price: float, option_type: str) -> tuple:
        """Generate realistic strike price and implied volatility"""
        # Strike selection based on momentum strategy
        if option_type == 'call':
            # Slightly out of the money calls for leverage
            strike = current_price * np.random.choice([1.02, 1.03, 1.05, 1.08])
        else:
            # Slightly out of the money puts
            strike = current_price * np.random.choice([0.98, 0.97, 0.95, 0.92])
        
        # Round to reasonable strike intervals
        if current_price < 50:
            strike = round(strike * 2) / 2  # $0.50 intervals
        elif current_price < 100:
            strike = round(strike)  # $1 intervals
        else:
            strike = round(strike / 5) * 5  # $5 intervals
        
        # IV based on symbol type and market conditions
        if symbol in ['TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'TNA', 'TZA']:
            # Leveraged ETFs - higher IV
            iv = np.random.uniform(0.35, 0.60)
        elif symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            # Index ETFs - moderate IV
            iv = np.random.uniform(0.15, 0.30)
        elif symbol in ['TSLA', 'NVDA', 'AMD', 'COIN']:
            # High beta stocks - high IV
            iv = np.random.uniform(0.40, 0.70)
        else:
            # Standard stocks/ETFs
            iv = np.random.uniform(0.20, 0.40)
        
        return strike, iv
    
    def enter_position(self, symbol: str, signal: str, signal_strength: float, current_price: float) -> None:
        """Enter new options position"""
        if len(self.positions) >= self.max_positions:
            return
        
        # Position sizing based on signal strength and risk management
        base_risk_amount = self.capital * self.risk_per_trade
        adjusted_risk = base_risk_amount * min(signal_strength * 2, 2.0)  # Scale with signal strength
        
        if adjusted_risk < 50:  # Minimum position size
            return
        
        # Generate strike and IV
        strike, iv = self.generate_strike_and_iv(symbol, current_price, signal)
        
        # Random DTE
        dte = np.random.randint(self.min_dte, self.max_dte + 1)
        expiration = self.current_time + timedelta(days=int(dte))
        
        # Calculate option price
        T = dte / 365.0
        theoretical_price = self.black_scholes_price(
            current_price, strike, T, self.risk_free_rate, iv, signal
        )
        
        # Apply bid/ask spread (we pay ask when buying)
        entry_price = theoretical_price * (1 + self.bid_ask_spread)
        
        # Calculate position size
        contracts = max(1, int(adjusted_risk / (entry_price * 100)))
        total_cost = contracts * entry_price * 100 + contracts * self.commission
        
        if total_cost > self.capital:
            return
        
        # Calculate delta for risk management
        delta = self.calculate_delta(current_price, strike, T, self.risk_free_rate, iv, signal)
        
        # Create position
        position = {
            'id': len(self.positions) + 1,
            'symbol': symbol,
            'type': signal,
            'strike': strike,
            'expiration': expiration,
            'dte': dte,
            'contracts': contracts,
            'entry_price': entry_price,
            'entry_cost': total_cost,
            'underlying_entry': current_price,
            'iv': iv,
            'delta': delta,
            'entry_date': self.current_time,
            'signal_strength': signal_strength
        }
        
        self.positions.append(position)
        self.capital -= total_cost
        
        logger.info(f"ENTER {symbol} {signal.upper()}: {contracts} contracts @ ${entry_price:.2f}")
        logger.info(f"  Strike: ${strike:.1f}, DTE: {dte}, IV: {iv:.1%}")
        logger.info(f"  Total cost: ${total_cost:.0f}, Delta: {delta:.2f}")
    
    def update_and_exit_positions(self, current_prices: dict) -> None:
        """Update and potentially exit positions"""
        positions_to_remove = []
        
        for i, pos in enumerate(self.positions):
            if pos['symbol'] not in current_prices:
                continue
            
            current_price = current_prices[pos['symbol']]
            days_held = (self.current_time - pos['entry_date']).days
            dte_remaining = (pos['expiration'] - self.current_time).days
            
            # Calculate current option value
            T = max(dte_remaining / 365.0, 0)
            current_theoretical = self.black_scholes_price(
                current_price, pos['strike'], T, self.risk_free_rate, pos['iv'], pos['type']
            )
            
            # Apply bid/ask spread (we receive bid when selling)
            current_price_option = current_theoretical * (1 - self.bid_ask_spread)
            current_value = pos['contracts'] * current_price_option * 100
            
            # Calculate P&L
            total_cost = pos['contracts'] * self.commission  # Exit commission
            pnl = current_value - pos['entry_cost'] - total_cost
            pnl_pct = pnl / pos['entry_cost'] if pos['entry_cost'] > 0 else 0
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Profit target
            if pnl_pct >= self.profit_target:
                should_exit = True
                exit_reason = "profit_target"
            
            # Stop loss
            elif pnl_pct <= -self.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Expiration risk (1 day before expiration)
            elif dte_remaining <= 1:
                should_exit = True
                exit_reason = "expiration_risk"
            
            # Time decay management (exit if held 3+ days with small profit)
            elif days_held >= 3 and pnl_pct > 0.1:
                should_exit = True
                exit_reason = "time_decay"
            
            if should_exit:
                # Execute exit
                self.capital += current_value - total_cost
                
                # Record trade
                trade = {
                    'entry_date': pos['entry_date'],
                    'exit_date': self.current_time,
                    'symbol': pos['symbol'],
                    'type': pos['type'],
                    'contracts': pos['contracts'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price_option,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'strike': pos['strike'],
                    'signal_strength': pos['signal_strength']
                }
                self.trades.append(trade)
                
                logger.info(f"EXIT {pos['symbol']} {pos['type'].upper()}: ${current_value:.0f} - {exit_reason}")
                logger.info(f"  P&L: ${pnl:+.0f} ({pnl_pct:+.1%}) in {days_held} days")
                logger.info(f"  New capital: ${self.capital:,.0f}")
                
                positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
    
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
            
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> None:
        """Run the enhanced high-frequency backtest"""
        
        logger.info("============================================================")
        logger.info("ENHANCED HIGH-FREQUENCY OPTIONS STRATEGY BACKTEST")
        logger.info("============================================================")
        logger.info("Fetching historical data...")
        
        # Get historical data for all symbols
        data = {}
        successful_symbols = []
        for symbol in self.symbols:
            symbol_data = self.get_historical_data(symbol, start_date, end_date)
            if not symbol_data.empty:
                data[symbol] = symbol_data
                successful_symbols.append(symbol)
        
        logger.info(f"Successfully loaded data for {len(successful_symbols)}/{len(self.symbols)} symbols")
        
        if not data:
            logger.error("No data loaded!")
            return
        
        # Get trading dates from the first symbol with data
        first_symbol = list(data.keys())[0]
        trading_dates = data[first_symbol].index[10:]  # Skip first 10 days for momentum calc
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        # Main backtest loop
        daily_trades = 0
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            daily_trades = 0
            
            # Get current prices
            current_prices = {}
            for symbol in successful_symbols:
                if symbol in data and current_date in data[symbol].index:
                    current_prices[symbol] = data[symbol].loc[current_date, 'close']
            
            if not current_prices:
                continue
            
            # Update and exit existing positions first
            self.update_and_exit_positions(current_prices)
            
            # Look for new trading opportunities
            for symbol in successful_symbols:
                if symbol not in current_prices:
                    continue
                
                if daily_trades >= self.trades_per_day_target:
                    break
                
                # Get trading signals (updated function name)
                signal, strength = self.get_trading_signals(data[symbol].loc[:current_date])
                
                if signal and strength >= self.min_signal_strength:  # Lower threshold
                    self.enter_position(symbol, signal, strength, current_prices[symbol])
                    daily_trades += 1
            
            # Log progress
            if i % 20 == 0 or i == len(trading_dates) - 1:  # Every 20 days or last day
                portfolio_value = self.capital + sum(pos['entry_cost'] for pos in self.positions)
                total_return = (portfolio_value / self.initial_capital - 1) * 100
                
                logger.info(f"{current_date.date()}: Capital=${self.capital:,.0f} (+{total_return:.1f}%), Positions={len(self.positions)}, Trades={len(self.trades)}")
        
        self.print_results()
    
    def print_results(self) -> None:
        """Print enhanced backtest results"""
        final_portfolio_value = self.capital + sum(pos['entry_cost'] for pos in self.positions)
        
        # Calculate returns
        total_return = (final_portfolio_value / self.initial_capital - 1) * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) * 100 if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) * 100 if losing_trades else 0
        
        # Trading frequency
        if self.trades:
            first_trade_date = min(t['entry_date'] for t in self.trades)
            last_trade_date = max(t['exit_date'] for t in self.trades)
            trading_days = (last_trade_date - first_trade_date).days
            trades_per_day = len(self.trades) / trading_days if trading_days > 0 else 0
        else:
            trades_per_day = 0
        
        logger.info("============================================================")
        logger.info("ENHANCED HIGH-FREQUENCY OPTIONS STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_portfolio_value:,.0f}")
        logger.info(f"Total Return: {total_return:+.1f}%")
        logger.info(f"Available Cash: ${self.capital:,.0f}")
        logger.info(f"Open Positions: {len(self.positions)}")
        logger.info("")
        logger.info("TRADING STATISTICS:")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Trades per Day: {trades_per_day:.1f}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Win: {avg_win:+.1f}%")
        logger.info(f"Average Loss: {avg_loss:+.1f}%")
        logger.info("============================================================")
        
        # Symbol performance
        symbol_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'pnl': 0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade['pnl']
        
        logger.info("TOP PERFORMING SYMBOLS:")
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]
        for symbol, stats in sorted_symbols:
            logger.info(f"{symbol}: {stats['trades']} trades, ${stats['pnl']:+,.0f} P&L")

if __name__ == "__main__":
    # Run the enhanced high-frequency backtest
    backtest = EnhancedHighFrequencyBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)