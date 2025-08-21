#!/usr/bin/env python3
"""
COMPOUND GROWTH STRATEGY - Target 20%+ Annual Returns
Revolutionary approach focusing on steady compounding:

1. Buy and hold strong trending stocks (AAPL, MSFT, GOOGL, QQQ)
2. Scale in during dips (dollar cost averaging on steroids)
3. No stops - only add to winners on pullbacks
4. Compound with reinvestment and regular additions
5. Focus on TIME IN MARKET not timing the market
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os
from pathlib import Path
import logging

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

class CompoundGrowthBacktest:
    """Compound Growth Strategy - Steady accumulation for consistent returns"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # FOCUS: Quality growth stocks and ETFs
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'QQQ']
        
        # Position allocation (equal weight initially)
        self.target_allocation = {
            'AAPL': 0.25,   # 25% allocation
            'MSFT': 0.25,   # 25% allocation  
            'GOOGL': 0.25,  # 25% allocation
            'QQQ': 0.25     # 25% allocation
        }
        
        # Buy parameters
        self.initial_buy_pct = 0.80  # Use 80% of capital for initial positions
        self.dip_buy_threshold = -0.05  # Buy more on 5% dips
        self.max_dip_buy_pct = 0.10   # Use up to 10% of capital for dip buys
        self.rebalance_threshold = 0.15  # Rebalance when allocation off by 15%
        
        # Compound growth features
        self.monthly_addition_amount = 200  # Add $200 monthly (simulating income)
        self.reinvest_gains = True   # Reinvest all gains
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Portfolio state
        self.positions = {symbol: {'shares': 0.0, 'cost_basis': 0.0, 'last_price': 0.0} 
                         for symbol in self.symbols}
        self.trades = []
        self.monthly_purchases = []
        self.current_time = None
        self.last_rebalance = None
        self.last_monthly_addition = None
        
        logger.info("COMPOUND GROWTH strategy initialized")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Target allocation: {self.target_allocation}")
        logger.info(f"Monthly addition: ${self.monthly_addition_amount}")
    
    def calculate_portfolio_value(self, current_prices: dict) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.capital  # Available cash
        
        for symbol in self.symbols:
            if symbol in current_prices:
                self.positions[symbol]['last_price'] = current_prices[symbol]
                position_value = self.positions[symbol]['shares'] * current_prices[symbol]
                portfolio_value += position_value
        
        return portfolio_value
    
    def get_current_allocation(self, current_prices: dict) -> dict:
        """Get current allocation percentages"""
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        if portfolio_value <= 0:
            return {symbol: 0.0 for symbol in self.symbols}
        
        allocation = {}
        for symbol in self.symbols:
            if symbol in current_prices:
                position_value = self.positions[symbol]['shares'] * current_prices[symbol]
                allocation[symbol] = position_value / portfolio_value
            else:
                allocation[symbol] = 0.0
        
        return allocation
    
    def buy_shares(self, symbol: str, dollar_amount: float, price: float, reason: str) -> None:
        """Buy shares and update position"""
        if dollar_amount <= 0 or price <= 0:
            return
        
        shares_to_buy = dollar_amount / price
        
        if shares_to_buy < 0.001:  # Minimum position size
            return
        
        # Update position
        old_shares = self.positions[symbol]['shares']
        old_cost = self.positions[symbol]['cost_basis'] * old_shares
        
        new_shares = old_shares + shares_to_buy
        new_cost = old_cost + dollar_amount
        new_cost_basis = new_cost / new_shares if new_shares > 0 else 0
        
        self.positions[symbol]['shares'] = new_shares
        self.positions[symbol]['cost_basis'] = new_cost_basis
        self.positions[symbol]['last_price'] = price
        
        # Update capital
        self.capital -= dollar_amount
        
        # Record trade
        trade = {
            'date': self.current_time,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares_to_buy,
            'price': price,
            'amount': dollar_amount,
            'reason': reason
        }
        self.trades.append(trade)
        
        logger.info(f"BUY {symbol}: {shares_to_buy:.3f} shares @ ${price:.2f} = ${dollar_amount:.0f} ({reason})")
        logger.info(f"  New position: {new_shares:.3f} shares, avg cost: ${new_cost_basis:.2f}")
    
    def initial_purchase(self, current_prices: dict) -> None:
        """Make initial purchases based on target allocation"""
        initial_investment = self.capital * self.initial_buy_pct
        
        for symbol in self.symbols:
            if symbol in current_prices:
                target_amount = initial_investment * self.target_allocation[symbol]
                self.buy_shares(symbol, target_amount, current_prices[symbol], "INITIAL")
    
    def check_dip_buying_opportunity(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check if we should buy on a dip"""
        if len(data) < 20:
            return False
        
        current_price = data['close'].iloc[-1]
        
        # Check against 20-day high
        high_20 = data['high'].rolling(20).max().iloc[-1]
        dip_from_high = (current_price - high_20) / high_20
        
        # Check against our cost basis
        cost_basis = self.positions[symbol]['cost_basis']
        if cost_basis > 0:
            dip_from_cost = (current_price - cost_basis) / cost_basis
            
            # Buy if down from recent high OR down from our cost basis
            if dip_from_high <= self.dip_buy_threshold or dip_from_cost <= self.dip_buy_threshold:
                return True
        
        return False
    
    def dip_buying(self, current_prices: dict, all_data: dict) -> None:
        """Execute dip buying strategy"""
        available_for_dips = self.capital * self.max_dip_buy_pct
        
        if available_for_dips < 50:  # Need at least $50 to make a purchase
            return
        
        # Find symbols that are dipping
        dip_symbols = []
        for symbol in self.symbols:
            if symbol in current_prices and symbol in all_data:
                if self.check_dip_buying_opportunity(symbol, all_data[symbol]):
                    dip_symbols.append(symbol)
        
        if not dip_symbols:
            return
        
        # Divide available money among dipping symbols
        amount_per_symbol = available_for_dips / len(dip_symbols)
        
        for symbol in dip_symbols:
            self.buy_shares(symbol, amount_per_symbol, current_prices[symbol], "DIP_BUY")
    
    def monthly_addition(self) -> None:
        """Add monthly investment (simulating regular income)"""
        if self.last_monthly_addition is None:
            self.last_monthly_addition = self.current_time
        
        # Check if a month has passed
        days_since_addition = (self.current_time - self.last_monthly_addition).days
        
        if days_since_addition >= 30:  # Monthly addition
            self.capital += self.monthly_addition_amount
            self.last_monthly_addition = self.current_time
            
            purchase = {
                'date': self.current_time,
                'amount': self.monthly_addition_amount,
                'reason': 'MONTHLY_ADDITION'
            }
            self.monthly_purchases.append(purchase)
            
            logger.info(f"MONTHLY ADDITION: +${self.monthly_addition_amount} (Total cash: ${self.capital:.0f})")
    
    def rebalance_portfolio(self, current_prices: dict) -> None:
        """Rebalance portfolio to target allocation"""
        if self.last_rebalance is None:
            self.last_rebalance = self.current_time
            return
        
        # Check if enough time has passed (quarterly rebalancing)
        days_since_rebalance = (self.current_time - self.last_rebalance).days
        if days_since_rebalance < 90:  # Quarterly
            return
        
        current_allocation = self.get_current_allocation(current_prices)
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        # Check if rebalancing is needed
        needs_rebalancing = False
        for symbol in self.symbols:
            allocation_diff = abs(current_allocation[symbol] - self.target_allocation[symbol])
            if allocation_diff > self.rebalance_threshold:
                needs_rebalancing = True
                break
        
        if not needs_rebalancing:
            return
        
        logger.info("REBALANCING PORTFOLIO:")
        logger.info(f"Current allocation: {current_allocation}")
        
        # Calculate target dollar amounts
        for symbol in self.symbols:
            if symbol not in current_prices:
                continue
            
            current_value = self.positions[symbol]['shares'] * current_prices[symbol]
            target_value = portfolio_value * self.target_allocation[symbol]
            difference = target_value - current_value
            
            if abs(difference) > 100:  # Only rebalance if difference > $100
                if difference > 0:  # Need to buy more
                    buy_amount = min(difference, self.capital)
                    if buy_amount > 0:
                        self.buy_shares(symbol, buy_amount, current_prices[symbol], "REBALANCE")
                
                # Note: We don't sell in this strategy, only buy
        
        self.last_rebalance = self.current_time
    
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
        """Run the compound growth backtest"""
        
        logger.info("============================================================")
        logger.info("COMPOUND GROWTH STRATEGY BACKTEST")
        logger.info("============================================================")
        logger.info("Fetching historical data...")
        
        # Get historical data for all symbols
        data = {}
        for symbol in self.symbols:
            symbol_data = self.get_historical_data(symbol, start_date, end_date)
            if not symbol_data.empty:
                data[symbol] = symbol_data
        
        if not data:
            logger.error("No data loaded!")
            return
        
        # Get trading dates from the first symbol with data
        first_symbol = list(data.keys())[0]
        trading_dates = data[first_symbol].index[1:]  # Skip first day
        
        logger.info(f"Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        initial_purchase_made = False
        
        # Main backtest loop
        for i, current_date in enumerate(trading_dates):
            self.current_time = current_date
            
            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                if symbol in data and current_date in data[symbol].index:
                    current_prices[symbol] = data[symbol].loc[current_date, 'close']
            
            if not current_prices:
                continue
            
            # Make initial purchase on first day
            if not initial_purchase_made and len(current_prices) == len(self.symbols):
                self.initial_purchase(current_prices)
                initial_purchase_made = True
            
            # Monthly addition (simulating regular investing)
            self.monthly_addition()
            
            # Dip buying opportunities
            if self.capital > 100:  # Only if we have cash available
                self.dip_buying(current_prices, data)
            
            # Rebalancing (quarterly)
            if self.capital > 200:  # Only rebalance if we have some cash
                self.rebalance_portfolio(current_prices)
            
            # Log progress
            if i % 60 == 0 or i == len(trading_dates) - 1:  # Every 60 days or last day
                portfolio_value = self.calculate_portfolio_value(current_prices)
                total_return = (portfolio_value / self.initial_capital - 1) * 100
                
                # Calculate total invested (initial + monthly additions)
                total_invested = self.initial_capital + len(self.monthly_purchases) * self.monthly_addition_amount
                investment_return = (portfolio_value / total_invested - 1) * 100
                
                logger.info(f"{current_date.date()}: Portfolio=${portfolio_value:,.0f} (+{total_return:.1f}% vs initial, +{investment_return:.1f}% vs invested)")
        
        self.print_results(current_prices)
    
    def print_results(self, final_prices: dict) -> None:
        """Print compound growth backtest results"""
        final_portfolio_value = self.calculate_portfolio_value(final_prices)
        
        # Calculate returns
        total_return_vs_initial = (final_portfolio_value / self.initial_capital - 1) * 100
        
        # Total invested (including monthly additions)
        total_invested = self.initial_capital + len(self.monthly_purchases) * self.monthly_addition_amount
        return_vs_invested = (final_portfolio_value / total_invested - 1) * 100
        
        # Position breakdown
        logger.info("============================================================")
        logger.info("COMPOUND GROWTH STRATEGY RESULTS")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Monthly Additions: {len(self.monthly_purchases)} × ${self.monthly_addition_amount} = ${len(self.monthly_purchases) * self.monthly_addition_amount:,.0f}")
        logger.info(f"Total Invested: ${total_invested:,.0f}")
        logger.info(f"Final Value: ${final_portfolio_value:,.0f}")
        logger.info(f"Total Return vs Initial: {total_return_vs_initial:+.1f}%")
        logger.info(f"Total Return vs Invested: {return_vs_invested:+.1f}%")
        logger.info(f"Available Cash: ${self.capital:,.0f}")
        logger.info("")
        
        # Position breakdown
        logger.info("POSITION BREAKDOWN:")
        total_equity_value = 0
        for symbol in self.symbols:
            if symbol in final_prices:
                shares = self.positions[symbol]['shares']
                cost_basis = self.positions[symbol]['cost_basis']
                current_price = final_prices[symbol]
                position_value = shares * current_price
                total_cost = shares * cost_basis if cost_basis > 0 else 0
                position_return = ((position_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                logger.info(f"{symbol}: {shares:.1f} shares @ ${current_price:.2f} = ${position_value:,.0f} (Cost: ${cost_basis:.2f}, Return: {position_return:+.1f}%)")
                total_equity_value += position_value
        
        logger.info(f"Total Equity Value: ${total_equity_value:,.0f}")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info("============================================================")
        
        # Calculate annualized return
        days_elapsed = (self.current_time - self.trades[0]['date']).days if self.trades else 365
        years = days_elapsed / 365.25
        annualized_return = ((final_portfolio_value / total_invested) ** (1/years) - 1) * 100
        
        logger.info(f"Annualized Return: {annualized_return:.1f}%")
        logger.info("============================================================")

if __name__ == "__main__":
    # Run the compound growth backtest
    backtest = CompoundGrowthBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)