#!/usr/bin/env python3
"""
PURE CAPITAL STRATEGY - No External Income
Work ONLY with initial $10,000 to generate maximum returns through trading strategy alone.
Focus on buy-and-hold with intelligent dip buying using only available capital.
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

class PureCapitalBacktest:
    """Pure Capital Strategy - ONLY use initial $10,000, no external income"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Quality growth stocks and ETFs
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'QQQ']
        
        # Position allocation (equal weight initially)
        self.target_allocation = {
            'AAPL': 0.25,   # 25% allocation
            'MSFT': 0.25,   # 25% allocation  
            'GOOGL': 0.25,  # 25% allocation
            'QQQ': 0.25     # 25% allocation
        }
        
        # Strategy parameters
        self.initial_buy_pct = 0.80  # Use 80% of capital initially, keep 20% for dip buying
        self.dip_threshold = -0.08   # Buy more on 8% dips
        self.major_dip_threshold = -0.15  # Major dip buying opportunity
        
        # Initialize data client
        self.data_client = AlpacaDataClient()
        
        # Portfolio state
        self.positions = {symbol: {'shares': 0.0, 'cost_basis': 0.0, 'last_price': 0.0} 
                         for symbol in self.symbols}
        self.trades = []
        self.current_time = None
        
        logger.info("PURE CAPITAL strategy initialized - NO external income")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"Symbols: {self.symbols}")
    
    def calculate_portfolio_value(self, current_prices: dict) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.capital  # Available cash
        
        for symbol in self.symbols:
            if symbol in current_prices:
                self.positions[symbol]['last_price'] = current_prices[symbol]
                position_value = self.positions[symbol]['shares'] * current_prices[symbol]
                portfolio_value += position_value
        
        return portfolio_value
    
    def buy_shares(self, symbol: str, dollar_amount: float, price: float, reason: str) -> None:
        """Buy shares and update position"""
        if dollar_amount <= 0 or price <= 0 or dollar_amount > self.capital:
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
        logger.info(f"  New position: {new_shares:.3f} shares, avg cost: ${new_cost_basis:.2f}, Cash left: ${self.capital:.0f}")
    
    def initial_purchase(self, current_prices: dict) -> None:
        """Make initial purchases based on target allocation"""
        initial_investment = self.capital * self.initial_buy_pct
        
        for symbol in self.symbols:
            if symbol in current_prices:
                target_amount = initial_investment * self.target_allocation[symbol]
                self.buy_shares(symbol, target_amount, current_prices[symbol], "INITIAL")
    
    def check_dip_buying_opportunity(self, symbol: str, data: pd.DataFrame, current_price: float) -> tuple:
        """Check if we should buy on a dip and return (should_buy, amount)"""
        if len(data) < 20 or self.capital < 100:  # Need cash and history
            return False, 0
        
        # Check against recent highs
        high_20 = data['high'].rolling(20).max().iloc[-1]
        high_60 = data['high'].rolling(60).max().iloc[-1] if len(data) >= 60 else high_20
        
        dip_from_20_high = (current_price - high_20) / high_20
        dip_from_60_high = (current_price - high_60) / high_60
        
        # Check against our cost basis
        cost_basis = self.positions[symbol]['cost_basis']
        dip_from_cost = 0
        if cost_basis > 0:
            dip_from_cost = (current_price - cost_basis) / cost_basis
        
        # Determine buy amount based on dip severity
        available_cash = self.capital
        
        # Major dip (15%+ down) - use more cash
        if dip_from_60_high <= self.major_dip_threshold or dip_from_cost <= self.major_dip_threshold:
            buy_amount = min(available_cash * 0.30, available_cash)  # Use up to 30% of remaining cash
            return True, buy_amount
        
        # Regular dip (8%+ down) - moderate buying
        elif dip_from_20_high <= self.dip_threshold or dip_from_cost <= self.dip_threshold:
            buy_amount = min(available_cash * 0.15, available_cash)  # Use up to 15% of remaining cash
            return True, buy_amount
        
        return False, 0
    
    def dip_buying(self, current_prices: dict, all_data: dict) -> None:
        """Execute dip buying strategy with available capital only"""
        if self.capital < 100:  # Need at least $100 to make purchases
            return
        
        # Find symbols with dip buying opportunities
        dip_opportunities = []
        for symbol in self.symbols:
            if symbol in current_prices and symbol in all_data:
                should_buy, amount = self.check_dip_buying_opportunity(symbol, all_data[symbol], current_prices[symbol])
                if should_buy and amount > 50:  # Minimum $50 purchase
                    dip_opportunities.append((symbol, amount))
        
        # Execute dip buys in order of opportunity size
        dip_opportunities.sort(key=lambda x: x[1], reverse=True)  # Largest opportunities first
        
        for symbol, amount in dip_opportunities:
            if self.capital >= amount:
                self.buy_shares(symbol, amount, current_prices[symbol], "DIP_BUY")
    
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
        """Run the pure capital backtest"""
        
        logger.info("============================================================")
        logger.info("PURE CAPITAL STRATEGY BACKTEST - NO EXTERNAL INCOME")
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
            
            # Dip buying opportunities (only with remaining capital)
            self.dip_buying(current_prices, data)
            
            # Log progress
            if i % 60 == 0 or i == len(trading_dates) - 1:  # Every 60 days or last day
                portfolio_value = self.calculate_portfolio_value(current_prices)
                total_return = (portfolio_value / self.initial_capital - 1) * 100
                
                logger.info(f"{current_date.date()}: Portfolio=${portfolio_value:,.0f} (+{total_return:.1f}% total return), Cash: ${self.capital:.0f}")
        
        self.print_results(current_prices)
    
    def print_results(self, final_prices: dict) -> None:
        """Print pure capital backtest results"""
        final_portfolio_value = self.calculate_portfolio_value(final_prices)
        
        # Calculate returns (only against initial capital)
        total_return = (final_portfolio_value / self.initial_capital - 1) * 100
        
        # Position breakdown
        logger.info("============================================================")
        logger.info("PURE CAPITAL STRATEGY RESULTS - NO EXTERNAL INCOME")
        logger.info("============================================================")
        logger.info(f"Period: 2024-01-01 to 2025-08-20")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_portfolio_value:,.0f}")
        logger.info(f"Total Return: {total_return:+.1f}%")
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
        annualized_return = ((final_portfolio_value / self.initial_capital) ** (1/years) - 1) * 100
        
        logger.info(f"Annualized Return: {annualized_return:.1f}%")
        logger.info("============================================================")

if __name__ == "__main__":
    # Run the pure capital backtest
    backtest = PureCapitalBacktest(initial_capital=10000.0)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 8, 20)
    
    backtest.run_backtest(start_date, end_date)