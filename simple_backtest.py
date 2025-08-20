#!/usr/bin/env python3
"""
Simple backtest runner that avoids complex import dependencies.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


@dataclass
class SimpleBacktestConfig:
    """Simple backtest configuration."""
    start_date: str = "2023-01-01"
    end_date: str = "2023-06-30"
    initial_capital: float = 10000.0
    symbols: List[str] = None
    position_size_pct: float = 10.0  # 10% per position


class SimpleStrategy:
    """Simple buy-and-hold strategy for testing."""
    
    def __init__(self, name: str = "simple_strategy"):
        self.name = name
        
    def generate_signals(self, data: pd.DataFrame, current_positions: dict) -> List[dict]:
        """Generate simple buy signals."""
        signals = []
        
        if len(current_positions) == 0 and len(data) > 20:
            # Simple momentum: buy if price is above 20-day moving average
            data['ma20'] = data['Close'].rolling(20).mean()
            
            if data['Close'].iloc[-1] > data['ma20'].iloc[-1]:
                signals.append({
                    'action': 'buy',
                    'symbol': 'AAPL',  # Hard-coded for simplicity
                    'confidence': 75.0
                })
        
        return signals


class SimpleBroker:
    """Simple broker simulation."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        
    def buy(self, symbol: str, quantity: float, price: float, date: datetime):
        """Buy shares."""
        cost = quantity * price
        if cost <= self.cash:
            self.cash -= cost
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'entry_date': date
            }
            
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': price,
                'value': cost
            })
            return True
        return False
    
    def sell(self, symbol: str, price: float, date: datetime):
        """Sell all shares of a symbol."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            proceeds = pos['quantity'] * price
            self.cash += proceeds
            
            pnl = proceeds - (pos['quantity'] * pos['avg_price'])
            
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'sell',
                'quantity': pos['quantity'],
                'price': price,
                'value': proceeds,
                'pnl': pnl
            })
            
            del self.positions[symbol]
            return True
        return False
    
    def get_portfolio_value(self, prices: dict) -> float:
        """Get total portfolio value."""
        position_value = 0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                position_value += pos['quantity'] * prices[symbol]
        
        return self.cash + position_value


def download_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Download historical data."""
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                data[symbol] = df
                print(f"Downloaded {len(df)} days of data for {symbol}")
            else:
                print(f"No data found for {symbol}")
                
        except Exception as e:
            print(f"Failed to download {symbol}: {e}")
    
    return data


def run_simple_backtest(config: SimpleBacktestConfig) -> dict:
    """Run a simple backtest."""
    print(f"Running simple backtest from {config.start_date} to {config.end_date}")
    
    # Default symbols if none provided
    symbols = config.symbols or ['AAPL', 'MSFT']
    
    # Download data
    print("Downloading market data...")
    market_data = download_data(symbols, config.start_date, config.end_date)
    
    if not market_data:
        print("No market data available!")
        return {}
    
    # Initialize broker and strategy
    broker = SimpleBroker(config.initial_capital)
    strategy = SimpleStrategy()
    
    # Get all trading dates from the first symbol
    first_symbol = list(market_data.keys())[0]
    trading_dates = market_data[first_symbol].index
    
    equity_curve = []
    
    print(f"Running simulation over {len(trading_dates)} trading days...")
    
    for i, date in enumerate(trading_dates):
        # Get current prices
        current_prices = {}
        for symbol, data in market_data.items():
            if date in data.index:
                current_prices[symbol] = data.loc[date, 'Close']
        
        # Generate signals (using AAPL data for simplicity)
        if 'AAPL' in market_data and date in market_data['AAPL'].index:
            recent_data = market_data['AAPL'].loc[:date]
            
            if len(recent_data) >= 20:  # Enough data for moving average
                signals = strategy.generate_signals(recent_data, broker.positions)
                
                # Process signals
                for signal in signals:
                    if signal['action'] == 'buy' and signal['symbol'] in current_prices:
                        symbol = signal['symbol']
                        price = current_prices[symbol]
                        
                        # Calculate position size
                        position_value = config.initial_capital * (config.position_size_pct / 100)
                        quantity = position_value / price
                        
                        # Execute trade
                        if broker.buy(symbol, quantity, price, date):
                            print(f"{date.strftime('%Y-%m-%d')}: Bought {quantity:.2f} shares of {symbol} at ${price:.2f}")
        
        # Record portfolio value
        portfolio_value = broker.get_portfolio_value(current_prices)
        equity_curve.append({
            'date': date,
            'value': portfolio_value,
            'cash': broker.cash
        })
        
        # Simple exit: sell if holding for 30 days or more
        positions_to_close = []
        for symbol, pos in broker.positions.items():
            days_held = (date - pos['entry_date']).days
            if days_held >= 30 and symbol in current_prices:
                positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            price = current_prices[symbol]
            if broker.sell(symbol, price, date):
                print(f"{date.strftime('%Y-%m-%d')}: Sold {symbol} at ${price:.2f}")
    
    # Calculate results
    final_value = equity_curve[-1]['value']
    total_return = (final_value / config.initial_capital - 1) * 100
    
    # Create equity curve DataFrame
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('date', inplace=True)
    
    # Calculate volatility and other metrics
    returns = equity_df['value'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
    
    # Max drawdown
    rolling_max = equity_df['value'].expanding().max()
    drawdown = (equity_df['value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assume 2% risk-free rate)
    excess_returns = returns - (0.02 / 252)  # Daily risk-free rate
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    results = {
        'initial_capital': config.initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'volatility_pct': volatility,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(broker.trades),
        'equity_curve': equity_df,
        'trades': broker.trades,
        'final_positions': broker.positions
    }
    
    return results


def print_results(results: dict):
    """Print backtest results."""
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Volatility: {results['volatility_pct']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Total Trades: {results['total_trades']}")
    
    if results['trades']:
        print("\nTrade History:")
        for trade in results['trades']:
            pnl_str = f", P&L: ${trade['pnl']:.2f}" if 'pnl' in trade else ""
            print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action'].upper()} {trade['quantity']:.2f} {trade['symbol']} @ ${trade['price']:.2f}{pnl_str}")
    
    if results['final_positions']:
        print("\nFinal Positions:")
        for symbol, pos in results['final_positions'].items():
            print(f"  {symbol}: {pos['quantity']:.2f} shares @ ${pos['avg_price']:.2f}")


def main():
    """Main entry point."""
    print("Simple Backtest Runner")
    print("Testing basic backtesting framework...")
    
    # Create configuration
    config = SimpleBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-06-30",
        initial_capital=10000.0,
        symbols=['AAPL', 'MSFT'],
        position_size_pct=20.0  # 20% per position
    )
    
    # Run backtest
    try:
        results = run_simple_backtest(config)
        
        if results:
            print_results(results)
            
            # Save results
            output_dir = Path("simple_backtest_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save equity curve
            equity_file = output_dir / "equity_curve.csv"
            results['equity_curve'].to_csv(equity_file)
            print(f"\nEquity curve saved to: {equity_file}")
            
            print(f"\nBacktest completed successfully!")
        else:
            print("Backtest failed - no results generated")
            
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()