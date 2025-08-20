#!/usr/bin/env python3
"""
Debug the data structure to understand the date formatting issue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.alpaca_client import AlpacaDataClient
from alpaca.data.timeframe import TimeFrame

def debug_data_structure():
    """Debug the data structure returned by Alpaca"""
    
    print("Debugging Data Structure Issues")
    print("=" * 40)
    
    data_client = AlpacaDataClient()
    
    # Get a small sample of data
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 8, 10)
    
    try:
        df = data_client.get_stock_bars(
            symbols=['AAPL'],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame type: {type(df)}")
        print(f"Index type: {type(df.index)}")
        print(f"Index name: {df.index.name}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\nIndex details:")
        for i, idx in enumerate(df.index[:3]):
            print(f"  Index {i}: {idx} (type: {type(idx)})")
            
        print(f"\nIterating through DataFrame:")
        for i, (date, row) in enumerate(df.iterrows()):
            print(f"  Row {i}: date={date} (type: {type(date)}), close=${row['close']:.2f}")
            if i >= 2:
                break
                
        # Test date formatting
        print(f"\nTesting date formatting:")
        for i, idx in enumerate(df.index[:3]):
            try:
                if hasattr(idx, 'strftime'):
                    formatted = idx.strftime('%Y-%m-%d')
                    print(f"  Success: {formatted}")
                else:
                    print(f"  No strftime method: {idx} (type: {type(idx)})")
            except Exception as e:
                print(f"  Error formatting {idx}: {e}")
            
            if i >= 2:
                break
                
    except Exception as e:
        print(f"Error getting data: {e}")

if __name__ == "__main__":
    debug_data_structure()