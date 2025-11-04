"""
Identify which Financials stocks had extreme volatility changes between 0922 and 1222
This explains why total variance jumped from 0.56 to 3.53 (6.2x increase)
"""

import pandas as pd
import numpy as np
import sys

print("=" * 80)
print("IDENTIFYING STOCKS THAT CAUSED THE VOLATILITY EXPLOSION")
print("=" * 80)

periods = ['0922', '1222']
stock_volatilities = {}
returns_matrices = {}

for period in periods:
    returns_file = f"Data/log_returns/sector_log_returns_comp_{period}.xlsx"

    try:
        # Read Financials returns
        log_returns = pd.read_excel(returns_file, sheet_name='Financials')

        # Drop Date column and clean
        if 'Date' in log_returns.columns:
            dates = log_returns['Date']
            returns_clean = log_returns.drop(columns=['Date']).dropna()
        else:
            returns_clean = log_returns.dropna()

        returns_matrices[period] = returns_clean

        # Calculate volatility for each stock
        stock_vols = returns_clean.std()
        stock_volatilities[period] = stock_vols.sort_values(ascending=False)

        print(f"\n{'='*80}")
        print(f"PERIOD {period}")
        print(f"{'='*80}")

        print(f"\nData: {returns_clean.shape[0]} observations × {returns_clean.shape[1]} stocks")
        print(f"\nVolatility statistics:")
        print(f"  Mean: {stock_vols.mean():.6f}")
        print(f"  Median: {stock_vols.median():.6f}")
        print(f"  Std: {stock_vols.std():.6f}")
        print(f"  Min: {stock_vols.min():.6f}")
        print(f"  Max: {stock_vols.max():.6f}")
        print(f"  95th percentile: {stock_vols.quantile(0.95):.6f}")

        # Show top 10 most volatile stocks
        print(f"\n  Top 10 most volatile stocks:")
        for i, (stock, vol) in enumerate(stock_vols.nlargest(10).items(), 1):
            print(f"    {i:2d}. {stock:15s} σ = {vol:.6f}")

        # Count extreme volatilities
        high_vol_count = (stock_vols > 0.10).sum()
        very_high_vol_count = (stock_vols > 0.20).sum()
        print(f"\n  Stocks with σ > 0.10: {high_vol_count}/{len(stock_vols)}")
        print(f"  Stocks with σ > 0.20: {very_high_vol_count}/{len(stock_vols)}")

    except Exception as e:
        print(f"\nPeriod {period}: Error - {e}")
        import traceback
        traceback.print_exc()

