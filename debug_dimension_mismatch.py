"""Debug dimension mismatch for Consumer Staples 1221"""

import pandas as pd
import numpy as np

period = "1221"
sector_name = "Consumer Staples"

# Load benchmark weights
benchmark_file = f"data/datasets/benchmark_weights_carbon_intensity_{period}.xlsx"
data = pd.read_excel(benchmark_file)

# Get Consumer Staples sector data
sector = data[data['GICS Sector'] == sector_name].copy()

print(f"Period: {period}")
print(f"Sector: {sector_name}")
print(f"\nBenchmark weights:")
print(f"  Total stocks in sector: {len(sector)}")
print(f"  Stocks: {list(sector['SYMBOL'].values)}")

# Load log returns
returns_file = f"data/log_returns/sector_log_returns_comp_{period}_new.xlsx"
log_returns_all = pd.read_excel(returns_file, sheet_name=None)

# Get sector returns
R = log_returns_all[sector_name]
R_clean = R.drop(columns=['Date']).dropna()

print(f"\nLog returns:")
print(f"  Total stocks with returns: {len(R_clean.columns)}")
print(f"  Stocks: {list(R_clean.columns)}")

# Find mismatches
benchmark_symbols = set(sector['SYMBOL'].values)
returns_symbols = set(R_clean.columns)

missing_in_returns = benchmark_symbols - returns_symbols
missing_in_benchmark = returns_symbols - benchmark_symbols

print(f"\nDimension check:")
print(f"  w_bench length: {len(sector)}")
print(f"  Sigma dimensions: {len(R_clean.columns)} x {len(R_clean.columns)}")
print(f"  MISMATCH: {len(sector) != len(R_clean.columns)}")

if missing_in_returns:
    print(f"\nStocks in benchmark but NOT in returns ({len(missing_in_returns)}):")
    for sym in sorted(missing_in_returns):
        stock_info = sector[sector['SYMBOL'] == sym]
        if len(stock_info) > 0:
            print(f"  - {sym}: {stock_info['NAME'].values[0]}")

if missing_in_benchmark:
    print(f"\nStocks in returns but NOT in benchmark ({len(missing_in_benchmark)}):")
    for sym in sorted(missing_in_benchmark):
        print(f"  - {sym}")

# Check weights sum
print(f"\nBenchmark weights sum: {sector['weight_in_sector'].sum():.6f}")
print(f"Any NaN in carbon intensity: {sector['Carbon Intensity'].isna().any()}")
