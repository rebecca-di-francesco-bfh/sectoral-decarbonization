"""
Simple analysis to understand the Financials drop in 1222
Uses only standard library + pandas/openpyxl which should be available
"""

import sys
import os

try:
    import pandas as pd
    import pickle
except ImportError as e:
    print(f"Missing module: {e}")
    print("Please run this with your anaconda environment")
    sys.exit(1)

print("=" * 80)
print("ANALYZING FINANCIALS FLEXIBILITY SCORE DROP IN PERIOD 1222")
print("=" * 80)

# Periods to compare
periods = ['0922', '1222', '0323']

print("\n1. CHECKING RAW METRICS FROM INDIVIDUAL PERIOD FILES")
print("-" * 80)

for period in periods:
    excel_file = f"results/flexibility/l2_bandwidth_turnover_{period}.xlsx"

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)

        # Find Financials row
        fin_row = df[df['Sector'] == 'Financials']

        if not fin_row.empty:
            print(f"\nPeriod {period} - Financials:")
            print(f"  L2_lower_bound_same_obj: {fin_row['L2_lower_bound_same_obj'].values[0]}")
            print(f"  Avg_bandwidth: {fin_row['Avg_bandwidth'].values[0]}")
            print(f"  Median_bandwidth: {fin_row['Median_bandwidth'].values[0]}")
            print(f"  Max_bandwidth: {fin_row['Max_bandwidth'].values[0]}")

            # Show all sectors for this period to understand normalization
            print(f"\n  All sectors in {period} (sorted by L2):")
            df_sorted = df[['Sector', 'L2_lower_bound_same_obj', 'Avg_bandwidth']].sort_values('L2_lower_bound_same_obj', ascending=False)
            for idx, row in df_sorted.iterrows():
                marker = " <<<" if row['Sector'] == 'Financials' else ""
                print(f"    {row['Sector']:30s} L2={row['L2_lower_bound_same_obj']:.6f}, AvgBW={row['Avg_bandwidth']:.6f}{marker}")
        else:
            print(f"\nPeriod {period}: Financials not found!")
    else:
        print(f"\nPeriod {period}: File not found - {excel_file}")

print("\n\n2. CHECKING OPTIMAL PORTFOLIO DATA")
print("-" * 80)

for period in periods:
    pkl_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"

    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            portfolios = pickle.load(f)

        if 'Financials' in portfolios:
            fin_data = portfolios['Financials']

            print(f"\nPeriod {period} - Financials optimal portfolio:")

            # Show diagnostics if available
            if 'diagnostics' in fin_data:
                print(f"  Diagnostics: {fin_data['diagnostics']}")

            # Show TE info
            if 'tracking_errors' in fin_data:
                import numpy as np
                te_list = np.array(fin_data['tracking_errors'])
                print(f"  Number of TE points: {len(te_list)}")
                print(f"  TE range: [{te_list.min():.2f}, {te_list.max():.2f}] bps")

                # Find closest to 200 bps
                idx_200 = np.argmin(np.abs(te_list - 200))
                print(f"  TE closest to 200 bps: {te_list[idx_200]:.2f} bps")

                # Check carbon reduction
                if 'carbon_reductions' in fin_data:
                    c_red = fin_data['carbon_reductions'][idx_200]
                    print(f"  Carbon reduction at ~200bps: {c_red:.2f}%")

                # Check weights
                if 'weights_by_te' in fin_data and len(fin_data['weights_by_te']) > idx_200:
                    w = fin_data['weights_by_te'][idx_200]
                    print(f"  Portfolio size: {len(w)} stocks")
                    print(f"  Non-zero weights: {(w > 1e-6).sum()}")
                    print(f"  Max weight: {w.max():.4f}")
                    print(f"  HHI: {(w**2).sum():.4f}")
        else:
            print(f"\nPeriod {period}: Financials not in optimal portfolios!")
    else:
        print(f"\nPeriod {period}: File not found - {pkl_file}")

print("\n\n3. CHECKING LOG RETURNS DATA QUALITY")
print("-" * 80)

for period in periods:
    returns_file = f"Data/log_returns/sector_log_returns_comp_{period}.xlsx"

    if os.path.exists(returns_file):
        try:
            log_returns = pd.read_excel(returns_file, sheet_name='Financials')

            # Drop Date column
            if 'Date' in log_returns.columns:
                returns_data = log_returns.drop(columns=['Date'])
            else:
                returns_data = log_returns

            nan_count = returns_data.isna().sum().sum()
            total_cells = returns_data.size
            nan_pct = (nan_count / total_cells) * 100 if total_cells > 0 else 0

            print(f"\nPeriod {period} - Financials log returns:")
            print(f"  Shape: {returns_data.shape} (rows x stocks)")
            print(f"  NaN count: {nan_count} ({nan_pct:.2f}%)")

            # After dropping NaN rows
            clean_data = returns_data.dropna()
            print(f"  After dropping NaN rows: {clean_data.shape[0]} observations")
            print(f"  Stocks with data: {clean_data.shape[1]}")

            if clean_data.shape[0] < 20:
                print(f"  ⚠️ WARNING: Very few observations after cleaning!")

        except Exception as e:
            print(f"\nPeriod {period}: Error reading Financials sheet - {e}")
    else:
        print(f"\nPeriod {period}: File not found - {returns_file}")

print("\n\n4. CHECKING BENCHMARK/CARBON DATA")
print("-" * 80)

for period in periods:
    dataset_file = f"Data/datasets/benchmark_weights_carbon_intensity_{period}.xlsx"

    if os.path.exists(dataset_file):
        data = pd.read_excel(dataset_file)

        fin_data = data[data['GICS Sector'] == 'Financials']

        print(f"\nPeriod {period} - Financials in dataset:")
        print(f"  Number of stocks: {len(fin_data)}")

        # Check for NaN in carbon intensity
        nan_carbon = fin_data['Carbon Intensity'].isna().sum()
        if nan_carbon > 0:
            print(f"  ⚠️ NaN in Carbon Intensity: {nan_carbon} stocks")
        else:
            print(f"  ✅ No NaN in Carbon Intensity")

        # Show stats
        print(f"  Carbon intensity range: [{fin_data['Carbon Intensity'].min():.2f}, {fin_data['Carbon Intensity'].max():.2f}]")

    else:
        print(f"\nPeriod {period}: File not found - {dataset_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
