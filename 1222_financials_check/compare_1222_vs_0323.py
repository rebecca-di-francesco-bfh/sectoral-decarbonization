"""
Compare Financials between 1222 (flexibility = 0.0) and 0323 (flexibility = 0.37)
to understand what changed as the sector recovered
"""

import pandas as pd
import numpy as np
import sys

print("=" * 80)
print("FINANCIALS RECOVERY ANALYSIS: 1222 → 0323")
print("Flexibility Score: 0.0 → 0.37")
print("=" * 80)

# =============================================================================
# PART 1: STOCK COMPOSITION CHANGES
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: STOCK COMPOSITION CHANGES")
print("=" * 80)

periods = ['1222', '0323']
data_by_period = {}

for period in periods:
    dataset_file = f"Data/datasets/benchmark_weights_carbon_intensity_{period}.xlsx"

    try:
        df = pd.read_excel(dataset_file)
        financials = df[df['GICS Sector'] == 'Financials'].copy()
        data_by_period[period] = financials

        print(f"\nPeriod {period}:")
        print(f"  Total Financials stocks: {len(financials)}")

        if 'weight_in_sector' in financials.columns:
            total_weight = financials['weight_in_sector'].sum()
            print(f"  Total benchmark weight: {total_weight:.4f}")
            print(f"  Max benchmark weight: {financials['weight_in_sector'].max():.4f}")

    except Exception as e:
        print(f"\nPeriod {period}: Error - {e}")

# Compare stock composition
if '1222' in data_by_period and '0323' in data_by_period:
    df_1222 = data_by_period['1222']
    df_0323 = data_by_period['0323']

    # Find ticker column
    ticker_col = None
    for col in ['SYMBOL']:
        if col in df_1222.columns:
            ticker_col = col
            break

    if ticker_col:
        tickers_1222 = set(df_1222[ticker_col].values)
        tickers_0323 = set(df_0323[ticker_col].values)

        exited = tickers_1222 - tickers_0323
        entered = tickers_0323 - tickers_1222
        common = tickers_1222 & tickers_0323

        print(f"\n📊 Stock Changes (1222 → 0323):")
        print(f"  Common stocks: {len(common)}")
        print(f"  Exited in 0323: {len(exited)}")
        print(f"  Entered in 0323: {len(entered)}")

        # Analyze exited stocks (with volatility from 1222)
        if len(exited) > 0:
            print(f"\n{'='*80}")
            print(f"STOCKS THAT EXITED (were in 1222, not in 0323):")
            print(f"{'='*80}")

            exited_data = df_1222[df_1222[ticker_col].isin(exited)].copy()
          
            if 'weight_in_sector' in exited_data.columns:
                exited_data = exited_data.sort_values('weight_in_sector', ascending=False)

            # Get volatility data from 1222 if available
            vols_1222_dict = {}
            try:
                returns_file_1222 = f"Data/log_returns/sector_log_returns_comp_1222.xlsx"
                log_returns_1222 = pd.read_excel(returns_file_1222, sheet_name='Financials')
                if 'Date' in log_returns_1222.columns:
                    returns_clean_1222 = log_returns_1222.drop(columns=['Date']).dropna()
                else:
                    returns_clean_1222 = log_returns_1222.dropna()
                stock_vols_1222 = returns_clean_1222.std()
                vols_1222_dict = stock_vols_1222.to_dict()
           
            except:
                pass

            print(f"\n{'Stock':<10} {'Company':<40} {'Bench Weight':>12} {'Volatility':>12} {'Carbon':>10}")
            print("-" * 90)

            for idx, row in exited_data.iterrows():
                ticker = row[ticker_col]
                company = row.get('NAME', 'N/A')[:38]
                weight = row.get('weight_in_sector', 0)
                carbon = row.get('Carbon Intensity', 0)
                volatility = vols_1222_dict.get(ticker, 0)
                pct_of_sector = (weight / df_1222['weight_in_sector'].sum()) * 100

                # Flags
                weight_flag = " ⚠️ HIGH WT" if weight > 0.01 else ""
                vol_flag = " ⚠️ HIGH VOL" if volatility > 0.15 else ""

                print(f"{ticker:<10} {company:<40} {weight:12.6f} {volatility:12.6f} {carbon:10.2f}{weight_flag}{vol_flag}")

            # Summary for exited stocks
            if 'weight_in_sector' in exited_data.columns:
                total_weight_exited = exited_data['weight_in_sector'].sum()
                print(f"\nTotal benchmark weight exited: {total_weight_exited:.6f} ({100*total_weight_exited/df_1222['weight_in_sector'].sum():.2f}% of sector)")

            if len(vols_1222_dict) > 0:
                exited_vols = [vols_1222_dict.get(t, 0) for t in exited]
                if exited_vols:
                    mean_vol_exited = np.mean([v for v in exited_vols if v > 0])
                    print(f"Mean volatility of exited stocks: {mean_vol_exited:.6f}")

        # Analyze entered stocks (with volatility from 0323)
        if len(entered) > 0:
            print(f"\n{'='*80}")
            print(f"STOCKS THAT ENTERED (new in 0323):")
            print(f"{'='*80}")

            entered_data = df_0323[df_0323[ticker_col].isin(entered)].copy()
            if 'weight_in_sector' in entered_data.columns:
                entered_data = entered_data.sort_values('weight_in_sector', ascending=False)

            # Get volatility data from 0323 if available
            vols_0323_dict = {}
            try:
                returns_file_0323 = f"Data/log_returns/sector_log_returns_comp_0323.xlsx"
                log_returns_0323 = pd.read_excel(returns_file_0323, sheet_name='Financials')
                if 'Date' in log_returns_0323.columns:
                    returns_clean_0323 = log_returns_0323.drop(columns=['Date']).dropna()
                else:
                    returns_clean_0323 = log_returns_0323.dropna()
                stock_vols_0323 = returns_clean_0323.std()
                vols_0323_dict = stock_vols_0323.to_dict()
            except:
                pass

            print(f"\n{'Stock':<10} {'Company':<40} {'Bench Weight':>12} {'Volatility':>12} {'Carbon':>10}")
            print("-" * 90)

            for idx, row in entered_data.iterrows():
                ticker = row[ticker_col]
                company = row.get('NAME', 'N/A')[:38]
                weight = row.get('weight_in_sector', 0)
                carbon = row.get('Carbon Intensity', 0)
                volatility = vols_0323_dict.get(ticker, 0)

                pct_of_sector = (weight / df_0323['weight_in_sector'].sum()) * 100

                # Flags
                weight_flag = " ⚠️ HIGH WT" if weight > 0.01 else ""
                vol_flag = " ⚠️ HIGH VOL" if volatility > 0.15 else " ✅ LOW VOL" if volatility < 0.08 else ""

                print(f"{ticker:<10} {company:<40} {weight:12.6f} {volatility:12.6f} {carbon:10.2f}{weight_flag}{vol_flag}")

            # Summary for entered stocks
            if 'weight_in_sector' in entered_data.columns:
                total_weight_entered = entered_data['weight_in_sector'].sum()
                print(f"\nTotal benchmark weight entered: {total_weight_entered:.6f} ({100*total_weight_entered/df_0323['weight_in_sector'].sum():.2f}% of sector)")

            if len(vols_0323_dict) > 0:
                entered_vols = [vols_0323_dict.get(t, 0) for t in entered]
                if entered_vols:
                    mean_vol_entered = np.mean([v for v in entered_vols if v > 0])
                    print(f"Mean volatility of entered stocks: {mean_vol_entered:.6f}")

        # Weight redistribution for common stocks
        if 'weight_in_sector' in df_1222.columns and 'weight_in_sector' in df_0323.columns:
            common_1222 = df_1222[df_1222[ticker_col].isin(common)].set_index(ticker_col)['weight_in_sector']
            common_0323 = df_0323[df_0323[ticker_col].isin(common)].set_index(ticker_col)['weight_in_sector']

            weight_changes = common_0323 - common_1222

            top_gains = weight_changes.nlargest(5)
            top_losses = weight_changes.nsmallest(5)

            print(f"\n{'='*80}")
            print("WEIGHT REDISTRIBUTION (Common Stocks)")
            print(f"{'='*80}")

            print(f"\nTop 5 weight increases (1222 → 0323):")
            for ticker, change in top_gains.items():
                old_w = common_1222[ticker]
                new_w = common_0323[ticker]
                print(f"  {ticker:10s}: {old_w:.6f} → {new_w:.6f} (Δ = +{change:.6f})")

            print(f"\nTop 5 weight decreases (1222 → 0323):")
            for ticker, change in top_losses.items():
                old_w = common_1222[ticker]
                new_w = common_0323[ticker]
                print(f"  {ticker:10s}: {old_w:.6f} → {new_w:.6f} (Δ = {change:.6f})")

# =============================================================================
# PART 2: VOLATILITY CHANGES
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: VOLATILITY CHANGES")
print("=" * 80)

stock_volatilities = {}
returns_matrices = {}

for period in periods:
    returns_file = f"Data/log_returns/sector_log_returns_comp_{period}.xlsx"

    try:
        log_returns = pd.read_excel(returns_file, sheet_name='Financials')

        if 'Date' in log_returns.columns:
            returns_clean = log_returns.drop(columns=['Date']).dropna()
        else:
            returns_clean = log_returns.dropna()

        returns_matrices[period] = returns_clean

        stock_vols = returns_clean.std()
        stock_volatilities[period] = stock_vols

        print(f"\nPeriod {period}:")
        print(f"  Shape: {returns_clean.shape[0]} observations × {returns_clean.shape[1]} stocks")
        print(f"  Mean volatility: {stock_vols.mean():.6f}")
        print(f"  Median volatility: {stock_vols.median():.6f}")
        print(f"  Max volatility: {stock_vols.max():.6f}")
        print(f"  Stocks with σ > 0.10: {(stock_vols > 0.10).sum()}/{len(stock_vols)}")

    except Exception as e:
        print(f"\nPeriod {period}: Error - {e}")

# Compare volatilities
if '1222' in stock_volatilities and '0323' in stock_volatilities:
    vols_1222 = stock_volatilities['1222']
    vols_0323 = stock_volatilities['0323']

    common_stocks = set(vols_1222.index) & set(vols_0323.index)

    vol_changes = pd.DataFrame({
        'vol_1222': vols_1222,
        'vol_0323': vols_0323
    })
    vol_changes = vol_changes.loc[list(common_stocks)].copy()
    vol_changes['abs_change'] = vol_changes['vol_0323'] - vol_changes['vol_1222']
    vol_changes['pct_change'] = 100 * (vol_changes['vol_0323'] - vol_changes['vol_1222']) / vol_changes['vol_1222']

    print(f"\n{'='*80}")
    print("VOLATILITY CHANGES (Common Stocks)")
    print(f"{'='*80}")

    print(f"\nTop 10 volatility DECREASES (1222 → 0323, recovery):")
    print(f"{'Stock':<15} {'1222 σ':>12} {'0323 σ':>12} {'Change':>12} {'% Change':>12}")
    print("-" * 70)

    vol_changes_sorted = vol_changes.sort_values('abs_change')
    for stock, row in vol_changes_sorted.head(10).iterrows():
        print(f"{stock:<15} {row['vol_1222']:12.6f} {row['vol_0323']:12.6f} "
              f"{row['abs_change']:+12.6f} {row['pct_change']:+11.1f}%")

    print(f"\nSummary:")
    decreased = (vol_changes['abs_change'] < 0).sum()
    increased = (vol_changes['abs_change'] > 0).sum()
    print(f"  Volatility decreased: {decreased}/{len(vol_changes)} stocks ({100*decreased/len(vol_changes):.1f}%)")
    print(f"  Volatility increased: {increased}/{len(vol_changes)} stocks ({100*increased/len(vol_changes):.1f}%)")
    print(f"  Mean change: {vol_changes['pct_change'].mean():+.1f}%")
    print(f"  Median change: {vol_changes['pct_change'].median():+.1f}%")

# =============================================================================
# PART 3: CORRELATION CHANGES
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: CORRELATION CHANGES")
print("=" * 80)

for period in periods:
    if period in returns_matrices:
        corr_matrix = returns_matrices[period].corr()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        correlations = upper_triangle.stack()

        print(f"\nPeriod {period}:")
        print(f"  Mean correlation: {correlations.mean():.4f}")
        print(f"  Median correlation: {correlations.median():.4f}")
        print(f"  Pairs with corr > 0.7: {(correlations > 0.7).sum()}/{len(correlations)} ({100*(correlations > 0.7).sum()/len(correlations):.1f}%)")

# =============================================================================
# PART 4: OPTIMAL PORTFOLIO DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 80)
print("PART 4: OPTIMAL PORTFOLIO DIAGNOSTICS")
print("=" * 80)

for period in periods:
    pkl_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"

    try:
        import pickle
        with open(pkl_file, 'rb') as f:
            portfolios = pickle.load(f)

        if 'Financials' in portfolios:
            fin_data = portfolios['Financials']
            diag = fin_data.get('diagnostics', {})

            print(f"\nPeriod {period}:")
            print(f"  Shrinkage Alpha: {diag.get('Shrinkage Alpha', 'N/A')}")
            print(f"  Carbon reduction @2% TE: {diag.get('Reduction @2% TE (%)', 'N/A')}")
            print(f"  Condition number: {diag.get('Min_Eigval1', 0.0):.6f} (min eigenvalue)")

    except Exception as e:
        print(f"\nPeriod {period}: Error - {e}")

print("\n" + "=" * 80)
print("CONCLUSION: WHAT CHANGED FROM 1222 TO 0323?")
print("=" * 80)
print("""
Check the results above to understand the recovery:

1. STOCK COMPOSITION: Did problematic stocks exit?
2. VOLATILITY: Did volatility normalize?
3. CORRELATION: Did stocks become less correlated?
4. SHRINKAGE: Did the shrinkage factor decrease?

The flexibility score recovered from 0.0 to 0.37 because
one or more of these factors improved.
""")

print("=" * 80)
