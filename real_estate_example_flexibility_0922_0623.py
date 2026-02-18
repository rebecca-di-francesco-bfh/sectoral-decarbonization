"""
Compare Real Estate between 0922 (flexibility = 0.0) and 0623 (flexibility = 0.37)
to understand what changed as the sector recovered
"""

import pandas as pd
import numpy as np
import sys

print("=" * 80)
print("Real Estate RECOVERY ANALYSIS: 0922 → 0623")
print("Flexibility Score: 0.0 → 0.37")
print("=" * 80)

# =============================================================================
# PART 1: STOCK COMPOSITION CHANGES
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: STOCK COMPOSITION CHANGES")
print("=" * 80)

periods = ['0922', '0623']
data_by_period = {}

for period in periods:
    dataset_file = f"Data/Datasets/benchmark_weights_carbon_intensity_{period}.xlsx"

    try:
        df = pd.read_excel(dataset_file)
        real_estate = df[df['GICS Sector'] == 'Real Estate'].copy()
        data_by_period[period] = real_estate 

        print(f"\nPeriod {period}:")
        print(f"  Total Real Estate stocks: {len(real_estate )}")

        if 'weight_in_sector' in real_estate .columns:
            total_weight = real_estate ['weight_in_sector'].sum()
            print(f"  Total benchmark weight: {total_weight:.4f}")
            print(f"  Max benchmark weight: {real_estate['weight_in_sector'].max():.4f}")

    except Exception as e:
        print(f"\nPeriod {period}: Error - {e}")

# Compare stock composition
if '0922' in data_by_period and '0623' in data_by_period:
    df_0922 = data_by_period['0922']
    df_0623 = data_by_period['0623']

    # Find ticker column
    ticker_col = None
    for col in ['SYMBOL']:
        if col in df_0623.columns:
            ticker_col = col
            break

    if ticker_col:
        tickers_0922 = set(df_0922[ticker_col].values)
        tickers_0623 = set(df_0623[ticker_col].values)

        exited = tickers_0922 - tickers_0623
        entered = tickers_0623 - tickers_0922
        common = tickers_0922 & tickers_0623

        print(f"\n📊 Stock Changes (0922 → 0623):")
        print(f"  Common stocks: {len(common)}")
        print(f"  Exited in 0623: {len(exited)}")
        print(f"  Entered in 0623: {len(entered)}")

        # Analyze exited stocks (with volatility from 0922)
        if len(exited) > 0:
            print(f"\n{'='*80}")
            print(f"STOCKS THAT EXITED (were in 0922, not in 0623):")
            print(f"{'='*80}")

            exited_data = df_0922[df_0922[ticker_col].isin(exited)].copy()
          
            if 'weight_in_sector' in exited_data.columns:
                exited_data = exited_data.sort_values('weight_in_sector', ascending=False)

            # Get volatility data from 0922 if available
            vols_0922_dict = {}
            try:
                returns_file_0922 = f"Data/log_returns/sector_log_returns_comp_0922.xlsx"
                log_returns_0922 = pd.read_excel(returns_file_0922, sheet_name='Real Estate')
                if 'Date' in log_returns_0922.columns:
                    returns_clean_0922 = log_returns_0922.drop(columns=['Date']).dropna()
                else:
                    returns_clean_0922 = log_returns_0922.dropna()
                stock_vols_0922 = returns_clean_0922.std()
                vols_0922_dict = stock_vols_0922.to_dict()

            except:
                pass

            print(f"\n{'Stock':<10} {'Company':<40} {'Bench Weight':>12} {'Volatility':>12} {'Carbon':>10}")
            print("-" * 90)

            for idx, row in exited_data.iterrows():
                ticker = row[ticker_col]
                company = row.get('NAME', 'N/A')[:38]
                weight = row.get('weight_in_sector', 0)
                carbon = row.get('Carbon Intensity', 0)
                volatility = vols_0922_dict.get(ticker, 0)
                pct_of_sector = (weight / df_0922['weight_in_sector'].sum()) * 100

                # Flags
                weight_flag = " ⚠️ HIGH WT" if weight > 0.01 else ""
                vol_flag = " ⚠️ HIGH VOL" if volatility > 0.15 else ""

                print(f"{ticker:<10} {company:<40} {weight:12.6f} {volatility:12.6f} {carbon:10.2f}{weight_flag}{vol_flag}")

            # Summary for exited stocks
            if 'weight_in_sector' in exited_data.columns:
                total_weight_exited = exited_data['weight_in_sector'].sum()
                print(f"\nTotal benchmark weight exited: {total_weight_exited:.6f} ({100*total_weight_exited/df_0922['weight_in_sector'].sum():.2f}% of sector)")

            # Calculate carbon footprint contribution of exited stocks
            if 'weight_in_sector' in exited_data.columns and 'Carbon Intensity' in exited_data.columns:
                # Weighted carbon intensity of exited stocks
                exited_carbon_contrib = (exited_data['weight_in_sector'] * exited_data['Carbon Intensity']).sum()

                # Total sector weighted carbon intensity
                total_carbon_contrib = (df_0922['weight_in_sector'] * df_0922['Carbon Intensity']).sum()

                if total_carbon_contrib > 0:
                    pct_carbon_exited = (exited_carbon_contrib / total_carbon_contrib) * 100
                    print(f"Carbon footprint contribution of exited stocks: {pct_carbon_exited:.2f}% of sector's total carbon")

                    # Also show average carbon intensity
                    avg_carbon_exited = (exited_data['Carbon Intensity'] * exited_data['weight_in_sector']).sum() / exited_data['weight_in_sector'].sum() if exited_data['weight_in_sector'].sum() > 0 else 0
                    avg_carbon_sector = (df_0922['Carbon Intensity'] * df_0922['weight_in_sector']).sum() / df_0922['weight_in_sector'].sum()
                    print(f"Avg carbon intensity of exited stocks: {avg_carbon_exited:.2f} (sector avg: {avg_carbon_sector:.2f})")

            if len(vols_0922_dict) > 0:
                exited_vols = [vols_0922_dict.get(t, 0) for t in exited]
                if exited_vols:
                    mean_vol_exited = np.mean([v for v in exited_vols if v > 0])
                    print(f"Mean volatility of exited stocks: {mean_vol_exited:.6f}")

        # Analyze entered stocks (with volatility from 0623)
        if len(entered) > 0:
            print(f"\n{'='*80}")
            print(f"STOCKS THAT ENTERED (new in 0623):")
            print(f"{'='*80}")

            entered_data = df_0623[df_0623[ticker_col].isin(entered)].copy()
            if 'weight_in_sector' in entered_data.columns:
                entered_data = entered_data.sort_values('weight_in_sector', ascending=False)

            # Get volatility data from 0623 if available
            vols_0623_dict = {}
            try:
                returns_file_0623 = f"Data/log_returns/sector_log_returns_comp_0623.xlsx"
                log_returns_0623 = pd.read_excel(returns_file_0623, sheet_name='Real Estate')
                if 'Date' in log_returns_0623.columns:
                    returns_clean_0623 = log_returns_0623.drop(columns=['Date']).dropna()
                else:
                    returns_clean_0623 = log_returns_0623.dropna()
                stock_vols_0623 = returns_clean_0623.std()
                vols_0623_dict = stock_vols_0623.to_dict()
            except:
                pass

            print(f"\n{'Stock':<10} {'Company':<40} {'Bench Weight':>12} {'Volatility':>12} {'Carbon':>10}")
            print("-" * 90)

            for idx, row in entered_data.iterrows():
                ticker = row[ticker_col]
                company = row.get('NAME', 'N/A')[:38]
                weight = row.get('weight_in_sector', 0)
                carbon = row.get('Carbon Intensity', 0)
                volatility = vols_0623_dict.get(ticker, 0)

                pct_of_sector = (weight / df_0623['weight_in_sector'].sum()) * 100

                # Flags
                weight_flag = " ⚠️ HIGH WT" if weight > 0.01 else ""
                vol_flag = " ⚠️ HIGH VOL" if volatility > 0.15 else " ✅ LOW VOL" if volatility < 0.08 else ""

                print(f"{ticker:<10} {company:<40} {weight:12.6f} {volatility:12.6f} {carbon:10.2f}{weight_flag}{vol_flag}")

            # Summary for entered stocks
            if 'weight_in_sector' in entered_data.columns:
                total_weight_entered = entered_data['weight_in_sector'].sum()
                print(f"\nTotal benchmark weight entered: {total_weight_entered:.6f} ({100*total_weight_entered/df_0623['weight_in_sector'].sum():.2f}% of sector)")

            # Calculate carbon footprint contribution of entered stocks
            if 'weight_in_sector' in entered_data.columns and 'Carbon Intensity' in entered_data.columns:
                # Weighted carbon intensity of entered stocks
                entered_carbon_contrib = (entered_data['weight_in_sector'] * entered_data['Carbon Intensity']).sum()

                # Total sector weighted carbon intensity
                total_carbon_contrib = (df_0623['weight_in_sector'] * df_0623['Carbon Intensity']).sum()

                if total_carbon_contrib > 0:
                    pct_carbon_entered = (entered_carbon_contrib / total_carbon_contrib) * 100
                    print(f"Carbon footprint contribution of entered stocks: {pct_carbon_entered:.2f}% of sector's total carbon")

                    # Also show average carbon intensity
                    avg_carbon_entered = (entered_data['Carbon Intensity'] * entered_data['weight_in_sector']).sum() / entered_data['weight_in_sector'].sum() if entered_data['weight_in_sector'].sum() > 0 else 0
                    avg_carbon_sector = (df_0623['Carbon Intensity'] * df_0623['weight_in_sector']).sum() / df_0623['weight_in_sector'].sum()
                    print(f"Avg carbon intensity of entered stocks: {avg_carbon_entered:.2f} (sector avg: {avg_carbon_sector:.2f})")

            if len(vols_0623_dict) > 0:
                entered_vols = [vols_0623_dict.get(t, 0) for t in entered]
                if entered_vols:
                    mean_vol_entered = np.mean([v for v in entered_vols if v > 0])
                    print(f"Mean volatility of entered stocks: {mean_vol_entered:.6f}")

        # Carbon concentration analysis
        print(f"\n{'='*80}")
        print("CARBON CONCENTRATION ANALYSIS")
        print(f"{'='*80}")

        for period_code, period_df in [('0922', df_0922), ('0623', df_0623)]:
            if 'weight_in_sector' in period_df.columns and 'Carbon Intensity' in period_df.columns:
                # Sort by weight
                sorted_df = period_df.sort_values('weight_in_sector', ascending=False).copy()

                # Calculate cumulative weight
                sorted_df['cumulative_weight'] = sorted_df['weight_in_sector'].cumsum()

                # Find top stocks accounting for 20%, 50%, 80% of portfolio
                thresholds = [0.20, 0.50, 0.80]

                print(f"\nPeriod {period_code}:")
                for threshold in thresholds:
                    # Get stocks up to this threshold
                    top_stocks = sorted_df[sorted_df['cumulative_weight'] <= threshold].copy()

                    # If cumulative weight doesn't reach threshold exactly, add one more stock
                    if len(top_stocks) < len(sorted_df):
                        cum_so_far = top_stocks['cumulative_weight'].max() if len(top_stocks) > 0 else 0
                        if cum_so_far < threshold:
                            next_idx_pos = len(top_stocks)
                            if next_idx_pos < len(sorted_df):
                                top_stocks = sorted_df.iloc[:next_idx_pos + 1].copy()

                    if len(top_stocks) > 0:
                        # Calculate weighted carbon intensity of top stocks
                        top_carbon = (top_stocks['Carbon Intensity'] * top_stocks['weight_in_sector']).sum()
                        top_weight = top_stocks['weight_in_sector'].sum()
                        top_avg_carbon = top_carbon / top_weight if top_weight > 0 else 0

                        # Overall sector average
                        sector_carbon = (period_df['Carbon Intensity'] * period_df['weight_in_sector']).sum()
                        sector_weight = period_df['weight_in_sector'].sum()
                        sector_avg_carbon = sector_carbon / sector_weight if sector_weight > 0 else 0

                        # Percentage of carbon from top stocks
                        pct_carbon_from_top = (top_carbon / sector_carbon * 100) if sector_carbon > 0 else 0

                        print(f"  Top stocks (~{threshold*100:.0f}% of weight):")
                        print(f"    Number of stocks: {len(top_stocks)}")
                        print(f"    Actual weight: {top_weight:.4f} ({top_weight*100:.1f}%)")
                        print(f"    Avg carbon intensity: {top_avg_carbon:.2f} (sector: {sector_avg_carbon:.2f})")
                        print(f"    Carbon contribution: {pct_carbon_from_top:.1f}% of sector's total carbon")
                  
                        print("    Top constituents:")
                        print("      Ticker      Weight      CarbonIntensity")
                        for idx, row in top_stocks.iterrows():
                            ticker = row.get('SYMBOL', 'N/A')
                            w = row['weight_in_sector']
                            c = row['Carbon Intensity']
                            print(f"      {ticker:<10}  {w:>8.4f}     {c:>8.2f}")
                        print()

                        if top_avg_carbon > sector_avg_carbon * 1.1:
                            print(f"    ⚠️  Top stocks are MORE carbon-intensive (+{((top_avg_carbon/sector_avg_carbon - 1)*100):.1f}%)")
                        elif top_avg_carbon < sector_avg_carbon * 0.9:
                            print(f"    ✅ Top stocks are LESS carbon-intensive ({((top_avg_carbon/sector_avg_carbon - 1)*100):.1f}%)")

        # Compare concentration between periods
        if 'weight_in_sector' in df_0922.columns and 'weight_in_sector' in df_0623.columns and 'Carbon Intensity' in df_0922.columns and 'Carbon Intensity' in df_0623.columns:
            print(f"\n{'='*80}")
            print("CHANGE IN CARBON CONCENTRATION (0922 → 0623)")
            print(f"{'='*80}")

            # Calculate for both periods at key thresholds
            for threshold in [0.20, 0.50]:
                # 0922
                sorted_0922 = df_0922.sort_values('weight_in_sector', ascending=False).copy()
                sorted_0922['cumulative_weight'] = sorted_0922['weight_in_sector'].cumsum()
                top_0922 = sorted_0922[sorted_0922['cumulative_weight'] <= threshold].copy()

                if len(top_0922) < len(sorted_0922):
                    cum_so_far = top_0922['cumulative_weight'].max() if len(top_0922) > 0 else 0
                    if cum_so_far < threshold:
                        next_idx_pos = len(top_0922)
                        if next_idx_pos < len(sorted_0922):
                            top_0922 = sorted_0922.iloc[:next_idx_pos + 1].copy()

                carbon_0922 = (top_0922['Carbon Intensity'] * top_0922['weight_in_sector']).sum() / top_0922['weight_in_sector'].sum() if top_0922['weight_in_sector'].sum() > 0 else 0

                # 0623
                sorted_0623 = df_0623.sort_values('weight_in_sector', ascending=False).copy()
                sorted_0623['cumulative_weight'] = sorted_0623['weight_in_sector'].cumsum()
                top_0623 = sorted_0623[sorted_0623['cumulative_weight'] <= threshold].copy()

                if len(top_0623) < len(sorted_0623):
                    cum_so_far = top_0623['cumulative_weight'].max() if len(top_0623) > 0 else 0
                    if cum_so_far < threshold:
                        next_idx_pos = len(top_0623)
                        if next_idx_pos < len(sorted_0623):
                            top_0623 = sorted_0623.iloc[:next_idx_pos + 1].copy()

                carbon_0623 = (top_0623['Carbon Intensity'] * top_0623['weight_in_sector']).sum() / top_0623['weight_in_sector'].sum() if top_0623['weight_in_sector'].sum() > 0 else 0

                # Compare
                change = carbon_0623 - carbon_0922
                pct_change = (change / carbon_0922 * 100) if carbon_0922 > 0 else 0

                print(f"\nTop ~{threshold*100:.0f}% of portfolio by weight:")
                print(f"  Avg carbon intensity 0922: {carbon_0922:.2f}")
                print(f"  Avg carbon intensity 0623: {carbon_0623:.2f}")
                print(f"  Change: {change:+.2f} ({pct_change:+.1f}%)")

                if change < -5:
                    print(f"  ✅ Carbon concentration DECREASED (easier to reduce carbon)")
                elif change > 5:
                    print(f"  ⚠️  Carbon concentration INCREASED (harder to reduce carbon)")
                else:
                    print(f"  → Carbon concentration remained STABLE")

       