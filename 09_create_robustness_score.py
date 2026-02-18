"""
Robustness Score Computation for Sector-Level Optimal Portfolios
----------------------------------------------------------------

This module implements the robustness dimension of the Decarbonization Readiness
Index. Robustness measures how stable the realized tracking error of decarbonized
portfolios remains when evaluated out of sample, relative to the underlying risk
of the sector benchmark.

For each sector and period, the procedure:
    1. Extracts the optimized decarbonized portfolio at a fixed ex-ante tracking
       error target (2% annualized).
    2. Computes daily portfolio and benchmark returns over the subsequent
       three-month out-of-sample window.
    3. Computes the realized annualized tracking error of the decarbonized
       portfolio relative to the benchmark.
    4. Scales the realized tracking error by the sector’s realized benchmark
       volatility to account for structural differences in sector risk.
    5. Constructs a robustness score by applying an inverted within-period
       min–max normalization across sectors.

Definitions
-----------
Let TE_{s,t} denote the realized annualized tracking error of the decarbonized
portfolio for sector s in period t, computed from daily out-of-sample returns.
Let σ_{s,t} denote the realized annualized volatility of the corresponding sector
benchmark over the same window.

The raw robustness metric is defined as the volatility-adjusted tracking error:
    m^{rob}_{s,t} = TE_{s,t} / σ_{s,t}.

Within each period, this ratio is min–max normalized across sectors and inverted
so that higher values correspond to greater robustness. A sector is therefore
considered more robust when its decarbonized portfolio maintains low realized
tracking error relative to its benchmark volatility, compared to other sectors
in the same evaluation period.

The output consists of realized tracking errors, benchmark volatilities,
volatility-adjusted robustness ratios, and normalized robustness scores for each
sector and period.
"""

import os
import pickle
import numpy as np
import pandas as pd
from utils import extract_optimal_portfolios_at_target_te
from plot_functions import plot_sector_evolution


# =============================================================================
# CONFIGURATION
# =============================================================================

# Target tracking error in basis points
TARGET_TE_BPS = 200

# Tracking error mode: "annualized" or "daily"
TE_MODE = "annualized"  # "annualized" or "daily"

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252

# Period definitions
PERIODS = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222",
           "0323", "0623", "0923", "1223"]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_tracking_error(r_b, r_d, mode="annualized"):
    """
    Compute tracking error between benchmark and portfolio returns.

    Parameters
    ----------
    r_b : pd.Series
        Benchmark daily returns
    r_d : pd.Series
        Portfolio daily returns
    mode : str
        "annualized" or "daily"

    Returns
    -------
    float
        Tracking error in the chosen unit
    """
    active = r_d - r_b

    # Daily TE
    te_daily = active.std()
    print("TE_DAILY: ", te_daily)
    print("TE DAILY -> ANNUALIZED: ", te_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
    if mode == "daily":
        return te_daily

    elif mode == "annualized":

        return te_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    else:
        raise ValueError(f"Unknown TE mode: {mode}. Use 'annualized' or 'daily'.")



def process_sector(sector, period, optimal_portfolios_all_te):
    """
    Process a single sector to compute tracking error.

    Parameters
    ----------
    sector : str
        Sector name
    period : str
        Period code (e.g., '0321')
    benchmark_return_index_period : pd.DataFrame
        Benchmark returns for the period
    optimal_portfolios_all_te : dict
        Optimal portfolios at all TE levels

    Returns
    -------
    float
        Annualized tracking error for the sector
    """
    # Get benchmark returns for this sector
   

    # Load daily returns for this sector
    daily_returns_3m_period = pd.read_excel(
        f"data/daily_returns_3m/daily_returns_{period}.xlsx",
        sheet_name=sector,
        index_col=0
    )

    # Extract optimal portfolio at target TE -> 2% 
    optimal_portfolios_shrink_2_TE = extract_optimal_portfolios_at_target_te(
        optimal_portfolios_all_te,
        target_te_bps=TARGET_TE_BPS
    )[sector]

    # Get portfolio weights of the optimised portfolios 
    w_opt = optimal_portfolios_shrink_2_TE["w_opt"].astype(float).ravel()
    stock_labels = list(optimal_portfolios_shrink_2_TE["stock_labels"])
    opt_weights_df = pd.DataFrame([w_opt], columns=stock_labels)

    # Align columns between daily returns and weights
    common_columns = daily_returns_3m_period.columns.intersection(opt_weights_df.columns)

    # Validate alignment
    if len(common_columns) != len(stock_labels):
        print(f"      Warning: Column mismatch for {sector}")
        print(f"      Expected {len(stock_labels)}, got {len(common_columns)} common columns")


    # Calculate portfolio daily returns
    r_d = (daily_returns_3m_period[common_columns] * opt_weights_df[common_columns].values).sum(axis=1)

        # Get portfolio weights of the optimised portfolios 
    w_bench = optimal_portfolios_shrink_2_TE["w_bench"].astype(float).ravel()
    stock_labels = list(optimal_portfolios_shrink_2_TE["stock_labels"])
    bench_weights_df = pd.DataFrame([w_bench], columns=stock_labels)

    # Align columns between daily returns and weights
    common_columns = daily_returns_3m_period.columns.intersection(bench_weights_df.columns)
    
    # Validate alignment
    if len(common_columns) != len(stock_labels):
        print(f"      Warning: Column mismatch for {sector}")
        print(f"      Expected {len(stock_labels)}, got {len(common_columns)} common columns")

    
    # Calculate portfolio daily returns
    r_b = (daily_returns_3m_period[common_columns] * bench_weights_df[common_columns].values).sum(axis=1)

    # Align time indices
    common_idx = r_b.index.intersection(r_d.index)

    r_b, r_d = r_b.loc[common_idx], r_d.loc[common_idx]

    if len(common_idx) == 0:
        print(f"      Warning: No common dates for {sector} in period {period}")
        return np.nan

    # Compute tracking error
    te_ann = compute_tracking_error(r_b, r_d, mode=TE_MODE)

    return te_ann, r_b, r_d


def process_period(period):
    """
    Process a single time period to compute tracking errors and robustness scores.

    Parameters
    ----------
    period : str
        Period code (e.g., '0321')

    Returns
    -------
    pd.DataFrame or None
        DataFrame with robustness metrics for the period, or None if processing fails
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PERIOD: {period}")
    print(f"{'='*80}")

    # Check if optimal portfolio file exists
    optim_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"
    if not os.path.exists(optim_file):
        print(f"ERROR: Missing optimal portfolios file: {optim_file}")
        print(f"Skipping period {period}\n")
        return None

    # Load optimal portfolios
    with open(optim_file, "rb") as f:
        optimal_portfolios_all_te = pickle.load(f)


    # Compute tracking error for each sector
    sectors = optimal_portfolios_all_te.keys()
    print(f"      Processing {len(sectors)} sectors...")

    sector_te = {}
    sector_bench_vol = {}
    sector_bench_vol_daily = {}

    timeseries_rows = []   # <- NEW
    summary_rows = []      # <- NEW

    for sector in sectors:
        try:
            te_ann, r_b, r_d = process_sector(sector, period, optimal_portfolios_all_te)
            sector_te[sector] = te_ann

            # --- OOS summary stats (3 months) ---
            # total return over the 3M window
            bench_ret = (1 + r_b).prod() - 1
            decarb_ret = (1 + r_d).prod() - 1

            # annualized vol over the same window
            bench_vol = r_b.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            decarb_vol = r_d.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

            sector_bench_vol[sector] = float(bench_vol)
            sector_bench_vol_daily[sector] = float(r_b.std())

            # annualized TE (already computed as te_ann), but keep consistent:
            te_oos = te_ann

            summary_rows.append({
                "Sector": sector,
                "Period": period,
                "Bench_Return_3m": float(bench_ret),
                "Decarb_Return_3m": float(decarb_ret),
                "Bench_Vol_ann": float(bench_vol),
                "Decarb_Vol_ann": float(decarb_vol),
                "TE_ann": float(te_oos),
            })

            # --- Timeseries in long format ---
            tmp_ts = pd.DataFrame({
                "Date": pd.to_datetime(r_b.index),
                "Sector": sector,
                "Period": period,
                "Bench_Return": r_b.values,
                "Decarb_Return": r_d.values,
            })
            timeseries_rows.append(tmp_ts)

        except Exception as e:
            print(f"      Error processing {sector}: {str(e)}")
            sector_te[sector] = np.nan


    # Build DataFrame
    colname = "annualized_TE" if TE_MODE == "annualized" else "daily_TE"

    te_df = pd.DataFrame.from_dict(sector_te, orient="index", columns=[colname])
    te_df = te_df.reset_index().rename(columns={'index': 'sector'})

    # Save TE results for this period
    te_df.to_excel(f"results/robustness/te_results_{period}_next_3m.xlsx", index=False)
    print(f"      Saved TE results for period {period}")

    # --- Build realized benchmark volatility DF from OOS window ---
    vol_df = pd.DataFrame({
        "sector": list(sector_bench_vol.keys()),
        "annualized_volatility": list(sector_bench_vol.values()),
        "daily_volatility": list(sector_bench_vol_daily.values()),
    })

    # Merge TE + realized benchmark vol
    merged = te_df.merge(vol_df, on="sector", how="inner")
    merged["period"] = period   # ✅ add this

    # Choose which vol column to use based on TE_MODE
    VOL_COL = "annualized_volatility" if TE_MODE == "annualized" else "daily_volatility"


    # ------------------------------------------------------------------
    # Robustness metric: volatility-adjusted tracking error
    # ------------------------------------------------------------------

    # Raw robustness metric m^{rob}_{s,t} = TE / sigma
    merged["Robustness_Ratio"] = merged[colname] / merged[VOL_COL]

    def minmax_within_period(x: pd.Series) -> pd.Series:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            # If all sectors have identical values, assign neutral score
            return pd.Series(0.5, index=x.index)
        return (x - lo) / (hi - lo)

    # Within-period min–max normalization and inversion
    merged["Robustness_Score"] = 1.0 - minmax_within_period(merged["Robustness_Ratio"])


    print(f"      Computed robustness scores for {len(merged)} sectors")

    # Save timeseries for this period
    ts_df = pd.concat(timeseries_rows, ignore_index=True)
    os.makedirs("results/robustness", exist_ok=True)
    ts_df.to_parquet(f"results/robustness/risk_return_timeseries_{period}.parquet", index=False)

    # Save summary for this period
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(f"results/robustness/risk_return_summary_{period}.xlsx", index=False)

    return merged[['sector', 'period', colname, VOL_COL, 'Robustness_Ratio', 'Robustness_Score']]


def validate_robustness_scores(robust_df):
    """
    Validate the computed robustness scores.

    Parameters
    ----------
    robust_df : pd.DataFrame
        DataFrame with robustness scores
    """
    print("\n" + "="*80)
    print("      VALIDATING ROBUSTNESS SCORES")
    print("="*80)

    # Check for NaN values
    nan_counts = robust_df.isna().sum()
    if nan_counts.sum() > 0:
        print("       WARNING: Found NaN values:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"   - {col}: {count} NaN values")
    else:
        print("      No NaN values found")

    # Check robustness score range
    score_min = robust_df['Robustness_Score'].min()
    score_max = robust_df['Robustness_Score'].max()
    print(f"Robustness Score range: [{score_min:.4f}, {score_max:.4f}]")

    if score_min < 0 or score_max > 1:
        print("      WARNING: Robustness scores outside expected [0, 1] range!")

    # Check sector count per period
    sector_counts = robust_df.groupby('period')['sector'].count()
    print(f"\n      Sector count per period:")
    for period, count in sector_counts.items():
        print(f"   - {period}: {count} sectors")

    # Summary statistics
    print(f"\n Summary Statistics:")
    print(f"   - Total records: {len(robust_df)}")
    print(f"   - Unique sectors: {robust_df['sector'].nunique()}")
    print(f"   - Unique periods: {robust_df['period'].nunique()}")
    print(f"   - Mean TE: {robust_df[f"{TE_MODE}_TE"].mean():.4f}")
    print(f"   - Mean Volatility: {robust_df[f"{TE_MODE}_volatility"].mean():.4f}")
    print(f"   - Mean Robustness Score: {robust_df['Robustness_Score'].mean():.4f}")


def plot_robustness_metrics(robust_df_plot):
    """
    Generate plots for robustness metrics.

    Parameters
    ----------
    robust_df_plot : pd.DataFrame
        DataFrame with renamed columns for plotting
    """
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # Plot 1: Tracking Error
    print("\nPlotting Tracking Error...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col=f'{TE_MODE}_TE',
        title=f'{TE_MODE} Tracking Error Evolution by Sector (2021-2023)',
        ylabel=f'{TE_MODE} Tracking Error',
        figsize=(12, 7)
    )

    # Plot 2: Volatility
    print("Plotting Volatility...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col=f'{TE_MODE}_volatility',
        title=f'{TE_MODE} Volatility Evolution by Sector (2021-2023)',
        ylabel=f'{TE_MODE} Volatility',
        figsize=(12, 7)
    )

    plot_sector_evolution(
        df=robust_df_plot,
        value_col='Robustness_Ratio',
        title='Robustness Ratio Evolution by Sector (2021-2023)',
        ylabel='Robustness Ratio (Lower = More Robust)',
        figsize=(12, 7)
    )

    plot_sector_evolution(
    df=robust_df_plot,
    value_col="Robustness_Ratio",
    title=None,   # <- no title in figure
    ylabel="Robustness Ratio (Lower = More Robust)",
    figsize=(12, 7),
    show=False,
    savepath="results/robustness/robustness_ratio_evolution.pdf",
)

  
    # Plot 4: Robustness Score (from Robustness Ratio)
    print("Plotting Robustness Score...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col='Robustness_Score',
        title='Robustness Score Evolution by Sector (2021-2023)',
        ylabel='Robustness Score (Higher = More Robust)',
        figsize=(12, 7)
    )


    print("All plots generated successfully")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to compute robustness scores across all periods.
    """
    print("\n" + "="*80)
    print("ROBUSTNESS SCORE COMPUTATION")
    print("="*80)
    print(f"Processing {len(PERIODS)} time periods...")
    print(f"Target Tracking Error: {TARGET_TE_BPS} bps")
    print("="*80)

    # Create output directory
    os.makedirs("results/robustness", exist_ok=True)

    # Process all periods
    robustness_records = []

    for period in PERIODS:
        result = process_period(
        period
    )
        if result is not None:
            robustness_records.append(result)

    # Check if we have any results
    if len(robustness_records) == 0:
        print("\nERROR: No robustness data was computed for any period!")
        print("   Please check the errors above and ensure all required files exist.")
        return

    # Combine all periods
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)

    robust_df = pd.concat(robustness_records, ignore_index=True)

    print(f"Combined robustness data from {len(robustness_records)} periods")
    print(f"   Total rows: {len(robust_df)}")

    # Validate results
    validate_robustness_scores(robust_df)

    # Save combined results
    output_file = "results/robustness/robustness_scores_by_period.xlsx"
    robust_df.to_excel(output_file, index=False)
    robust_df.to_parquet("results/robustness/robustness_scores_by_period.parquet", index=False)
    print(f"\nSaved robustness scores to: {output_file}")

    # Prepare data for plotting
    robust_df_plot = robust_df.rename(columns={'sector': 'Sector', 'period': 'Period'})

    # Generate plots
    plot_robustness_metrics(robust_df_plot)

    print("\n" + "="*80)
    print("ROBUSTNESS SCORE COMPUTATION COMPLETE")
    print("="*80)

    # Display top 5 most robust sectors (averaged across all periods)
    print("\nTop 5 Most Robust Sectors (Average across all periods):")
    avg_robustness = robust_df.groupby('sector')['Robustness_Score'].mean().sort_values(ascending=False)
    for i, (sector, score) in enumerate(avg_robustness.head(5).items(), 1):
        print(f"   {i}. {sector}: {score:.4f}")

    # Display top 5 least robust sectors
    print("\nTop 5 Least Robust Sectors (Average across all periods):")
    for i, (sector, score) in enumerate(avg_robustness.tail(5).items(), 1):
        print(f"   {i}. {sector}: {score:.4f}")

    # Combine and save all-period risk/return summaries
    all_summary = []
    all_ts = []

    for p in PERIODS:
        sx = f"results/robustness/risk_return_summary_{p}.xlsx"
        tx = f"results/robustness/risk_return_timeseries_{p}.parquet"
        if os.path.exists(sx): all_summary.append(pd.read_excel(sx, dtype={"Period": str}))
        if os.path.exists(tx): all_ts.append(pd.read_parquet(tx))

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_excel(
            "results/robustness/risk_return_summary_all_periods.xlsx", index=False
        )
        pd.concat(all_summary, ignore_index=True).to_parquet(
            "results/robustness/risk_return_summary_all_periods.parquet", index=False
        )   
    if all_ts:
        pd.concat(all_ts, ignore_index=True).to_parquet(
            "results/robustness/risk_return_timeseries_all_periods.parquet", index=False
        )


if __name__ == "__main__":
    main()
