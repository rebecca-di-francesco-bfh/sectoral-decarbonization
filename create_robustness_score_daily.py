"""
Robustness Score Computation for Optimal Portfolio Analysis

This script computes robustness scores for optimal portfolios by comparing
tracking error against sector volatility across different time periods.

The robustness score is calculated as:
    Robustness Score = 1 / (1 + TE/Volatility)

Where:
    - TE (Tracking Error): Standard deviation of active returns (portfolio vs benchmark)
    - Volatility: Sector benchmark volatility
    - Higher score = More robust portfolio (lower TE relative to volatility)
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

# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252

# Period definitions
PERIODS = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222",
           "0323", "0623", "0923", "1223"]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_data():
    """
    Load benchmark returns and volatility data.

    Returns
    -------
    tuple
        (benchmark_sectors_daily_returns, sector_annualized_volatility_by_quarter)
    """
    print("📂 Loading benchmark data...")

    benchmark_sectors_daily_returns = pd.read_excel(
        "data/benchmark_returns_volatility/sector_portfolio_returns_all_periods.xlsx",
        dtype={'period': str},
        index_col=0
    )

    sector_annualized_volatility_by_quarter = pd.read_excel(
        "data/benchmark_returns_volatility/sector_annualized_volatility_by_quarter.xlsx",
        dtype={'period': str}
    )

    print(f"✅ Loaded benchmark data")
    print(f"   - Daily returns shape: {benchmark_sectors_daily_returns.shape}")
    print(f"   - Volatility records: {len(sector_annualized_volatility_by_quarter)}")

    return benchmark_sectors_daily_returns, sector_annualized_volatility_by_quarter


def compute_tracking_error(r_b, r_d):
    """
    Compute annualized tracking error from benchmark and portfolio returns.

    Parameters
    ----------
    r_b : pd.Series
        Benchmark daily returns
    r_d : pd.Series
        Portfolio daily returns

    Returns
    -------
    float
        Annualized tracking error
    """
    # Active returns
    active = r_d - r_b
    TE_3m = active.std() * np.sqrt(len(active))
    return TE_3m


def process_sector(sector, period, benchmark_return_index_period, optimal_portfolios_all_te):
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
    r_b = benchmark_return_index_period[sector]

    # Load daily returns for this sector
    daily_returns_3m_period = pd.read_excel(
        f"data/daily_returns_3m/daily_returns_{period}.xlsx",
        sheet_name=sector,
        index_col=0
    )

    # Extract optimal portfolio at target TE
    optimal_portfolios_shrink_2_TE = extract_optimal_portfolios_at_target_te(
        optimal_portfolios_all_te,
        target_te_bps=TARGET_TE_BPS
    )[sector]

    # Get portfolio weights
    w_opt = optimal_portfolios_shrink_2_TE["w_opt"].astype(float).ravel()
    stock_labels = list(optimal_portfolios_shrink_2_TE["stock_labels"])
    weights_df = pd.DataFrame([w_opt], columns=stock_labels)

    # Align columns between daily returns and weights
    common_columns = daily_returns_3m_period.columns.intersection(weights_df.columns)

    # Validate alignment
    if len(common_columns) != len(stock_labels):
        print(f"   ⚠️  Warning: Column mismatch for {sector}")
        print(f"      Expected {len(stock_labels)}, got {len(common_columns)} common columns")

    # Calculate portfolio daily returns
    r_d = (daily_returns_3m_period[common_columns] * weights_df[common_columns].values).sum(axis=1)

    # Align time indices
    common_idx = r_b.index.intersection(r_d.index)

    r_b, r_d = r_b.loc[common_idx], r_d.loc[common_idx]

    if len(common_idx) == 0:
        print(f"   ⚠️  Warning: No common dates for {sector} in period {period}")
        return np.nan

    # Compute tracking error
    te_ann = compute_tracking_error(r_b, r_d)

    return te_ann


def process_period(period, benchmark_sectors_daily_returns, sector_annualized_volatility_by_quarter):
    """
    Process a single time period to compute tracking errors and robustness scores.

    Parameters
    ----------
    period : str
        Period code (e.g., '0321')
    benchmark_sectors_daily_returns : pd.DataFrame
        Benchmark returns for all periods
    sector_annualized_volatility_by_quarter : pd.DataFrame
        Sector volatility by quarter

    Returns
    -------
    pd.DataFrame or None
        DataFrame with robustness metrics for the period, or None if processing fails
    """
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING PERIOD: {period}")
    print(f"{'='*80}")

    # Check if optimal portfolio file exists
    optim_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"
    if not os.path.exists(optim_file):
        print(f"⚠️  ERROR: Missing optimal portfolios file: {optim_file}")
        print(f"   Skipping period {period}\n")
        return None

    # Load optimal portfolios
    with open(optim_file, "rb") as f:
        optimal_portfolios_all_te = pickle.load(f)

    # Filter benchmark returns for this period
    benchmark_return_index_period = benchmark_sectors_daily_returns.loc[
        benchmark_sectors_daily_returns['period'] == period
    ].drop(columns='period')

    if benchmark_return_index_period.empty:
        print(f"⚠️  WARNING: No benchmark returns found for period {period}")
        print(f"   Skipping period {period}\n")
        return None

    # Compute tracking error for each sector
    sector_te = {}
    sectors = benchmark_return_index_period.columns
    print(f"⚙️  Processing {len(sectors)} sectors...")

    for sector in sectors:
        try:
            te_ann = process_sector(sector, period, benchmark_return_index_period, optimal_portfolios_all_te)
            sector_te[sector] = te_ann
        except Exception as e:
            print(f"   ⚠️  Error processing {sector}: {str(e)}")
            sector_te[sector] = np.nan

    # Build DataFrame
    te_df = pd.DataFrame.from_dict(sector_te, orient="index", columns=["3_months_TE"])
    te_df = te_df.reset_index().rename(columns={'index': 'sector'})

    # Save TE results for this period
    te_df.to_excel(f"results/robustness/te_results_{period}_next_3m.xlsx", index=False)
    print(f"   ✅ Saved TE results for period {period}")

    # Get volatility for this period
    sector_annualized_volatility_period = sector_annualized_volatility_by_quarter.loc[
        sector_annualized_volatility_by_quarter['period'] == period
    ]

    if sector_annualized_volatility_period.empty:
        print(f"⚠️  WARNING: No volatility data found for period {period}")
        print(f"   Skipping period {period}\n")
        return None

    # Merge TE and volatility
    merged = pd.merge(te_df, sector_annualized_volatility_period, on="sector", how="inner")

    if merged.empty:
        print(f"⚠️  WARNING: No matching sectors between TE and volatility for period {period}")
        print(f"   Skipping period {period}\n")
        return None

    # Compute robustness ratio (TE / Volatility)
    merged["Robustness_Ratio"] = merged["3_months_TE"] / merged["annualized_volatility"]

    # Convert to intuitive robustness score (higher = more robust)
    merged["Robustness_Score"] = 1 / (1 + merged["Robustness_Ratio"])

    print(f"   ✅ Computed robustness scores for {len(merged)} sectors")

    return merged[['sector', 'period', '3_months_TE', 'annualized_volatility',
                   'Robustness_Ratio', 'Robustness_Score']]


def validate_robustness_scores(robust_df):
    """
    Validate the computed robustness scores.

    Parameters
    ----------
    robust_df : pd.DataFrame
        DataFrame with robustness scores
    """
    print("\n" + "="*80)
    print("🔍 VALIDATING ROBUSTNESS SCORES")
    print("="*80)

    # Check for NaN values
    nan_counts = robust_df.isna().sum()
    if nan_counts.sum() > 0:
        print("⚠️  WARNING: Found NaN values:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"   - {col}: {count} NaN values")
    else:
        print("✅ No NaN values found")

    # Check robustness score range
    score_min = robust_df['Robustness_Score'].min()
    score_max = robust_df['Robustness_Score'].max()
    print(f"✅ Robustness Score range: [{score_min:.4f}, {score_max:.4f}]")

    if score_min < 0 or score_max > 1:
        print("⚠️  WARNING: Robustness scores outside expected [0, 1] range!")

    # Check sector count per period
    sector_counts = robust_df.groupby('period')['sector'].count()
    print(f"\n✅ Sector count per period:")
    for period, count in sector_counts.items():
        print(f"   - {period}: {count} sectors")

    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   - Total records: {len(robust_df)}")
    print(f"   - Unique sectors: {robust_df['sector'].nunique()}")
    print(f"   - Unique periods: {robust_df['period'].nunique()}")
    print(f"   - Mean TE: {robust_df['3_months_TE'].mean():.4f}")
    print(f"   - Mean Volatility: {robust_df['annualized_volatility'].mean():.4f}")
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
    print("📈 GENERATING PLOTS")
    print("="*80)

    # Plot 1: Annualized Tracking Error
    print("\n📊 Plotting Annualized Tracking Error...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col='3_months_TE',
        title='3-Month Tracking Error Evolution by Sector (2021-2023)',
        ylabel='3-Month Tracking Error',
        figsize=(12, 7)
    )

    # Plot 2: Annualized Volatility
    print("📊 Plotting Annualized Volatility...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col='annualized_volatility',
        title='Annualized Volatility Evolution by Sector (2021-2023)',
        ylabel='Annualized Volatility',
        figsize=(12, 7)
    )

    # Plot 3: Robustness Score
    print("📊 Plotting Robustness Score...")
    plot_sector_evolution(
        df=robust_df_plot,
        value_col='Robustness_Score',
        title='Robustness Score Evolution by Sector (2021-2023)',
        ylabel='Robustness Score (Higher = More Robust)',
        figsize=(12, 7)
    )

    print("✅ All plots generated successfully")


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

    # Load benchmark data
    benchmark_sectors_daily_returns, sector_annualized_volatility_by_quarter = load_data()

    # Process all periods
    robustness_records = []

    for period in PERIODS:
        result = process_period(
            period,
            benchmark_sectors_daily_returns,
            sector_annualized_volatility_by_quarter
        )

        if result is not None:
            robustness_records.append(result)

    # Check if we have any results
    if len(robustness_records) == 0:
        print("\n⚠️  ERROR: No robustness data was computed for any period!")
        print("   Please check the errors above and ensure all required files exist.")
        return

    # Combine all periods
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)

    robust_df = pd.concat(robustness_records, ignore_index=True)

    print(f"✅ Combined robustness data from {len(robustness_records)} periods")
    print(f"   Total rows: {len(robust_df)}")

    # Validate results
    validate_robustness_scores(robust_df)

    # Save combined results
    output_file = "results/robustness/robustness_scores_by_period.xlsx"
    robust_df.to_excel(output_file, index=False)
    print(f"\n✅ Saved robustness scores to: {output_file}")

    # Prepare data for plotting
    robust_df_plot = robust_df.rename(columns={'sector': 'Sector', 'period': 'Period'})

    # Generate plots
    plot_robustness_metrics(robust_df_plot)

    print("\n" + "="*80)
    print("✅ ROBUSTNESS SCORE COMPUTATION COMPLETE")
    print("="*80)

    # Display top 5 most robust sectors (averaged across all periods)
    print("\n🏆 Top 5 Most Robust Sectors (Average across all periods):")
    avg_robustness = robust_df.groupby('sector')['Robustness_Score'].mean().sort_values(ascending=False)
    for i, (sector, score) in enumerate(avg_robustness.head(5).items(), 1):
        print(f"   {i}. {sector}: {score:.4f}")

    # Display top 5 least robust sectors
    print("\n⚠️  Top 5 Least Robust Sectors (Average across all periods):")
    for i, (sector, score) in enumerate(avg_robustness.tail(5).items(), 1):
        print(f"   {i}. {sector}: {score:.4f}")


if __name__ == "__main__":
    main()
