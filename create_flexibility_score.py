"""
Flexibility Score Computation for Optimal Portfolio Analysis

This script computes flexibility scores for optimal portfolios across different sectors
and time periods. Flexibility is measured through:
1. Epsilon-bands (weight variation ranges)
2. L2 lower bounds (portfolio distance measures)

Author: IRP17 Team
"""

import os
import pickle
import numpy as np
import pandas as pd
import cvxpy as cp
from joblib import Parallel, delayed
from utils import sigma_shrink_fn, extract_optimal_portfolios_at_target_te
from plot_functions import plot_sector_evolution
from utils import solve_qp_with_fallback

# =============================================================================
# CONFIGURATION
# =============================================================================
def minmax_norm_grouped(df, col):
    return df.groupby("Period")[col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
# Optimization parameters
EPS = 0.02
DELTA_R = 1e-3
DELTA_TE = 1e-8
K_DIRS = 500
NO_IMPROVE_PATIENCE = 40
ZERO_TOL = 1e-10

# Period definitions
PERIODS = {
    "0321": "Mar 2021",
    "0621": "Jun 2021",
    "0921": "Sep 2021",
    "1221": "Dec 2021",
    "0322": "Mar 2022",
    "0622": "Jun 2022",
    "0922": "Sep 2022",
    "1222": "Dec 2022",
    "0323": "Mar 2023",
    "0623": "Jun 2023",
    "0923": "Sep 2023",
    "1223": "Dec 2023"
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def process_sector(sector_name, info, data, log_returns_all):
    """
    Process a single sector to compute epsilon-bands and L2 lower bounds.

    Parameters
    ----------
    sector_name : str
        Name of the sector
    info : dict
        Portfolio information including w_bench, w_opt, stock_labels
    data : pd.DataFrame
        Dataset with carbon intensity information
    log_returns_all : dict
        Dictionary of log returns by sector

    Returns
    -------
    tuple
        (sector_name, bands_df, summary_row)
    """
    # Covariance for this sector
    
    R_clean = log_returns_all[sector_name].drop(columns=['Date'])
    R_clean = R_clean.iloc[:-3]
    if R_clean.isna().any().any():
        print(f"   ⚠️  Warning: NaN values found in {sector_name} returns")

    Sigma_sector, _ = sigma_shrink_fn(R_clean)

    w_bench = info["w_bench"].astype(float).ravel()
    w_opt = info["w_opt"].astype(float).ravel()
    stock_labels = list(info["stock_labels"])

    # Carbon vector
    c_vec = data.loc[data['GICS Sector'] == sector_name, ['Carbon Intensity']].values.flatten().astype(float)

    # Baseline / optimal reduction
    c_b = float(w_bench @ c_vec)
    c_opt = float(w_opt @ c_vec)
    R_star = (c_b - c_opt) / c_b

    # TE cap (2% annual)
    te_annual = 0.02
    te_var_monthly_cap = (te_annual / np.sqrt(12))**2

    N = len(w_bench)

    # -------------------------------
    # A) ε-bands computation
    # -------------------------------
    def eps_constraints(w):
        return [
            cp.quad_form(w - w_bench, cp.psd_wrap(Sigma_sector)) <= te_var_monthly_cap,
            cp.sum(w) == 1,
            w >= 0,
            (c_b - c_vec @ w) / c_b >= (1 - EPS) * R_star
        ]

    bands = []
    for i, name in enumerate(stock_labels):
        # Max w_i
        w = cp.Variable(N)
        prob_max = cp.Problem(cp.Maximize(w[i]), eps_constraints(w))
        solve_qp_with_fallback(prob_max)
        if prob_max.status not in ("optimal", "optimal_inaccurate"):
            w_max = np.nan
        else:
            w_max = float(w.value[i])


        # Min w_i
        w = cp.Variable(N)
        prob_min = cp.Problem(cp.Minimize(w[i]), eps_constraints(w))
        solve_qp_with_fallback(prob_min)
        if prob_min.status not in ("optimal", "optimal_inaccurate"):
            w_min = np.nan
        else:
            w_min = float(w.value[i])
        # Hygiene
        if (not np.isfinite(w_min)) or (w_min < ZERO_TOL): w_min = 0.0
        if (not np.isfinite(w_max)) or (w_max < ZERO_TOL): w_max = 0.0
        w_min = float(min(max(w_min, 0.0), 1.0))
        w_max = float(min(max(w_max, 0.0), 1.0))

        bands.append([name, w_min, w_max, w_max - w_min])

    bands_df = pd.DataFrame(bands, columns=["symbol", "w_min", "w_max", "bandwidth"]).sort_values("bandwidth", ascending=False)

    # Quick ε-band stats
    avg_bw = float(bands_df["bandwidth"].mean())
    med_bw = float(bands_df["bandwidth"].median())
    max_bw = float(bands_df["bandwidth"].max())
    pct_wmin_zero = float((bands_df["w_min"] <= ZERO_TOL).mean()) * 100.0

    # -------------------------------
    # B) L2 lower bound
    # -------------------------------
    w = cp.Variable(N)
    v = cp.Parameter(N)

    constraints_same_obj = [
        cp.quad_form(w - w_bench, cp.psd_wrap(Sigma_sector)) <= te_var_monthly_cap + DELTA_TE,
        cp.sum(w) == 1,
        w >= 0,
        ((c_b - c_vec @ w) / c_b) >= R_star - DELTA_R,
        ((c_b - c_vec @ w) / c_b) <= R_star + DELTA_R
    ]
    prob_dir = cp.Problem(cp.Maximize(v @ w), constraints_same_obj)

    rng = np.random.default_rng(42)
    best_lb = 0.0
    stale = 0

    for it in range(K_DIRS):
        vv = rng.standard_normal(N)
        nrm = np.linalg.norm(vv)
        if nrm == 0:
            continue
        vv /= nrm

        # max v^T w
        v.value = vv
        solve_qp_with_fallback(prob_dir)
        cand1 = abs(float(vv @ (w.value - w_opt))) if (w.value is not None) else 0.0

        # max (-v)^T w
        v.value = -vv
        solve_qp_with_fallback(prob_dir)
        cand2 = abs(float((-vv) @ (w.value - w_opt))) if (w.value is not None) else 0.0

        new_best = max(best_lb, cand1, cand2)
        if new_best > best_lb + 1e-6:
            best_lb = new_best
            stale = 0
        else:
            stale += 1
            if stale >= NO_IMPROVE_PATIENCE:
                break

    l2_lower_bound = best_lb
    l1_turnover_upper = float(np.sqrt(N) * l2_lower_bound)

    # Extra stats
    hhi_b = float(np.sum(w_bench**2))
    hhi_o = float(np.sum(w_opt**2))
    top3 = bands_df.sort_values("bandwidth", ascending=False).head(3)[["symbol", "bandwidth"]]
    top3_list = [f"{r.symbol} ({r.bandwidth:.2%})" for r in top3.itertuples(index=False)]

    # -------------------------------
    # Summary row
    # -------------------------------
    summary_row = {
        "Sector": sector_name,
        "Names (N)": N,
        "90th pct ε-band": f"{bands_df['bandwidth'].quantile(0.9):.2%}",
        "% names at w_min=0": f"{pct_wmin_zero:.1f}%",
        "Top flexible names": "; ".join(top3_list),
        "Most pinned name": bands_df.loc[bands_df["bandwidth"].idxmin(), "symbol"],
        # Raw numeric fields for sorting
        "L2_lower_bound_same_obj": l2_lower_bound,
        "Avg_bandwidth": avg_bw,
        "Median_bandwidth": med_bw,
        "Max_bandwidth": max_bw,
        "Pct_wmin_zero": pct_wmin_zero,
    }

    return sector_name, bands_df, summary_row


def validate_data_quality(data, log_returns_all, optimal_portfolios, period_code):
    """
    Validate data quality before processing.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with carbon intensity
    log_returns_all : dict
        Log returns by sector
    optimal_portfolios : dict
        Optimal portfolios by sector
    period_code : str
        Period identifier

    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    print("🔍 Running data quality checks...")

    # Check 1: Validate dataset structure
    required_cols = ['GICS Sector', 'Carbon Intensity']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠️  ERROR: Missing required columns in dataset: {missing_cols}")
        return False

    # Check 2: Check for NaN values in carbon intensity
    nan_carbon = data['Carbon Intensity'].isna().sum()
    if nan_carbon > 0:
        print(f"⚠️  WARNING: Found {nan_carbon} NaN values in Carbon Intensity")

    # Check 3: Validate log returns data
    returns_issues = []
    for sector_name, sector_returns in log_returns_all.items():
        if 'Date' not in sector_returns.columns:
            returns_issues.append(f"{sector_name}: Missing 'Date' column")
            continue

        returns_data = sector_returns.drop(columns=['Date'])
        nan_count = returns_data.isna().sum().sum()

        if nan_count > 0:
            total_cells = returns_data.size
            nan_pct = (nan_count / total_cells) * 100
            returns_issues.append(f"{sector_name}: {nan_count} NaNs ({nan_pct:.2f}% of data)")

    if returns_issues:
        print(f"⚠️  WARNING: Found NaN values in log returns:")
        for issue in returns_issues[:5]:
            print(f"     • {issue}")
        if len(returns_issues) > 5:
            print(f"     • ... and {len(returns_issues) - 5} more sectors")
    else:
        print(f"✅ No NaN values found in log returns")

    # Check 4: Validate sectors match
    sectors_in_data = set(data['GICS Sector'].unique())
    sectors_in_returns = set(log_returns_all.keys())
    sectors_in_portfolios = set(optimal_portfolios.keys())

    missing_returns = sectors_in_portfolios - sectors_in_returns
    missing_data = sectors_in_portfolios - sectors_in_data

    if missing_returns:
        print(f"⚠️  WARNING: Sectors missing log returns: {missing_returns}")
    if missing_data:
        print(f"⚠️  WARNING: Sectors missing in dataset: {missing_data}")

    if not missing_returns and not missing_data:
        print(f"✅ All {len(sectors_in_portfolios)} sectors have matching data")

    print(f"✅ Data quality checks complete\n")
    return True


def build_flexibility_score(df):
    """
    Build flexibility scores from raw metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with L2 and bandwidth metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with added flexibility scores
    """
    df = df.copy()

    # Unnormalized absolute metrics
    df["Flexibility_Score_abs"] = 0.5 * df["L2_lower_bound_same_obj"] + 0.5 * df["Avg_bandwidth"]

    # Normalized version (0-1 scaling within period)
    def norm(x):
        x = pd.to_numeric(x, errors='coerce')
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if x_max - x_min == 0:
            return pd.Series(0.5, index=x.index)
        return (x - x_min) / (x_max - x_min)

    df["n_L2"] = norm(df["L2_lower_bound_same_obj"])
    df["n_AvgBW"] = norm(df["Avg_bandwidth"])
    df["Flexibility_Score"] = 0.5 * df["n_L2"] + 0.5 * df["n_AvgBW"]

    return df


def process_period(period_code, period_label):
    """
    Process a single time period.

    Parameters
    ----------
    period_code : str
        Period code (e.g., '0321')
    period_label : str
        Period label (e.g., 'Mar 2021')

    Returns
    -------
    tuple or None
        (sector_flexibility, sector_bands) if successful, None otherwise
    """
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING PERIOD: {period_label} ({period_code})")
    print(f"{'='*80}")

    # File paths
    optim_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period_code}.pkl"
    pickle_file = f"results/flexibility/sector_bands_{period_code}.pkl"
    excel_file = f"results/flexibility/l2_bandwidth_turnover_{period_code}.xlsx"
    dataset_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_code}.xlsx"
    returns_file = f"data/log_returns/sector_log_returns_comp_{period_code}.xlsx"

    # Check if required files exist
    missing_files = []
    for fname, fpath in [("optimal portfolios", optim_file),
                         ("dataset", dataset_file),
                         ("returns", returns_file)]:
        if not os.path.exists(fpath):
            missing_files.append(fname)

    if missing_files:
        print(f"⚠️  ERROR: Missing files: {', '.join(missing_files)}")
        print(f"   Skipping period {period_code}\n")
        return None

    # Load optimal frontier
    with open(optim_file, "rb") as f:
        optimal_portfolios_all_te = pickle.load(f)

    optimal_portfolios_shrink_2_TE = extract_optimal_portfolios_at_target_te(
        optimal_portfolios_all_te, target_te_bps=200
    )

    if len(optimal_portfolios_shrink_2_TE) == 0:
        print(f"⚠️  WARNING: No portfolios found at 2% TE for period {period_code}")
        print(f"   Skipping period {period_code}\n")
        return None

    # Use cache if available
    if os.path.exists(pickle_file) and os.path.exists(excel_file):
        print("✅ Using cached flexibility results.")
        with open(pickle_file, "rb") as f:
            sector_bands = pickle.load(f)
        sector_flexibility = pd.read_excel(excel_file)
    else:
        print("⚙️  Running fresh ε-band & L₂ optimization...")

        # Load data
        data = pd.read_excel(dataset_file)
        log_returns_all = pd.read_excel(returns_file, sheet_name=None)

        # Validate data quality
        if not validate_data_quality(data, log_returns_all, optimal_portfolios_shrink_2_TE, period_code):
            print(f"   Skipping period {period_code} due to data quality issues\n")
            return None

        # Process all sectors in parallel
        print(f"⚙️  Processing {len(optimal_portfolios_shrink_2_TE)} sectors...")
        results = Parallel(n_jobs=-1)(
            delayed(process_sector)(s, i, data, log_returns_all)
            for s, i in optimal_portfolios_shrink_2_TE.items()
        )

        sector_bands = {s: b for s, b, _ in results}
        summary_rows = [r for _, _, r in results]
        sector_flexibility = pd.DataFrame(summary_rows).sort_values(
            ["L2_lower_bound_same_obj", "Avg_bandwidth"],
            ascending=[False, False]
        )

        # Save cache
        os.makedirs("results/flexibility", exist_ok=True)
        sector_flexibility.to_excel(excel_file, index=False)
        with open(pickle_file, "wb") as f:
            pickle.dump(sector_bands, f)

        print(f"✅ Saved results for period {period_code}")

    # Add period and return
    sector_flexibility["Period"] = period_code
    return sector_flexibility, sector_bands


def validate_final_results(flexibility_score_df, periods):
    """
    Validate the final flexibility scores.

    Parameters
    ----------
    flexibility_score_df : pd.DataFrame
        Final flexibility scores
    periods : dict
        Dictionary of period codes
    """
    print("\n" + "="*80)
    print("🔍 FINAL VALIDATION")
    print("="*80)

    # Check 1: NaN values
    nan_scores = flexibility_score_df['Flexibility_Score'].isna().sum()
    if nan_scores > 0:
        print(f"⚠️  WARNING: {nan_scores} NaN values in Flexibility_Score")
        sectors_with_nan = flexibility_score_df[
            flexibility_score_df['Flexibility_Score'].isna()
        ][['Sector', 'Period']]
        print(f"   Affected sector-period combinations:")
        print(sectors_with_nan.to_string(index=False))
    else:
        print(f"✅ No NaN values in Flexibility_Score")

    # Check 2: Score range
    score_min = flexibility_score_df['Flexibility_Score'].min()
    score_max = flexibility_score_df['Flexibility_Score'].max()
    print(f"✅ Flexibility Score range: [{score_min:.4f}, {score_max:.4f}]")

    if score_min < 0 or score_max > 1:
        print(f"⚠️  WARNING: Flexibility scores outside expected [0, 1] range!")

    # Check 3: Sector count per period
    check = flexibility_score_df.groupby("Period")["Sector"].nunique()
    print("\n✅ Sector count per period:")
    print(check)

    # Check 4: Verify all periods
    expected_periods = set(periods.keys())
    actual_periods = set(flexibility_score_df['Period'].unique())
    missing_periods = expected_periods - actual_periods
    if missing_periods:
        print(f"\n⚠️  WARNING: Missing periods in final results: {missing_periods}")
    else:
        print(f"\n✅ All {len(expected_periods)} periods present in results")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to compute flexibility scores across all periods.
    """
    print("\n" + "="*80)
    print("FLEXIBILITY SCORE COMPUTATION")
    print("="*80)
    print(f"Processing {len(PERIODS)} time periods...")
    print(f"Parameters: ε={EPS}, K_dirs={K_DIRS}")
    print("="*80)

    # Create output directories
    os.makedirs("results/flexibility", exist_ok=True)

    # Process all periods
    all_flexibility_dfs = []
    all_sector_bands = {}

    for period_code, period_label in PERIODS.items():
        result = process_period(period_code, period_label)
        if result is not None:
            sector_flexibility, sector_bands = result
            all_flexibility_dfs.append(sector_flexibility)
            all_sector_bands[period_code] = sector_bands

    # Check if we have any results
    if len(all_flexibility_dfs) == 0:
        print("\n⚠️  ERROR: No flexibility data was computed for any period!")
        print("   Please check the errors above and ensure all required files exist.")
        return

    # Combine all periods
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)

    flexibility_panel = pd.concat(all_flexibility_dfs, ignore_index=True)

    print(f"✅ Combined flexibility data from {len(all_flexibility_dfs)} periods")
    print(f"   Total rows: {len(flexibility_panel)}")
    print(f"   Unique sectors: {flexibility_panel['Sector'].nunique()}")

    # Data cleaning
    flexibility_panel.columns = flexibility_panel.columns.str.strip()

    cols_to_fix = [c for c in flexibility_panel.columns
                   if "bandwidth" in c.lower() or "l2" in c.lower()]
    for col in cols_to_fix:
        flexibility_panel[col] = pd.to_numeric(flexibility_panel[col], errors="coerce")

    # Check combined data quality
    print("\n🔍 Checking combined data quality...")
    key_metrics = ['L2_lower_bound_same_obj', 'Avg_bandwidth', 'Median_bandwidth', 'Max_bandwidth']
    for metric in key_metrics:
        if metric in flexibility_panel.columns:
            nan_count = flexibility_panel[metric].isna().sum()
            if nan_count > 0:
                print(f"⚠️  WARNING: {nan_count} NaN values in {metric}")
            else:
                print(f"✅ No NaN values in {metric}")

    # Sort by period
    period_order = list(PERIODS.keys())
    flexibility_panel["Period"] = pd.Categorical(
        flexibility_panel["Period"], categories=period_order, ordered=True
    )
    flexibility_panel = flexibility_panel.sort_values(["Sector", "Period"])

    # Save panel
    flexibility_panel.to_excel("results/flexibility/sector_flexibility_raw.xlsx", index=False)
    print("\n✅ Saved all-period flexibility panel.")

    # Compute flexibility scores
    print("\n" + "="*80)
    print("COMPUTING FLEXIBILITY SCORES")
    print("="*80)

    scored_panel = (
        flexibility_panel
        .groupby("Period", group_keys=False, observed=True)
        .apply(build_flexibility_score)
        .reset_index(drop=True)
    )
    scored_panel.to_excel("results/flexibility/sector_flexibility_normalised.xlsx", index=False)
    flexibility_score_df = scored_panel[['Sector', 'Period', 'Flexibility_Score']].drop_duplicates()

    # Final validation
    validate_final_results(flexibility_score_df, PERIODS)

    # Save flexibility scores
    flexibility_score_output = "results/flexibility/flexibility_scores_by_period.xlsx"
    flexibility_score_df['Flexibility_Score'] = minmax_norm_grouped(flexibility_score_df, "Flexibility_Score")
    flexibility_score_df.to_excel(flexibility_score_output, index=False)
    print(f"\n✅ Flexibility scores saved to: {flexibility_score_output}")

    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    flexibility_score_df = flexibility_score_df.sort_values(by=["Period", "Sector"])

    # Plot normalized flexibility
    print("📈 Plotting normalized flexibility score...")
    plot_sector_evolution(
        scored_panel,
        value_col="Flexibility_Score",
        title="Normalized Flexibility Score (0–1) by Sector",
        ylabel="Flexibility Score (0–1)"
    )

    # Plot unnormalized flexibility
    print("📈 Plotting unnormalized flexibility score...")
    plot_sector_evolution(
        scored_panel,
        value_col="Flexibility_Score_abs",
        title="Unnormalized Flexibility Score by Sector",
        ylabel="Flexibility Score (Absolute Scale)"
    )

    print("\n" + "="*80)
    print("✅ FLEXIBILITY SCORE COMPUTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
