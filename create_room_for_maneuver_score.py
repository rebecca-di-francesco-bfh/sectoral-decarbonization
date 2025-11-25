"""
Room for Maneuver Score Calculation

This script computes metrics to characterize the carbon reduction efficient frontier
for each sector across all time periods. These metrics capture the "room for maneuver"
that each sector has in reducing carbon while staying within tracking error constraints.

Key Metrics:
- Slope at 2% TE: How steep is the frontier near the 2% tracking error level
- Elasticity at 2% TE: Percentage responsiveness of carbon cuts to TE changes
- AUC to 5% TE: Area under the curve (total carbon reduction potential)
- Max Cut at 5% TE: Maximum carbon reduction achievable at 5% tracking error
- TE for 50% Cut: Tracking error needed to achieve 50% of maximum carbon reduction

Output:
- CSV file with all metrics for each sector-period combination
- Can be used for visualization and analysis of sector decarbonization capacity
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_functions import plot_sector_evolution

# =============================================================================
# STEP 1: LOAD SECTOR VOLATILITIES
# =============================================================================
# We need sector volatilities to normalize the room for maneuver scores.
# This accounts for the fact that high-volatility sectors naturally have
# more "room" to deviate from the benchmark.

print("Loading sector volatilities...")

# Load precomputed annualized sector volatilities from benchmark replication
vol_df = pd.read_excel("data/benchmark_returns_volatility/sector_annualized_volatility_2021_2023.xlsx")

# Ensure proper column naming (Excel may add an index column)
if 'Unnamed: 0' in vol_df.columns:
    vol_df = vol_df.rename(columns={'Unnamed: 0': 'Sector'})

# The volatility values are already in the correct format (annualized)
print(f"✓ Loaded volatilities for {len(vol_df)} sectors")
print(f"   Volatility range: {vol_df.iloc[:, 1].min():.4f} to {vol_df.iloc[:, 1].max():.4f}\n")

# Rename column to standard name if needed
if vol_df.columns[1] != 'Sector Volatility':
    vol_df.columns = ['Sector', 'Sector Volatility']


# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================

# Define all periods to analyze
periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

# Tracking error levels for metric computation
TE_TARGET = 0.02      # 2% tracking error (common benchmark)
TE_AUC_MAX = 0.05     # 5% tracking error (maximum for AUC calculation)
TE_WIN_BPS = 25       # Window size in basis points for slope/elasticity (±0.25%)


# =============================================================================
# STEP 3: HELPER FUNCTIONS FOR FRONTIER ANALYSIS
# =============================================================================

def _interp(te0, te_grid, c_grid):
    """
    Interpolate carbon reduction at a specific tracking error level.

    Args:
        te0: Target tracking error level
        te_grid: Array of tracking error values (x-axis of frontier)
        c_grid: Array of carbon reduction values (y-axis of frontier)

    Returns:
        Interpolated carbon reduction at te0
    """
    return float(np.interp(np.clip(te0, te_grid[0], te_grid[-1]), te_grid, c_grid))


def _prep_frontier(te_bps, cuts_pct):
    """
    Prepare and clean the efficient frontier data.

    This function:
    1. Converts tracking error from basis points to decimal (e.g., 200 bps -> 0.02)
    2. Converts carbon reduction from percentage to decimal (e.g., 50% -> 0.50)
    3. Removes invalid (NaN or inf) points
    4. Sorts by tracking error
    5. Collapses duplicate TE values by taking the maximum carbon reduction

    Args:
        te_bps: Tracking errors in basis points (e.g., [20, 50, 100, ...])
        cuts_pct: Carbon reductions in percentage (e.g., [5.2, 12.3, 18.7, ...])

    Returns:
        tuple: (te_grid, c_grid) both as sorted, cleaned arrays in decimal form
    """
    # Convert to decimals
    te = np.asarray(te_bps, float) / 10000.0
    c = np.asarray(cuts_pct, float) / 100.0

    # Remove invalid points
    m = np.isfinite(te) & np.isfinite(c)
    te, c = te[m], c[m]

    # Sort by tracking error
    order = np.argsort(te)
    te, c = te[order], c[order]

    # Collapse duplicate TE values (keep maximum carbon reduction)
    uniq_te = np.unique(te)
    uniq_c = [np.nanmax(c[te == t]) for t in uniq_te]

    return uniq_te, np.array(uniq_c)


def _slope_at(te_grid, c_grid, te0, win_bps=25):
    """
    Calculate the slope of the frontier at a specific tracking error level.

    Uses a symmetric window around te0 to estimate the derivative:
    slope ≈ (c(te0 + h) - c(te0 - h)) / (2h)

    This measures how much additional carbon reduction is gained per unit
    increase in tracking error at the specified point.

    Args:
        te_grid: Array of tracking error values
        c_grid: Array of carbon reduction values
        te0: Point at which to calculate slope
        win_bps: Window size in basis points (default 25 = ±0.25%)

    Returns:
        Slope at te0, or NaN if calculation fails
    """
    h = win_bps / 10000.0  # Convert basis points to decimal

    # Define window boundaries
    tl, tr = np.clip([te0 - h, te0 + h], te_grid[0], te_grid[-1])

    if tr <= tl:
        return np.nan

    # Calculate slope using interpolated values
    return (_interp(tr, te_grid, c_grid) - _interp(tl, te_grid, c_grid)) / (tr - tl)


def _elasticity_at(te_grid, c_grid, te0, win_bps=25):
    """
    Calculate the elasticity of carbon reduction with respect to tracking error.

    Elasticity = (dC/dTE) * (TE/C)

    This measures the percentage change in carbon reduction for a 1% change
    in tracking error. High elasticity means the sector is very responsive to
    allowing more tracking error.

    Args:
        te_grid: Array of tracking error values
        c_grid: Array of carbon reduction values
        te0: Point at which to calculate elasticity
        win_bps: Window size in basis points

    Returns:
        Elasticity at te0, or NaN if calculation fails
    """
    c0 = _interp(te0, te_grid, c_grid)
    s = _slope_at(te_grid, c_grid, te0, win_bps)

    if c0 <= 0 or not np.isfinite(s):
        return np.nan

    return s * (te0 / c0)


def _auc_to(te_grid, c_grid, te_max):
    """
    Calculate the area under the efficient frontier curve up to te_max.

    This measures the total carbon reduction potential available across
    all tracking error levels from 0 to te_max. Higher AUC means more
    "room for maneuver" in reducing carbon.

    Args:
        te_grid: Array of tracking error values
        c_grid: Array of carbon reduction values
        te_max: Maximum tracking error for AUC calculation

    Returns:
        Area under the curve from 0 to te_max
    """
    # Create evaluation points (include 0, all frontier points, and te_max)
    xs = np.unique(np.clip(np.r_[0.0, te_grid, te_max], 0, te_max))

    # Interpolate carbon reduction at each evaluation point
    ys = np.array([_interp(x, te_grid, c_grid) for x in xs])

    # Calculate area using trapezoidal integration
    return float(np.trapz(ys, xs))


def _te_for_cut(te_grid, c_grid, cut_frac):
    """
    Find the tracking error needed to achieve a specific carbon reduction fraction.

    This inverts the frontier: given a desired carbon reduction (as a fraction
    of the maximum), what tracking error is required?

    Args:
        te_grid: Array of tracking error values
        c_grid: Array of carbon reduction values (normalized 0-1)
        cut_frac: Target carbon reduction fraction (e.g., 0.50 for 50% of max)

    Returns:
        Tracking error needed to achieve cut_frac, or NaN if not achievable
    """
    # Check if target is achievable
    if c_grid[0] >= cut_frac:
        return 0.0  # Already achieved at minimum TE
    if c_grid[-1] < cut_frac:
        return np.nan  # Not achievable even at maximum TE

    # Find bracketing points
    idx = np.searchsorted(c_grid, cut_frac)
    if idx == 0:
        return te_grid[0]

    # Linear interpolation between bracketing points
    x0, x1 = te_grid[idx-1], te_grid[idx]
    y0, y1 = c_grid[idx-1], c_grid[idx]

    return float(x0 + (cut_frac - y0) * (x1 - x0) / (y1 - y0))


def _minmax(x):
    """
    Min-max normalization: scale values to [0, 1] range.

    Args:
        x: Array of values to normalize

    Returns:
        Normalized array, or array of NaN if normalization not possible
    """
    x = np.asarray(x, float)

    if len(x) < 2 or np.all(~np.isfinite(x)):
        return np.full_like(x, np.nan)

    lo, hi = np.nanmin(x), np.nanmax(x)

    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

# within period normalization 0-1
def norm(x):
    x = pd.to_numeric(x, errors='coerce')
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if x_max - x_min == 0:
        return pd.Series(0.5, index=x.index)
    return (x - x_min) / (x_max - x_min)

# =============================================================================
# STEP 4: COMPUTE METRICS FOR ALL SECTORS AND PERIODS
# =============================================================================

print("Computing frontier metrics for all sectors and periods...")

records = []

for period in periods:
    fname = f"results/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"

    if not os.path.exists(fname):
        print(f"⚠️  Missing file: {fname}")
        continue

    # Load optimization results for this period
    with open(fname, "rb") as f:
        sector_data = pickle.load(f)

    # Process each sector
    for sector, d in sector_data.items():
        te_bps = d.get("tracking_errors", [])
        c_pct = d.get("carbon_reductions", [])

        # Skip if insufficient data points
        if len(te_bps) < 3:
            continue

        # Prepare and clean the efficient frontier
        te, c = _prep_frontier(te_bps, c_pct)

        # Ensure TE is in decimal form (0.02, 0.05, etc.)
        te_frac = te / 10000 if np.nanmax(te) > 1 else te

        # Normalize carbon reduction to [0, 1] range for some metrics
        c_max = np.nanmax(c)
        c_norm = c / c_max if c_max > 0 else c

        # Compute metrics on the normalized frontier
        slope2 = _slope_at(te_frac, c_norm, TE_TARGET, TE_WIN_BPS)
        elast2 = _elasticity_at(te_frac, c_norm, TE_TARGET, TE_WIN_BPS)
        auc5 = _auc_to(te_frac, c_norm, TE_AUC_MAX) / TE_AUC_MAX
        te50 = _te_for_cut(te_frac, c_norm, 0.50)

        # Use raw carbon reduction (in %) for maximum cut at 5% TE
        max5 = _interp(TE_AUC_MAX, te_frac, c)

        # Store results
        records.append({
            "Sector": sector,
            "Period": period,
            "Slope_at_2pct": slope2,
            "Elasticity_at_2pct": elast2,
            "AUC_to_5pctTE": auc5,
            "MaxCut_at_5pctTE": max5,
            "TE_for_50pctCut": te50
        })

# Build results dataframe
df = pd.DataFrame(records)

print(f"\nComputed metrics for {len(df)} sector-period combinations")
print(f"Sectors: {df['Sector'].nunique()}")
print(f"Periods: {df['Period'].nunique()}\n")

# Display sample results
print("Sample results:")
print(df.head(10))


# =============================================================================
# STEP 5: SAVE RESULTS
# =============================================================================

output_file = "results/room_for_maneuver/room_for_maneuver_metrics.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")


# =============================================================================
# STEP 6: COMPUTE ROOM FOR MANEUVER SCORE
# =============================================================================
# Combine the individual metrics into a single "Room for Maneuver" score.
# This score represents the overall capacity of each sector to reduce carbon
# while maintaining reasonable tracking error.

print("\nComputing Room for Maneuver Score...")

# Create a copy for normalization
df_norm = df.copy()

# Normalize each metric to [0, 1] range across all sectors and periods
metrics_to_normalize = ['Slope_at_2pct', 'Elasticity_at_2pct', 'AUC_to_5pctTE', 'MaxCut_at_5pctTE']

for metric in metrics_to_normalize:
    df_norm[f'{metric}_norm'] = _minmax(df_norm[metric].values)

# Invert TE_for_50pctCut: lower TE needed is better (higher score)
# First normalize, then invert (1 - normalized_value)
df_norm['TE_for_50pctCut_norm'] = 1 - _minmax(df_norm['TE_for_50pctCut'].values)

# Compute composite Room for Maneuver score as average of normalized metrics
# Equal weighting: each metric contributes 20% to the final score
df_norm['Room_for_Maneuver_Score'] = (
    df_norm['Slope_at_2pct_norm'] * 0.20 +
    df_norm['Elasticity_at_2pct_norm'] * 0.20 +
    df_norm['AUC_to_5pctTE_norm'] * 0.20 +
    df_norm['MaxCut_at_5pctTE_norm'] * 0.20 +
    df_norm['TE_for_50pctCut_norm'] * 0.20
)

# Adjust by sector volatility to account for inherent risk differences
df_norm = df_norm.merge(vol_df, on="Sector", how="left")
df_norm["Room_for_Maneuver_Score_Adjusted"] = (
    df_norm["Room_for_Maneuver_Score"] / df_norm["Sector Volatility"]
)

print(f"✓ Room for Maneuver Score computed for all sectors and periods\n")




# =============================================================================
# STEP 7: PLOT ROOM FOR MANEUVER SCORE EVOLUTION
# =============================================================================

print("\nGenerating plots...")

# # Plot 1: Raw Room for Maneuver Score
# plot_sector_evolution(
#     df_norm,
#     value_col="Room_for_Maneuver_Score",
#     title="Room for Maneuver Score Evolution by Sector",
#     ylabel="Room for Maneuver Score (0-1)"
# )

# # Plot 2: Volatility-Adjusted Room for Maneuver Score
# plot_sector_evolution(
#     df_norm,
#     value_col="Room_for_Maneuver_Score_Adjusted",
#     title="Volatility-Adjusted Room for Maneuver Score Evolution by Sector",
#     ylabel="Adjusted Room for Maneuver Score"
# )



print("\n✓ Script completed successfully")
# Save the normalized dataframe with scores
output_file_scores = "results/room_for_maneuver/room_for_maneuver_scores_by_period.xlsx"



df_norm["Room_for_Maneuver_Score"] = df_norm['Room_for_Maneuver_Score_Adjusted']
# Sort by period
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

period_order = list(PERIODS.keys())
df_norm["Period"] = pd.Categorical(
    df_norm["Period"], categories=period_order, ordered=True
)
df_norm = df_norm.sort_values(["Sector", "Period"])

def normalize_within_period(df_norm):
    df_norm['Room_for_Maneuver_Score'] = norm(df_norm['Room_for_Maneuver_Score'])
    return df_norm

scored_panel = (
        df_norm
        .groupby("Period", group_keys=False, observed=True)
        .apply(normalize_within_period)
        .reset_index(drop=True)
    )
print(scored_panel)

# Plot 3: Normalized 
plot_sector_evolution(
    scored_panel,
    value_col="Room_for_Maneuver_Score",
    title="Normalized Volatility-Adjusted Room for Maneuver Score Evolution by Sector",
    ylabel="Room_for_Maneuver_Score"
)
scored_panel[['Sector', 'Period', 'Room_for_Maneuver_Score']].to_excel(output_file_scores, index=False)