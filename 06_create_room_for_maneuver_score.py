"""
Room for Maneuver Score Calculation

This script computes metrics to characterize the carbon reduction efficient frontier
for each sector across all time periods. These metrics capture the "room for maneuver"
that each sector has in reducing carbon while staying within tracking error constraints.

Key Metrics:
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
from scipy.stats import spearmanr

def minmax_within_period(x):
    """
    Min-max normalize a pandas Series within a single period group.

    Intended for use with groupby().transform(). If all values are equal or
    all NaN, returns 0.5 for all elements (neutral mid-point).

    Args:
        x: pandas Series of numeric values for one period

    Returns:
        Normalized Series in [0, 1] (or 0.5 if normalization is not possible)
    """
    x = pd.to_numeric(x, errors="coerce")
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        # if all equal or all NaN, return 0.5 for neutrality
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)

def _carbon_weight_alignment(w_bench, carbon_intensity):
    """
    Measures alignment between benchmark weights and benchmark carbon contributions.

    High value  → carbon concentrated in large weights (bad for low-TE decarb)
    Low value   → carbon concentrated in small weights (good for low-TE decarb)

    Returns:
        Spearman rank correlation in [-1, 1]
    """
    w = np.asarray(w_bench, float)
    c = np.asarray(carbon_intensity, float)

    m = np.isfinite(w) & np.isfinite(c)
    w, c = w[m], c[m]

    if len(w) < 3:
        return np.nan

    carbon_contrib = w * c

    corr, _ = spearmanr(w, carbon_contrib)
    return corr


# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================

# Define all periods to analyze
periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

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




# =============================================================================
# STEP 3: COMPUTE METRICS FOR ALL SECTORS AND PERIODS
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
        te_frac = te
        print(te_frac)

        # Normalize carbon reduction to [0, 1] range for some metrics
        c_max = np.nanmax(c)
        c_norm = c / c_max if c_max > 0 else c

       # --- Frontier-based metrics ---
        c_at_1pct = _interp(0.01, te_frac, c_norm)
        te50 = _te_for_cut(te_frac, c_norm, 0.50)
   

        # --- Structural metric: carbon concentration ---
        w_bench = d.get("w_bench")
        print(w_bench)
        carbon_intensity = d.get("carbon_intensity")
        print(d)
        alignment = _carbon_weight_alignment(
        w_bench=w_bench,
        carbon_intensity=carbon_intensity
        )
        records.append({
        "Sector": sector,
        "Period": period,
        "C_at_1pct": c_at_1pct,
        "TE_for_50pctCut": te50,
        "Alignment": alignment
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
# STEP 4: COMPUTE ROOM FOR MANEUVER SCORE
# =============================================================================
# Combine the individual metrics into a single "Room for Maneuver" score.
# This score represents the overall capacity of each sector to reduce carbon
# while maintaining reasonable tracking error.

print("\nComputing Room for Maneuver Score...")

df_norm = df.copy()

# Invert "bad-is-high" metrics *before* normalization (cleaner interpretation)
df_norm["TE_for_50pctCut_inv"] = -df_norm["TE_for_50pctCut"]   # lower TE50 is better
df_norm["Alignment_inv"]       = -df_norm["Alignment"]         # lower alignment is better

# Normalize each metric within each period
df_norm["C_at_1pct_norm"] = df_norm.groupby("Period")["C_at_1pct"].transform(minmax_within_period)
df_norm["TE50_norm"]      = df_norm.groupby("Period")["TE_for_50pctCut_inv"].transform(minmax_within_period)
df_norm["Align_norm"]     = df_norm.groupby("Period")["Alignment_inv"].transform(minmax_within_period)

# Average (already in [0,1], so no second normalization needed)
df_norm["Room_for_Maneuver_Score"] = (df_norm["C_at_1pct_norm"] + df_norm["TE50_norm"] + df_norm["Align_norm"]) / 3

# =============================================================================
# STEP 5: PLOT ROOM FOR MANEUVER SCORE EVOLUTION
# =============================================================================

print("\nGenerating plots...")




print("\n✓ Script completed successfully")
# # Save the normalized dataframe with scores


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

# Plot metric 1: carbon reduction achievable at 1% TE
plot_sector_evolution(
    df_norm,
    value_col='C_at_1pct',
    title="Carbon reduction at 1% TE",
    ylabel="Carbon reduction at 1% TE"
)


plot_sector_evolution(
    df_norm,
    value_col='Alignment',
    title="Carbon–Weight Alignment (Structural Accessibility)",
    ylabel="Spearman corr(weight, carbon contribution)"
)

plot_sector_evolution(
    df_norm,
    value_col='TE_for_50pctCut',
    title="TE required for 50% of maximum decarbonizationr",
    ylabel="TE required for 50% of maximum decarbonization"
)


# Plot room for maneuver score
plot_sector_evolution(
    df_norm,
    value_col='Room_for_Maneuver_Score',
    title="Room for Maneuver Score",
    ylabel="Room_for_Maneuver_Score"
)


out_cols = [
    "Sector", "Period",
    "Room_for_Maneuver_Score",
    "C_at_1pct",
    "Alignment",
    "TE_for_50pctCut",
]

out_xlsx = "results/room_for_maneuver/room_for_maneuver_scores_by_period.xlsx"
out_parq = "results/room_for_maneuver/room_for_maneuver_scores_by_period.parquet"

out_df = df_norm[out_cols].copy()

out_df.to_excel(out_xlsx, index=False)
out_df.to_parquet(out_parq, index=False)   # ✅ add this line

print(f"✅ Saved: {out_xlsx}")
print(f"✅ Saved: {out_parq}")