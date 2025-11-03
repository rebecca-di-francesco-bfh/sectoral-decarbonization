"""
Optimal Portfolio Creation for Carbon Reduction with Tracking Error Constraints

This script performs portfolio optimization across different sectors to minimize carbon
intensity while constraining tracking error relative to a benchmark. It generates an
efficient frontier showing the trade-off between tracking error and carbon reduction.

Key concepts:
- Tracking Error (TE): Measure of how much a portfolio deviates from its benchmark
- Carbon Intensity: Carbon emissions per unit of revenue
- Efficient Frontier: Set of optimal portfolios offering the best carbon reduction for each TE level
"""

from sklearn.covariance import LedoitWolf
import os
import pickle
import cvxpy as cp
import numpy as np
import pandas as pd

#  TE - CARBON FRONTIERS OPTIMIZATION

# --- Utility functions ---

def nearest_psd(A):
    """
    Convert a symmetric matrix to the nearest positive semi-definite (PSD) matrix.

    This is necessary because numerical errors can sometimes produce covariance matrices
    with slightly negative eigenvalues, which violates the PSD requirement for covariance.

    Method:
    1. Compute eigendecomposition of matrix A
    2. Set any negative eigenvalues to zero
    3. Reconstruct the matrix using the corrected eigenvalues

    Args:
        A: Square symmetric matrix (typically a covariance matrix)

    Returns:
        Nearest PSD matrix to A
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0  # Replace negative eigenvalues with zero
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def sigma_raw_fn(R_clean):
    """
    Calculate the raw (sample) covariance matrix from returns data.

    Args:
        R_clean: DataFrame of clean returns data (no NaN values)

    Returns:
        Sample covariance matrix
    """
    return R_clean.cov()


def sigma_shrink_fn(R_clean):
    """
    Calculate a shrunk covariance matrix using Ledoit-Wolf shrinkage estimator.

    Ledoit-Wolf shrinkage improves covariance estimation by shrinking the sample
    covariance towards a structured estimator (identity matrix). This is particularly
    useful when the number of observations is small relative to the number of assets.

    Benefits:
    - More stable estimates (especially with limited data)
    - Better conditioning (avoids numerical issues)
    - Improved out-of-sample performance

    Args:
        R_clean: DataFrame of clean returns data (no NaN values)

    Returns:
        tuple: (Regularized PSD covariance matrix, shrinkage intensity alpha)
    """
    lw = LedoitWolf().fit(R_clean)
    Sigma_shrink = lw.covariance_

    # Add small regularization term to diagonal for numerical stability
    Sigma_reg = Sigma_shrink + 1e-5 * np.eye(Sigma_shrink.shape[0])

    # Ensure PSD property
    return nearest_psd(Sigma_reg), lw.shrinkage_


def run_sector_optimisation(sector_name, sector, R, cov_type="raw", cache_dir="cache"):
    """
    Run portfolio optimization for a single sector to find the efficient frontier
    of tracking error vs. carbon reduction.

    This function solves a series of quadratic programming problems:

    Minimize: carbon_intensity · weights
    Subject to:
        - Tracking error² ≤ TE_cap²
        - Sum(weights) = 1
        - weights ≥ 0

    The tracking error is varied across a range to trace out the efficient frontier.

    Args:
        sector_name: Name of the sector (e.g., "Energy", "Financials")
        sector: DataFrame containing sector data with columns:
                - 'weight_in_sector': Benchmark weights
                - 'Carbon Intensity': Carbon intensity of each stock
        R: DataFrame of log returns for stocks in this sector
        cov_type: Type of covariance estimation - "raw" or "shrink"
        cache_dir: Directory to cache results (to avoid re-computation)

    Returns:
        Dictionary containing:
            - diagnostics: Summary statistics about the optimization
            - weights_by_te: Optimal weights for each TE level
            - tracking_errors: Realized tracking errors (in basis points)
            - carbon_reductions: Carbon reduction percentages
            - w_bench: Benchmark weights
            - stock_labels: Stock identifiers
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{sector_name}_{cov_type}.pkl"

    # --- Load from cache if exists ---
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- Prepare inputs ---
    w_bench = sector['weight_in_sector'].values  # Benchmark weights
    c_vec = sector['Carbon Intensity'].values     # Carbon intensity vector

    # Validate inputs
    assert np.isclose(w_bench.sum(), 1.0), "Benchmark weights must sum to 1"
    assert not np.isnan(c_vec).any(), "Carbon intensity values cannot be NaN"

    # Clean returns data (remove Date column and any rows with NaN)
    R_clean = R.drop(columns=['Date']).dropna()
    stock_labels = R_clean.columns

    # --- Calculate covariance matrix ---
    if cov_type == "raw":
        Sigma = sigma_raw_fn(R_clean)
        shrinkage_alpha = None
    elif cov_type == "shrink":
        Sigma, shrinkage_alpha = sigma_shrink_fn(R_clean)
    else:
        raise ValueError("cov_type must be 'raw' or 'shrink'")

    # --- Covariance matrix diagnostics ---
    # Check eigenvalues to assess numerical stability
    eigvals = np.linalg.eigvalsh(Sigma)
    smallest_eigs = np.sort(eigvals)[:5]  # Track 5 smallest eigenvalues
    rank = np.linalg.matrix_rank(Sigma)
    num_features = Sigma.shape[1]
    N = len(w_bench)

    # --- Run optimization across multiple tracking error levels ---
    # Create a grid of annual tracking error caps from 0.2% to 5%
    te_caps_annual = np.linspace(0.002, 0.05, 100)

    # Ensure 2% tracking error is included (common industry benchmark)
    if not np.isclose(te_caps_annual, 0.02).any():
        te_caps_annual = np.sort(np.append(te_caps_annual, 0.02))

    # Initialize storage for results
    tracking_errors, carbon_reductions, weights_by_te = [], [], []

    # Loop over each tracking error level
    for te_annual in te_caps_annual:
        # Convert annual TE to monthly variance
        # TE_monthly = TE_annual / sqrt(12)
        # Variance = TE²
        te_cap_var_monthly = (te_annual / np.sqrt(12)) ** 2

        # Define optimization variables
        w = cp.Variable(N)  # Portfolio weights to optimize

        # Tracking error constraint: (w - w_bench)' Σ (w - w_bench) ≤ TE_cap²
        tracking_error = cp.quad_form(w - w_bench, cp.psd_wrap(Sigma))

        # Define constraints
        constraints = [
            tracking_error <= te_cap_var_monthly,  # TE constraint
            cp.sum(w) == 1,                         # Weights sum to 1
            w >= 0                                   # Long-only constraint
        ]

        # Objective: Minimize portfolio carbon intensity
        prob = cp.Problem(cp.Minimize(c_vec @ w), constraints)

        # Solve the optimization problem - try ECOS first
        prob.solve(solver=cp.ECOS, verbose=False)

        # If solution is inaccurate, try SCS solver (more robust but slower)
        if prob.status == "optimal_inaccurate":
            print(f"    [{sector_name} - {cov_type}] ECOS inaccurate at TE={te_annual*100:.2f}%, trying SCS solver...")
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
            if prob.status == "optimal":
                print(f"    [{sector_name} - {cov_type}] SCS solved successfully")

        # Check if solution is valid
        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            print(f"    [{sector_name} - {cov_type}] No solution found at TE={te_annual*100:.2f}% (status: {prob.status})")
            continue  # Skip this TE level if no solution found

        # Extract optimal weights
        w_opt = w.value

        # Check constraint violations
        weights_sum_violation = abs(w_opt.sum() - 1.0)
        negative_weights_count = (w_opt < -1e-6).sum()

        # Warn if significant violations detected
        if weights_sum_violation > 1e-4:
            print(f"    [{sector_name} - {cov_type}] Warning: weights sum violation = {weights_sum_violation:.2e} at TE={te_annual*100:.2f}%")
        if negative_weights_count > 0:
            print(f"    [{sector_name} - {cov_type}] Warning: {negative_weights_count} negative weights at TE={te_annual*100:.2f}%")
            # Clip negative weights to zero
            w_opt = np.maximum(w_opt, 0)
            # Renormalize to sum to 1
            w_opt = w_opt / w_opt.sum()

        # Calculate realized tracking error (annualized, in basis points)
        diff = w_opt - w_bench
        te_real = np.sqrt(diff.T @ Sigma @ diff) * np.sqrt(12)

        # Calculate carbon reduction
        carbon_b = w_bench @ c_vec      # Benchmark carbon intensity
        carbon_opt = w_opt @ c_vec      # Optimized carbon intensity
        reduction_pct = (carbon_b - carbon_opt) / carbon_b * 100  # Reduction percentage

        # Store results (TE in basis points: multiply by 10,000)
        tracking_errors.append(te_real * 10000)
        carbon_reductions.append(reduction_pct)
        weights_by_te.append(w_opt)

    # --- Calculate summary statistics ---
    tracking_errors_np = np.array(tracking_errors)
    carbon_reductions_np = np.array(carbon_reductions)

    # Carbon reduction at lowest and highest TE
    start_red, end_red = carbon_reductions_np[0], carbon_reductions_np[-1]

    # Find carbon reduction at 2% tracking error (200 basis points)
    idx_2pct = np.argmin(np.abs(tracking_errors_np - 200))
    reduction_at_2pct = carbon_reductions_np[idx_2pct]

    # Compile diagnostics
    diagnostics = {
        "Sector": sector_name,
        "Num Features": num_features,                    # Number of stocks in sector
        "Covariance": cov_type,                          # Raw or shrunk covariance
        "Rank": rank,                                     # Matrix rank
        "Min_Eigval1": smallest_eigs[0],                 # Smallest eigenvalue
        "Min_Eigval2": smallest_eigs[1] if len(smallest_eigs) > 1 else np.nan,
        "Min_Eigval3": smallest_eigs[2] if len(smallest_eigs) > 2 else np.nan,
        "Low Rank?": rank < num_features,                # Is covariance matrix low rank?
        "Not PSD?": eigvals.min() < -1e-6,              # Is covariance not PSD?
        "Shrinkage Alpha": shrinkage_alpha,              # Ledoit-Wolf shrinkage intensity
        "Start Reduction (%)": round(start_red, 2),      # Carbon reduction at min TE
        "End Reduction (%)": round(end_red, 2),          # Carbon reduction at max TE
        "Gain (%)": round(end_red - start_red, 2),      # Total gain across TE range
        "Reduction @2% TE (%)": round(reduction_at_2pct, 2),  # Reduction at 2% TE
    }

    # Package results
    result = {
        "sector_name": sector_name,
        "cov_type": cov_type,
        "diagnostics": diagnostics,
        "weights_by_te": weights_by_te,           # Optimal weights for each TE level
        "tracking_errors": tracking_errors,        # Realized TEs (basis points)
        "carbon_reductions": carbon_reductions,    # Carbon reduction percentages
        "w_bench": w_bench,                        # Benchmark weights
        "stock_labels": list(stock_labels)         # Stock identifiers
    }

    # --- Save to cache ---
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


def run_period(period_tag):
    """
    Run portfolio optimization for all sectors in a given time period.

    This function:
    1. Loads benchmark weights and carbon intensity data
    2. Loads log returns for each sector
    3. Runs optimization for each sector using both raw and shrunk covariance
    4. Saves results and diagnostics

    Args:
        period_tag: Time period identifier (e.g., "0321" for March 2021)

    Returns:
        DataFrame of diagnostics for all sectors in this period
    """
    print(f"=== Running period {period_tag} ===")

    # Set up cache directory for this period
    cache_dir = f"data/covariances/{period_tag}"
    os.makedirs(cache_dir, exist_ok=True)

    # Load data files
    data_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_tag}.xlsx"
    log_file  = f"data/log_returns/sector_log_returns_comp_{period_tag}.xlsx"

    data = pd.read_excel(data_file)
    log_returns_all = pd.read_excel(log_file, sheet_name=None)  # Load all sheets (one per sector)

    results = []

    # Process each sector
    for sector_name in data['GICS Sector'].unique():
        print(sector_name)
        sector = data[data['GICS Sector'] == sector_name]
        R = log_returns_all[sector_name]

        # Run optimization with raw covariance
        res_raw = run_sector_optimisation(sector_name, sector, R, cov_type="raw", cache_dir=cache_dir)
        results.append(res_raw)

        # Run optimization with shrunk covariance (Ledoit-Wolf)
        res_shrink = run_sector_optimisation(sector_name, sector, R, cov_type="shrink", cache_dir=cache_dir)
        results.append(res_shrink)

    # --- Combine diagnostics into a summary DataFrame ---
    diagnostics_df = pd.DataFrame([r["diagnostics"] for r in results])

    # --- Combine full results into a dictionary ---
    combined_results = {
        r["sector_name"]: {
            "cov_type": r["cov_type"],
            "diagnostics": r["diagnostics"],
            "weights_by_te": r["weights_by_te"],
            "tracking_errors": r["tracking_errors"],
            "carbon_reductions": r["carbon_reductions"],
            "w_bench": r.get("w_bench"),
            "stock_labels": r.get("stock_labels")
        }
        for r in results
    }

    # --- Save results to pickle file ---
    out_path = f"results/optimal_portfolios/optimal_portfolios_all_te_{period_tag}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(combined_results, f)
    print(f"Saved results for {period_tag} to {out_path}")

    return diagnostics_df


# --- Main execution ---

# Define all periods to analyze (quarterly from March 2021 to December 2023)
periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222", "0323", "0623", "0923", "1223"]

all_diagnostics = []

# Run optimization for each period
for tag in periods:
    diag = run_period(tag)
    all_diagnostics.append(diag)

# Combine all diagnostics into a single summary DataFrame
all_diag_df = pd.concat(all_diagnostics, keys=periods, names=["Period", "Index"]).reset_index()

# Save comprehensive diagnostics summary
all_diag_df.to_excel("results/optimal_portfolios/diagnostics_summary_all_periods.xlsx", index=False)
print("Finished all periods.")
