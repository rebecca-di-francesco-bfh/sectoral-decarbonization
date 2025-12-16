"""
Sensitivity Score Computation for Optimal Portfolio Analysis

This script computes sensitivity scores by analyzing how optimal portfolios
respond to perturbations in input data (returns).

Author: IRP17 Team
"""

import os
import pickle
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from utils import extract_optimal_portfolios_at_target_te, solve_qp_with_fallback
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def l1_turnover_pct(w_from, w_to):
    """One-way turnover (% of portfolio) to go from w_from → w_to."""
    return 0.5 * float(np.abs(w_to - w_from).sum()) * 100.0

def cosine_similarity(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def realized_te_annual(w, w_b, Sigma):
    """Annualized TE in decimal (e.g., 0.02 == 2%)."""
    diff = (w - w_b)
    te_m = float(diff.T @ Sigma @ diff)
    te_a = np.sqrt(te_m) * np.sqrt(12.0)
    return te_a

def carbon_reduction_pct(w, w_b, c_vec):
    """% reduction relative to benchmark carbon; returns in percent units."""
    cb = float(np.dot(w_b, c_vec))
    co = float(np.dot(w,   c_vec))
    if cb == 0:
        return np.nan
    return (cb - co) / cb * 100.0

def nanpercentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else np.nan

def sensitivity_kpis_from_trials(
    w_trials,             # (n_trials, N) weights from perturbations (optimized each trial)
    te_trials_annual,     # (n_trials,) TE in decimal for each trial (annualized)
    R_clean,              # baseline returns (DataFrame without Date)
    w_bench,              # benchmark weights (N,)
    c_vec,                # carbon intensity per name (N,)
    Sigma_fn,             # function R_clean -> Sigma
    te_cap=0.02,          # baseline TE cap (annual)
    w_opt0=None           # baseline optimized weights; if None, will solve
):
    """
    Returns a dict with the KPIs + the raw per-trial series (for plots).
    """
    # Baseline (with clean Sigma)
    w_opt0, te0, Rstar0, Sigma0 = baseline_diagnostics(R_clean, w_bench, c_vec, Sigma_fn, te_cap, w_opt0)

    # Per-trial measures vs baseline optimized portfolio
    turnovers = []
    cosines   = []
    carbon_losses_pp = []   # baseline reduction minus trial reduction (pp)
     # |TE_trial - TE_baseline| in bps

    for w_t, te_t in zip(w_trials, te_trials_annual):
        if w_t is None or not np.all(np.isfinite(w_t)) or (te_t is None) or not np.isfinite(te_t):
            turnovers.append(np.nan); cosines.append(np.nan)
            carbon_losses_pp.append(np.nan)
            continue

        # Turnover and Cosine vs baseline optimized
        turnovers.append(l1_turnover_pct(w_opt0, w_t))
        cosines.append(cosine_similarity(w_opt0, w_t))

        # Carbon reduction loss (pp): max(0, baseline_reduction - trial_reduction)
        Rstar_t = carbon_reduction_pct(w_t, w_bench, c_vec)   # in %
        carbon_losses_pp.append(max(0.0, Rstar0 - Rstar_t))


    # Aggregate KPIs
    kpis = {
        "Median_Turnover_pct":        nanpercentile(turnovers, 50),
        "Median_Cosine":              nanpercentile(cosines, 50),
        "P95_CarbonLoss_pp":          nanpercentile(carbon_losses_pp, 95),
        # keep handy for plots/debug
        "series": {
            "turnover_pct": np.array(turnovers, dtype=float),
            "cosine":       np.array(cosines, dtype=float),
            "carbon_loss_pp": np.array(carbon_losses_pp, dtype=float),
            "baseline": {
                "w_opt0": w_opt0,
                "te0_annual": te0,
                "Rstar0_pct": Rstar0
            }
        }
    }
    return kpis

# ---------- baseline (can reuse your precomputed dict) ----------
def baseline_diagnostics(R_clean, w_bench, c_vec, Sigma_fn, te_cap=0.02, w_opt=None):
    """
    Returns (w_opt0, te0_annual, Rstar0, Sigma0)
    - If w_opt is None, solves the baseline optimization; otherwise uses provided w_opt.
    """
    # Sigma at baseline (on clean returns)
    Sigma0 = Sigma_fn(R_clean)

    if w_opt is None:
        # Solve baseline optim (same as your optimize_sector but minimal)
        N = len(w_bench)
        w = cp.Variable(N)
        te_cap_var_monthly = (te_cap / np.sqrt(12))**2
        tracking_error = cp.quad_form(w - w_bench, cp.psd_wrap(Sigma0))
        obj = cp.Minimize(cp.sum(cp.multiply(w, c_vec)))
        cons = [tracking_error <= te_cap_var_monthly, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, cons)
        solve_qp_with_fallback(prob)
        if prob.status not in ("optimal","optimal_inaccurate") or w.value is None:
            raise RuntimeError(f"Baseline optimization failed: {prob.status}")
        w_opt0 = w.value
    else:
        w_opt0 = np.asarray(w_opt, dtype=float)

    te0 = realized_te_annual(w_opt0, w_bench, Sigma0)              # decimal, e.g. 0.02
    Rstar0 = carbon_reduction_pct(w_opt0, w_bench, c_vec)          # percent
    return w_opt0, te0, Rstar0, Sigma0
# === UTILITIES ===

def sigma_raw_fn(R_clean):
    return R_clean.cov()

def sigma_reg_fn(R_clean):
    lw = LedoitWolf().fit(R_clean)
    Sigma_shrink = lw.covariance_
    lambda_I = 1e-5
    return Sigma_shrink + lambda_I * np.eye(Sigma_shrink.shape[0])

def simulate_parametric_noise(mu, Sigma, T, n_trials):
    return [np.random.multivariate_normal(mu, Sigma, T) for _ in range(n_trials)]

def bootstrap_returns(R_clean_np, n_trials):
    T = R_clean_np.shape[0]
    return [R_clean_np[np.random.choice(T, T, replace=True)] for _ in range(n_trials)]

def compute_perturbed_weights(
    R_clean,
    w_bench,
    c_vec,
    Sigma_fn,
    te_cap=0.03,
    n_trials=100,
    noise_std=0.01,
    noise_type="multiplicative",
    alpha=0.2
):
    N = R_clean.shape[1]
    te_cap_var_monthly = (te_cap / np.sqrt(12)) ** 2
    weights = []
    tracking_errors = []

    for seed in range(n_trials):
        np.random.seed(seed)

        if noise_type == "multiplicative":
            noise = np.random.normal(0, noise_std, R_clean.shape)
            R_perturbed = R_clean + R_clean.multiply(noise)

        elif noise_type == "additive":
            sigma = R_clean.std(axis=0).values.reshape(1, -1)
            noise = np.random.normal(0, 1, R_clean.shape)
            R_perturbed = R_clean + alpha * noise * sigma

        else:
            raise ValueError("noise_type must be 'multiplicative' or 'additive'")

        Sigma = Sigma_fn(R_perturbed)

        w = cp.Variable(N)
        tracking_error = cp.quad_form(w - w_bench, cp.psd_wrap(Sigma))
        constraints = [tracking_error <= te_cap_var_monthly, cp.sum(w) == 1, w >= 0]
        objective = cp.Minimize(cp.sum(cp.multiply(w, c_vec)))
        prob = cp.Problem(objective, constraints)
        solve_qp_with_fallback(prob)

        if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            weights.append(w.value)
            diff = w.value - w_bench
            te_real = np.sqrt(diff.T @ Sigma @ diff) * np.sqrt(12)
            tracking_errors.append(te_real)
        else:
            weights.append(np.full(N, np.nan))
            tracking_errors.append(np.nan)

    return np.array(weights), np.array(tracking_errors)


def compute_hhi(weights):
    return np.sum(np.square(weights))

def check_total_variability(R_clean, sector_name):
    sdev = R_clean.std(axis=0)
    annualised_sdev = sdev * (12**0.5)
    return annualised_sdev

# --- parameters ---
periods = ["0321", "0621", "0921", "1221", "0322", "0622", "0922", "1222",
           "0323", "0623", "0923", "1223"]
n_trials = 200

# --- containers ---
all_sensitivity = []

for period_tag in periods:
    print(f"\n🚀 Running sensitivity analysis for {period_tag} …")

    # define file paths
    data_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_tag}.xlsx"
    returns_file = f"Data/log_returns/sector_log_returns_comp_{period_tag}.xlsx"
    optim_file = f"results/optimal_portfolios/optimal_portfolios_all_te_{period_tag}.pkl"
    out_dir = f"results/sensitivity"
    out_pickle = f"{out_dir}/sensitivity_kpis_{period_tag}.pkl"
    out_excel = f"{out_dir}/sensitivity_kpis_{period_tag}.xlsx"

    # -------------------------------
    # Load optimization frontier
    # -------------------------------
    with open(optim_file, "rb") as f:
        optimal_portfolios_all_te = pickle.load(f)

    optimal_portfolios_shrink_2_TE = extract_optimal_portfolios_at_target_te(
        optimal_portfolios_all_te, target_te_bps=200
    )

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # Check cache
    # -------------------------------
    if os.path.exists(out_pickle) and os.path.exists(out_excel):
        print(f"✅ Using cached results for {period_tag}.")
        with open(out_pickle, "rb") as f:
            sensitivity_results = pickle.load(f)
        tmp = pd.read_excel(out_excel)
        tmp["Period"] = period_tag
        all_sensitivity.append(tmp)
        continue  # skip recomputation

    # -------------------------------
    # Run new computation
    # -------------------------------
    print(f"⚙️ Computing fresh sensitivity KPIs for {period_tag}…")

    data = pd.read_excel(data_file)
    sensitivity_results = {}

    for sector_name in data["GICS Sector"].unique():
        print(f"[{sector_name}] computing sensitivity KPIs…")

        # Load returns for this sector
        R = pd.read_excel(returns_file, sheet_name=sector_name)
        R = R.iloc[:-3]
        R_clean = R.drop(columns=["Date"]).dropna()

        # Benchmark weights + carbon intensity
        sector = data[data["GICS Sector"] == sector_name]
        w_bench = sector["weight_in_sector"].values
        c_vec = sector["Carbon Intensity"].values

        # Baseline optimized weights (already precomputed)
        w_opt0 = optimal_portfolios_shrink_2_TE[sector_name]["w_opt"]

        # === Perturbations ===
        w_trials, te_trials = compute_perturbed_weights(
            R_clean,
            w_bench,
            c_vec,
            sigma_reg_fn,
            te_cap=0.02,  # annual TE cap
            n_trials=n_trials,
            noise_std=0.2,
            noise_type="additive",
        )

        # === Compute KPIs ===
        kpis = sensitivity_kpis_from_trials(
            w_trials=w_trials,
            te_trials_annual=te_trials,
            R_clean=R_clean,
            w_bench=w_bench,
            c_vec=c_vec,
            Sigma_fn=sigma_reg_fn,
            te_cap=0.02,
            w_opt0=w_opt0,
        )

        sensitivity_results[sector_name] = {
            "Median_Turnover_pct": kpis["Median_Turnover_pct"],
            "Median_Cosine": kpis["Median_Cosine"],
            "P95_CarbonLoss_pp": kpis["P95_CarbonLoss_pp"]
        }

    # -------------------------------
    # Save new results
    # -------------------------------
    tmp = pd.DataFrame(sensitivity_results).T.reset_index()
    tmp.rename(columns={"index": "Sector"}, inplace=True)
    tmp["Period"] = period_tag

    with open(out_pickle, "wb") as f:
        pickle.dump(sensitivity_results, f)
    tmp.to_excel(out_excel, index=False)

    print(f"✅ Saved results for {period_tag} to {out_pickle}")

    all_sensitivity.append(tmp)

# -------------------------------
# Combine all periods
# -------------------------------

df = pd.concat(all_sensitivity, ignore_index=True)

def minmax_norm_grouped(df, col):
    return df.groupby("Period")[col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )


df["Turnover_norm"] = minmax_norm_grouped(df, "Median_Turnover_pct")

df["Inv_Median_Cosine"] = 1 - df["Median_Cosine"]
df["Cosine_norm"]   = minmax_norm_grouped(df, "Inv_Median_Cosine")

df["CarbonLoss_norm"] = minmax_norm_grouped(df, "P95_CarbonLoss_pp")

df["Sensitivity_Score_raw"] =1/3 * (
    df["Turnover_norm"] +  df["Cosine_norm"] +  df["CarbonLoss_norm"])

df["Sensitivity_Score"] = 1 - df["Sensitivity_Score_raw"]
df["Sensitivity_Score"] = minmax_norm_grouped(df, 'Sensitivity_Score')

# --- save and plot ---
df.to_excel("results/sensitivity/sensitivity_scores_by_period.xlsx", index=False)
print("✅ Saved sensitivity scores to results/sensitivity/sensitivity_scores_by_period.xlsx")

# --- plot sensitivity score evolution ---
from plot_functions import plot_sector_evolution


print("✅ Plotted sensitivity score evolution")

plot_sector_evolution(
    df=df,
    value_col="Median_Turnover_pct",
    title="Median_Turnover_pct Across Periods",
    ylabel="Median_Turnover_pct",
    figsize=(12, 7)
)

plot_sector_evolution(
    df=df,
    value_col="Inv_Median_Cosine",
    title="Inv_Median_Cosine Across Periods",
    ylabel="Inv_Median_Cosine",
    figsize=(12, 7)
)

plot_sector_evolution(
    df=df,
    value_col="P95_CarbonLoss_pp",
    title="Carbon loss Evolution Across Periods",
    ylabel="Inverted Sensitivity Score",
    figsize=(12, 7)
)

plot_sector_evolution(
    df=df,
    value_col="Sensitivity_Score",
    title="Sensitivtiy Score Across Periods",
    ylabel="Sensitivity Score",
    figsize=(12, 7)
)
