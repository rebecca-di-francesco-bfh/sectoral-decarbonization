import os
import numpy as np
import pandas as pd

from utils import sigma_shrink_fn  # make sure this is in your PYTHONPATH

# Periods you use in your study
PERIODS = ["0321", "0621", "0921", "1221",
           "0322", "0622", "0922", "1222",
           "0323", "0623", "0923", "1223"]

def run_ci_revenue_scatterplots(df, period_tag):
    """
    Creates scatterplots:
      1) Imputed CI vs log(Revenue)
      2) Non-imputed CI vs log(Revenue)
    Only for Real Estate and Information Technology.
    Saves PNGs to results/diagnostics/ci_revenue_scatterplots/.
    """

    target_sectors = ["Real Estate", "Information Technology"]
    outdir = "results/diagnostics/ci_revenue_scatterplots"
    os.makedirs(outdir, exist_ok=True)

    # Ensure revenue column exists
    if "Revenue" not in df.columns:
        print(f"   ❌ ERROR: 'Revenue' column not found in dataset for period {period_tag}")
        return

    # Drop NA revenue or CI rows
    df = df.dropna(subset=["Revenue", "Carbon Intensity"]).copy()
    df["log_revenue"] = np.log(df["Revenue"] + 1e-9)   # avoid log(0)

    for sector in target_sectors:
        sdf = df[df["GICS Sector"] == sector].copy()
        if sdf.empty:
            print(f"   ⚠️ No data for sector {sector} in period {period_tag}")
            continue

        # ---- Plot: Imputed CI ----
        imp = sdf[sdf["CI_imputed"]]
        if len(imp) > 0:
            plt.figure(figsize=(7,5))
            plt.scatter(imp["log_revenue"], imp["Carbon Intensity"], alpha=0.7)
            plt.title(f"{sector} — Imputed CI vs log(Revenue)\nPeriod {period_tag}")
            plt.xlabel("log(Revenue)")
            plt.ylabel("Carbon Intensity")
            plt.grid(True, alpha=0.3)

            fname = f"{outdir}/{sector.replace(' ','_')}_imputed_{period_tag}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   ✓ Saved: {fname}")
        else:
            print(f"   (No imputed CI in {sector} for {period_tag})")

        # ---- Plot: Non-imputed CI ----
        non = sdf[~sdf["CI_imputed"]]
        if len(non) > 0:
            plt.figure(figsize=(7,5))
            plt.scatter(non["log_revenue"], non["Carbon Intensity"], alpha=0.7)
            plt.title(f"{sector} — Non-imputed CI vs log(Revenue)\nPeriod {period_tag}")
            plt.xlabel("log(Revenue)")
            plt.ylabel("Carbon Intensity")
            plt.grid(True, alpha=0.3)

            fname = f"{outdir}/{sector.replace(' ','_')}_nonimputed_{period_tag}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   ✓ Saved: {fname}")
        else:
            print(f"   (No non-imputed CI in {sector} for {period_tag})")

    return

# ============================================================
# 1) COVARIANCE + TE SANITY DIAGNOSTICS (shrink cov only)
# ============================================================
import matplotlib.pyplot as plt

def run_ci_mcap_scatterplots(df, period_tag):
    """
    Creates scatterplots:
      1) Imputed CI vs log(Market Cap)
      2) Non-imputed CI vs log(Market Cap)
    Only for Real Estate and Information Technology.
    Saves PNGs to results/diagnostics/ci_mcap_scatterplots/.
    """

    target_sectors = ["Real Estate", "Information Technology"]
    outdir = "results/diagnostics/ci_mcap_scatterplots"
    os.makedirs(outdir, exist_ok=True)
    print(df.columns)
    # Drop rows missing market cap or CI
    df = df.dropna(subset=["float_mcap", "Carbon Intensity"]).copy()
    df["log_mcap"] = np.log(df["float_mcap"])

    for sector in target_sectors:
        sdf = df[df["GICS Sector"] == sector].copy()
        if sdf.empty:
            print(f"   ⚠️ No data for sector {sector} in period {period_tag}")
            continue

        # ---- Plot: Imputed CI ----
        imp = sdf[sdf["CI_imputed"]]
        if len(imp) > 0:
            plt.figure(figsize=(7,5))
            plt.scatter(imp["log_mcap"], imp["Carbon Intensity"], alpha=0.7)
            plt.title(f"{sector} — Imputed CI vs log(Market Cap)\nPeriod {period_tag}")
            plt.xlabel("log(Market Cap)")
            plt.ylabel("Carbon Intensity")
            plt.grid(True, alpha=0.3)

            fname = f"{outdir}/{sector.replace(' ','_')}_imputed_{period_tag}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   ✓ Saved: {fname}")
        else:
            print(f"   (No imputed CI in {sector} for {period_tag})")

        # ---- Plot: Non-imputed CI ----
        non = sdf[~sdf["CI_imputed"]]
        if len(non) > 0:
            plt.figure(figsize=(7,5))
            plt.scatter(non["log_mcap"], non["Carbon Intensity"], alpha=0.7)
            plt.title(f"{sector} — Non-imputed CI vs log(Market Cap)\nPeriod {period_tag}")
            plt.xlabel("log(Market Cap)")
            plt.ylabel("Carbon Intensity")
            plt.grid(True, alpha=0.3)

            fname = f"{outdir}/{sector.replace(' ','_')}_nonimputed_{period_tag}.png"
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"   ✓ Saved: {fname}")
        else:
            print(f"   (No non-imputed CI in {sector} for {period_tag})")

    return

def run_covariance_te_diagnostics():
    """
    For each period and sector:
      - Recompute Sigma_shrink using sigma_shrink_fn
      - Report:
          avg/min/max monthly vol
          avg pairwise correlation
          first PC share
          shrinkage alpha
      - TE sanity check:
          TE(w_bench) = 0
          TE_annual for a small weight perturbation (~1%) is reasonable
    Saves results to 'covariance_te_diagnostics.xlsx'.
    """

    diag_results = []

    for period_tag in PERIODS:
        print(f"\n===== PERIOD {period_tag} =====")

        data_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_tag}.xlsx"
        log_file  = f"data/log_returns/sector_log_returns_comp_{period_tag}.xlsx"

        # Load benchmark data and sector info
        df = pd.read_excel(data_file)
        # Load all sector log-return sheets
        log_returns_all = pd.read_excel(log_file, sheet_name=None)

        for sector_name in df["GICS Sector"].unique():
            print(f"\n--- Sector: {sector_name} ---")

            sector = df[df["GICS Sector"] == sector_name]
            w_bench = sector["weight_in_sector"].values.astype(float)

            # Skip sectors with fewer than 2 names (TE perturbation meaningless)
            if len(w_bench) < 2:
                print("   Skipping (less than 2 stocks).")
                continue

            # Load log returns for this sector
            if sector_name not in log_returns_all:
                print(f"   WARNING: Sector {sector_name} not in log_returns file.")
                continue

            R = log_returns_all[sector_name].copy()

            # Drop non-return columns (assume 'Date' is one of them)
            non_return_cols = [c for c in R.columns if c.lower() == "date" or R[c].dtype == "O"]
            R_clean = R.drop(columns=non_return_cols).dropna()

            # Recompute shrinkage covariance
            Sigma, alpha = sigma_shrink_fn(R_clean)

            # --- Covariance diagnostics ---

            # Monthly vol (since R_clean is monthly log returns)
            vol = np.sqrt(np.diag(Sigma))
            avg_vol = vol.mean()
            max_vol = vol.max()
            min_vol = vol.min()

            # Pairwise correlation
            try:
                corr = np.corrcoef(R_clean.T)
                avg_corr = corr[np.triu_indices_from(corr, k=1)].mean()
            except Exception:
                avg_corr = np.nan

            # Eigen structure
            eigvals = np.linalg.eigvalsh(Sigma)
            eigvals_sorted = np.sort(eigvals)[::-1]
            if eigvals_sorted.sum() > 0:
                first_pc_share = eigvals_sorted[0] / eigvals_sorted.sum()
            else:
                first_pc_share = np.nan

            # --- TE sanity check ---

            # A small perturbation of weights
            w2 = w_bench.copy()
            w2[0] += 0.01
            w2[1] -= 0.01

            # TE for benchmark should be 0
            TE0 = np.sqrt((w_bench - w_bench).T @ Sigma @ (w_bench - w_bench))

            # Monthly TE of perturbed weights
            TE_monthly = np.sqrt((w2 - w_bench).T @ Sigma @ (w2 - w_bench))
            # Annualized TE (your optimization uses annual TE with sqrt(12))
            TE_annual = TE_monthly * np.sqrt(12)

            print(f"   Avg monthly vol: {avg_vol:.4f}")
            print(f"   Max monthly vol: {max_vol:.4f}, Min monthly vol: {min_vol:.4f}")
            print(f"   Avg correlation: {avg_corr:.4f}")
            print(f"   First PC share: {first_pc_share:.4f}")
            print(f"   Shrinkage alpha: {alpha:.4f}")
            print(f"   TE0 (should be 0): {TE0:.8f}")
            print(f"   TE_annual after 1% perturbation: {TE_annual:.4f}")

            diag_results.append({
                "Period": period_tag,
                "Sector": sector_name,
                "avg_vol_monthly": avg_vol,
                "max_vol_monthly": max_vol,
                "min_vol_monthly": min_vol,
                "avg_corr": avg_corr,
                "first_PC_share": first_pc_share,
                "TE0": TE0,
                "TE_annual_1pct_perturb": TE_annual,
                "shrink_alpha": alpha
            })

    diag_df = pd.DataFrame(diag_results)
    diag_df.to_excel("results/diagnostics/covariance_te_diagnostics.xlsx", index=False)
    print("\n✓ Saved covariance and TE diagnostics to 'covariance_te_diagnostics.xlsx'")

    return diag_df


# ============================================================
# 2) CARBON INTENSITY DIAGNOSTICS
#    - zeros / negatives
#    - imputation bias
# ============================================================

def run_carbon_intensity_diagnostics():
    """
    For each period:
      - flag zero or negative Carbon Intensity
      - compare CI for imputed vs non-imputed firms
      - summarize by sector
    Saves:
      - 'ci_zero_negative_summary.xlsx'
      - 'ci_imputation_bias_summary.xlsx'
    """

    zero_neg_rows = []
    imputation_rows = []

    for period_tag in PERIODS:
        print(f"\n===== PERIOD {period_tag} (CI diagnostics) =====")

        data_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_tag}.xlsx"
        df = pd.read_excel(data_file)

        # 2.1 Zero or negative CI
        bad = df[df["Carbon Intensity"] <= 0].copy()
        if not bad.empty:
            print(f"   Found {len(bad)} rows with CI <= 0")
            for _, row in bad.iterrows():
                zero_neg_rows.append({
                    "Period": period_tag,
                    "SYMBOL": row["SYMBOL"],
                    "Sector": row["GICS Sector"],
                    "Carbon Intensity": row["Carbon Intensity"]
                })

        # 2.2 Imputation vs non-imputation
        # Create an "Imputed_any" flag: True if any of Scope 1/2/3 is imputed
        imputed_cols = ["Scope 1 Imputed", "Scope 2 Imputed", "Scope 3 Imputed"]
        df["Imputed_any"] = df[imputed_cols].any(axis=1)

        ci_imputed = df[df["Imputed_any"]]["Carbon Intensity"]
        ci_non_imputed = df[~df["Imputed_any"]]["Carbon Intensity"]

        imputation_rows.append({
            "Period": period_tag,
            "CI_mean_imputed": ci_imputed.mean(),
            "CI_mean_non_imputed": ci_non_imputed.mean(),
            "CI_median_imputed": ci_imputed.median(),
            "CI_median_non_imputed": ci_non_imputed.median(),
            "N_imputed": len(ci_imputed),
            "N_non_imputed": len(ci_non_imputed)
        })

        print(f"   CI mean (imputed): {ci_imputed.mean():.4f}, CI mean (non-imputed): {ci_non_imputed.mean():.4f}")
        print(f"   N imputed: {len(ci_imputed)}, N non-imputed: {len(ci_non_imputed)}")

    # Save zero/negative CI details
    if zero_neg_rows:
        zero_neg_df = pd.DataFrame(zero_neg_rows)
        zero_neg_df.to_excel("results/diagnostics/ci_zero_negative_summary.xlsx", index=False)
        print("\n✓ Saved CI <= 0 summary to 'ci_zero_negative_summary.xlsx'")
    else:
        print("\n✓ No zero or negative Carbon Intensity values found across all periods.")

    # Save imputation bias summary
    imputation_df = pd.DataFrame(imputation_rows)
    imputation_df.to_excel("results/diagnostics/ci_imputation_bias_summary.xlsx", index=False)
    print("✓ Saved CI imputation bias summary to 'ci_imputation_bias_summary.xlsx'")

    return


# ============================================================
# 3) WEIGHT NORMALIZATION DIAGNOSTICS
# ============================================================

def run_weight_normalization_diagnostics():
    """
    For each period and sector:
      - check that sum of weight_in_sector ≈ 1
      - check for negative weights
    Saves a summary to 'weight_normalization_diagnostics.xlsx'.
    """

    weight_rows = []

    for period_tag in PERIODS:
        print(f"\n===== PERIOD {period_tag} (weights diagnostics) =====")

        data_file = f"data/datasets/benchmark_weights_carbon_intensity_{period_tag}.xlsx"
        df = pd.read_excel(data_file)

        for sector_name, sdf in df.groupby("GICS Sector"):
            total_weight = sdf["weight_in_sector"].sum()
            min_weight = sdf["weight_in_sector"].min()
            max_weight = sdf["weight_in_sector"].max()

            # Flag problems
            sum_ok = np.isclose(total_weight, 1.0, atol=1e-4)
            neg_count = (sdf["weight_in_sector"] < 0).sum()

            if not sum_ok or neg_count > 0:
                print(f"   Sector {sector_name}: sum={total_weight:.6f}, min={min_weight:.6f}, neg_count={neg_count}")

            weight_rows.append({
                "Period": period_tag,
                "Sector": sector_name,
                "sum_weight": total_weight,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "neg_weight_count": neg_count,
                "sum_ok": sum_ok
            })

    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_excel("results/diagnostics/weight_normalization_diagnostics.xlsx", index=False)
    print("\n✓ Saved weight normalization diagnostics to 'weight_normalization_diagnostics.xlsx'")

    return

def run_ci_diagnostics_by_sector(period_tag, df):
    """
    df must contain:
    - 'Carbon Intensity'
    - 'CI_imputed' (True/False)
    - 'GICS Sector'
    """
    print(f"\n===== PERIOD {period_tag} (CI diagnostics) =====")

    ci = df['Carbon Intensity']
    imp = df['CI_imputed']

    # --- Overall means ---
    mean_imp = ci[imp].mean()
    mean_non = ci[~imp].mean()

    print(f"   Overall CI mean (imputed): {mean_imp:.4f}, "
          f"CI mean (non-imputed): {mean_non:.4f}")
    print(f"   N imputed: {imp.sum()}, N non-imputed: {(~imp).sum()}")

    # === Sector-level diagnostics ===
    print("\n   Sector-level differences:")
    sector_stats = []

    for sector, sub in df.groupby("GICS Sector"):
        ci_imp = sub.loc[sub["CI_imputed"], "Carbon Intensity"]
        ci_non = sub.loc[~sub["CI_imputed"], "Carbon Intensity"]

        if len(ci_imp) == 0 or len(ci_non) == 0:
            # Skip sectors where all firms are imputed or all are non-imputed
            continue

        mean_imp_s = ci_imp.mean()
        mean_non_s = ci_non.mean()
        diff = mean_imp_s - mean_non_s

        sector_stats.append((sector, mean_imp_s, mean_non_s, diff))

    # Convert to DataFrame for nicer display
    sector_df = pd.DataFrame(sector_stats,
                             columns=["Sector", "Mean CI (imputed)",
                                      "Mean CI (non-imputed)", "Difference"])

    sector_df["Flag_HIGH_gap"] = sector_df["Difference"] > (
        sector_df["Difference"].mean() + 2 * sector_df["Difference"].std()
    )

    print(sector_df.sort_values("Difference", ascending=False).to_string(index=False))

    return sector_df

# ============================================================
# MAIN ENTRY
# ============================================================

if __name__ == "__main__":
    print("\nRunning covariance and TE diagnostics...")
    cov_te_df = run_covariance_te_diagnostics()

    # print("\nRunning carbon intensity diagnostics...")
    # run_carbon_intensity_diagnostics()

    # print("\nRunning weight normalization diagnostics...")
    # run_weight_normalization_diagnostics()

    # print("\nAll diagnostics completed.")
    print("\nRunning sector-level CI diagnostics for all periods...\n")

    # for period_tag in PERIODS:

    #     # Load data for this period
    #     data_file = f"data/datasets/dataset_comp_{period_tag}.xlsx"
    #     df = pd.read_excel(data_file)

    #     # Create CI_imputed flag
    #     df["CI_imputed"] = df[
    #         ["Scope 1 Imputed", "Scope 2 Imputed", "Scope 3 Imputed"]
    #     ].any(axis=1)

    #     # Run diagnostics
    #     sector_df = run_ci_diagnostics_by_sector(period_tag, df)

    #     # Optionally: save sector diagnostics per period
    #     out_file = f"results/diagnostics/ci_sector_gap_{period_tag}.xlsx"
    #     os.makedirs("results/diagnostics", exist_ok=True)
    #     sector_df.to_excel(out_file, index=False)

        # print(f"  → Saved sector CI diagnostics to {out_file}")
        #   # === NEW: create scatterplots for Real Estate & IT ===
        # run_ci_mcap_scatterplots(df, period_tag)
        # run_ci_revenue_scatterplots(df, period_tag)

    print("\n✓ All sector-level CI diagnostics completed.")