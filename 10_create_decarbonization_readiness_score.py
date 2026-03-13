"""
Compute final sector-level DRI scores:
- Average per-period scores
- Combine into composite DRI
"""
import pandas as pd
from plot_functions import plot_sector_radar_grid, plot_all_dimension_evolution
import pickle

# =============================================================================
# 1. LOAD THE FOUR DIMENSIONS
# =============================================================================

# --- Room for Maneuver ---
room_df = pd.read_excel("results/room_for_maneuver/room_for_maneuver_scores_by_period.xlsx")
# Expected columns: ["Sector", "Period", "Room_for_Maneuver_Score_Adjusted"]

# --- Flexibility ---
flex_df = pd.read_excel("results/flexibility/flexibility_scores_by_period.xlsx")
# Expected columns: ["Sector", "Period", "Flexibility_Score"]

# --- Sensitivity ---
sens_df = pd.read_excel("results/sensitivity/sensitivity_scores_by_period.xlsx")
# Expected columns: ["Sector", "Period", "Sensitivity_Score"]

# --- Robustness ---
robust_df = pd.read_excel("results/robustness/robustness_scores_by_period.xlsx")
# Expected columns: ["sector", "period", "Robustness_Score"]
robust_df = robust_df.rename(columns={"sector": "Sector", "period": "Period"})




# =============================================================================
# 2. AVERAGE ACROSS PERIODS (sector-level averages)
# =============================================================================

room_avg = (
    room_df.groupby("Sector", as_index=False)["Room_for_Maneuver_Score"]
           .mean()
           .rename(columns={"Room_for_Maneuver_Score": "Room_Avg"})
)

flex_avg = (
    flex_df.groupby("Sector", as_index=False)["Flexibility_Score"]
           .mean()
           .rename(columns={"Flexibility_Score": "Flex_Avg"})
)

sens_avg = (
    sens_df.groupby("Sector", as_index=False)["Sensitivity_Score"]
           .mean()
           .rename(columns={"Sensitivity_Score": "Sens_Avg"})
)

robust_avg = (
    robust_df.groupby("Sector", as_index=False)["Robustness_Score"]
             .mean()
             .rename(columns={"Robustness_Score": "Robust_Avg"})
)

room_df["Period"] = room_df["Period"].astype(str).str.zfill(4)
flex_df["Period"] = flex_df["Period"].astype(str).str.zfill(4)
sens_df["Period"] = sens_df["Period"].astype(str).str.zfill(4)
robust_df["Period"] = robust_df["Period"].astype(str).str.zfill(4)

period_order = sorted(room_df["Period"].unique())

plot_all_dimension_evolution(room_df, flex_df, sens_df, robust_df, savepath="results/DRI/dri_dimension_evolution.pdf"
)


# =============================================================================
# 3. MERGE ALL DIMENSIONS
# =============================================================================
final_df = (
    room_avg[["Sector", "Room_Avg"]]
    .merge(flex_avg[["Sector", "Flex_Avg"]], on="Sector")
    .merge(sens_avg[["Sector", "Sens_Avg"]], on="Sector")
    .merge(robust_avg[["Sector", "Robust_Avg"]], on="Sector")
)


# =============================================================================
# 4. COMPOSITE DECARBONIZATION READINESS INDEX (DRI)
# =============================================================================
final_df["DRI"] = (
    final_df["Room_Avg"] +
    final_df["Flex_Avg"] +
    final_df["Sens_Avg"] +
    final_df["Robust_Avg"]
) / 4.0


# =============================================================================
# 5. SAVE RESULTS
# =============================================================================
final_df.to_excel("results/DRI/decarbonization_readiness_index.xlsx", index=False)
final_df.to_parquet("results/DRI/decarbonization_readiness_index.parquet", index=False)
print("✅ Saved DRI results to results/DRI/decarbonization_readiness_index.xlsx")

print("\n🏆 Top 5 sectors by DRI:")
print(final_df.sort_values("DRI", ascending=False).head())

cols = [  "Sens_Avg",
    "Flex_Avg",
    "Room_Avg",
    "Robust_Avg"]

plot_sector_radar_grid(
    df=final_df,
    cols_to_norm=cols,
    title="Decarbonization Readiness Index (DRI) — Radar Profiles across Sectors",
    savepath="results/DRI/dri_radar_profiles.pdf"
)


# ---------------------------------------------------------
# 6. CREATE RADAR DATA STRUCTURE FOR STREAMLIT
# ---------------------------------------------------------

radar_dict = {}

for _, row in final_df.iterrows():
    radar_dict[row["Sector"]] = {
        "Room for Maneuver": float(row["Room_Avg"]),
        "Flexibility": float(row["Flex_Avg"]),
        "Sensitivity": float(row["Sens_Avg"]),
        "Robustness": float(row["Robust_Avg"]),
    }

# Save pickle for the Streamlit dashboard
with open("results/DRI/dri_radar_data.pkl", "wb") as f:
    pickle.dump(radar_dict, f)

print("✅ Saved radar data to Data/dri_radar_data.pkl")
