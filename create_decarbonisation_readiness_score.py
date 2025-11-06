robustness_df = pd.read_excel("results/robustness/robustness_scores_by_period.xlsx")

# Average robustness per sector across all periods
robustness_sector_avg = (
    robustness_df
    .groupby("sector", as_index=False)["Robustness_Score"]
    .mean()
    .rename(columns={"Robustness_Score": "Robustness_Avg"})
)

def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())

robustness_sector_avg["Robustness_norm"] = minmax_norm(robustness_sector_avg["Robustness_Avg"])

sensitivity_df = pd.read_excel("results/sensitivity/sensitivity_scores_by_period.xlsx")
# --- then you average ---
sens_sector_avg = (
    sensitivity_df.groupby("Sector", as_index=False)["Sensitivity_Score"]
      .mean()
      .rename(columns={"Sensitivity_Score": "Sensitivity_Avg"})
)

# Optional final normalization across sectors for radar chart / DRI
sens_sector_avg["Sensitivity_norm"] = (
    (sens_sector_avg["Sensitivity_Avg"] - sens_sector_avg["Sensitivity_Avg"].min()) /
    (sens_sector_avg["Sensitivity_Avg"].max() - sens_sector_avg["Sensitivity_Avg"].min())
)

final_df = (
    room_df
    .merge(flex_df, on="sector")
    .merge(sens_sector_avg, on="sector")
    .merge(robustness_sector_avg[["sector", "Robustness_norm"]], on="sector")
)

final_df["DRI"] = (
    final_df["Room_for_Maneuver_norm"] +
    final_df["Flexibility_norm"] +
    final_df["Stability_norm"] +
    final_df["Robustness_norm"]
) / 4


