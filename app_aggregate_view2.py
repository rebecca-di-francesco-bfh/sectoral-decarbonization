import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go

# ---------------------------------------------------------
# Streamlit CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TE–Carbon Dashboard — Aggregate View", layout="wide")

# ---------------------------------------------------------
# GLOBAL AESTHETIC STYLE — UPDATED TO PURPLE/TEAL/ORANGE GRADIENT PALETTE
# ---------------------------------------------------------
st.markdown("""
<style>

/* Sidebar text to white ONLY (keep original purple background) */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Make selectbox label white */
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* Make selectbox selected text white */
section[data-testid="stSidebar"] .stSelectbox div[role="combobox"] * {
    color: #FFFFFF !important;
}

/* Make the dropdown arrow white */
section[data-testid="stSidebar"] svg {
    fill: #FFFFFF !important;
}

/* DO NOT override background colors (keep your pastel grey + purple theme) */

</style>
""", unsafe_allow_html=True)



PLOT_COLORS = {
    "primary": "#6A5AE0",   # main purple
    "accent": "#FF7B72",    # coral/orange highlight
    "teal": "#4CC9F0",       # cyan/teal
    "grey": "#E1E5EA"
}

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<div class="big-title">TE–Carbon Dashboard</div>', unsafe_allow_html=True)
st.caption("Aggregate view: compare sectors on TE–Carbon frontiers and DRI.")

# ---------------------------------------------------------
# SIDEBAR — ONLY “Aggregate”
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    view = st.selectbox("View", ["Aggregate"], index=0)


# LOAD AGGREGATE FRONTIER DATA (your precomputed file)
# ---------------------------------------------------------
st.markdown('<div class="section-title">TE–Carbon Frontiers Across Sectors</div>', unsafe_allow_html=True)

try:
    with open("results/optimal_portfolios/optimal_portfolios_all_te_1223.pkl", "rb") as f:
        sector_frontiers = pickle.load(f)   # dict: sector → {tracking_errors, carbon_reductions}
except:
    st.error("Missing Data/sector_frontiers_1223.pkl")
    st.stop()

sectors = list(sector_frontiers.keys())
ordered_sectors = sorted(sectors)

# Plot using Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))

for sector in ordered_sectors:
    metrics = sector_frontiers[sector]
    ax.plot(metrics["tracking_errors"], metrics["carbon_reductions"], label=sector)

ax.set_xlabel("Tracking Error (bps)")
ax.set_ylabel("Carbon Reduction (%)")
ax.set_title("TE–Carbon Frontier for All Sectors")
ax.grid(True)
ax.legend(title="Sector", bbox_to_anchor=(1.05, 0.5), loc="center left")

st.pyplot(fig)


# ---------------------------------------------------------
# DRI RADAR PANEL (all sectors)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Decarbonization Readiness Index (DRI)</div>', unsafe_allow_html=True)

try:
    with open("results/DRI/dri_radar_data.pkl", "rb") as f:
        dri_data = pickle.load(f)   # dict: sector → dict of 4 dimensions (already normalized)
except:
    st.error("Missing Data/dri_radar_data.pkl")
    st.stop()

# Radar helper
def make_radar(sector, values):
    labels = list(values.keys())
    vals = list(values.values())
    
    vals += vals[:1]  # close loop
    angles = np.linspace(0, 2*np.pi, len(vals))

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(sector, fontsize=10)
    return fig

# Display radar grid
cols = st.columns(4)
i = 0
for sector in ordered_sectors:
    fig = make_radar(sector, dri_data[sector])
    cols[i].pyplot(fig)
    i = (i + 1) % 4


st.markdown("—— End of Aggregate View ——")

