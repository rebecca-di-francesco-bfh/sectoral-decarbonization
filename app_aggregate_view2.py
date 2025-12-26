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

sector_colors = {
    'Communication Services': '#E63946',      # Red
    'Consumer Discretionary': '#F77F00',      # Orange
    'Consumer Staples': '#FCBF49',            # Yellow
    'Energy': '#06FFA5',                      # Mint Green
    'Financials': '#118AB2',                  # Blue
    'Health Care': '#073B4C',                 # Dark Blue
    'Industrials': '#8B5A3C',                 # Brown
    'Information Technology': '#9D4EDD',      # Purple
    'Materials': '#FF69B4',                   # Pink
    'Real Estate': '#BC4749',                 # Burgundy
    'Utilities': '#808080'                    # Gray
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
fig = go.Figure()

for sector in ordered_sectors:
    d = sector_frontiers[sector]
    abs_carbon = [np.dot(w, d["carbon_intensity"]) for w in d["weights_by_te"]]

    fig.add_trace(
        go.Scatter(
            x=d["tracking_errors"],
            y=abs_carbon,
            mode="lines",
            name=sector,
            line=dict(color=sector_colors.get(sector), width=2)
        )
    )

x_min = min(np.min(d["tracking_errors"]) for d in sector_frontiers.values())
x_max = max(np.max(d["tracking_errors"]) for d in sector_frontiers.values())

fig.update_xaxes(range=[x_min, x_max], fixedrange=True)
fig.update_yaxes(fixedrange=False)

fig.update_layout(
    title="Tracking Error vs Absolute Portfolio Carbon Intensity",
    xaxis_title="Tracking Error (bps)",
    yaxis_title="Weighted Carbon Intensity",
    template="plotly_white",
    height=650,                     # 🔴 taller figure
    margin=dict(l=70, r=50, t=70, b=70),
    legend_title="Sector",
    dragmode="zoom"
)

st.plotly_chart(fig, use_container_width=True)




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

labels = ["Room for Maneuver", "Flexibility", "Sensitivity", "Robustness"]

def make_sector_radar(sector_name, values, color):
    r = [
        values["Room for Maneuver"],
        values["Flexibility"],
        values["Sensitivity"],
        values["Robustness"]
    ]
    r = r + [r[0]]  # close polygon

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=labels + [labels[0]],
            fill="toself",
            name=sector_name,
            line=dict(color=color, width=3),
            opacity=0.85
        )
    )
    fig.add_annotation(
    x=0.97,
    y=0.05,
    xref="paper",
    yref="paper",
    text=(
        f"<b>{sector_name}</b><br>"
        f"<span style='font-size:11px'>DRI = {dri_scores[sector_name]:.2f}</span>"
    ),
    showarrow=False,
    align="right",
    font=dict(size=12, color="white"),
    bgcolor="rgba(0,0,0,0.55)",
    bordercolor="white",
    borderwidth=1.5,
    borderpad=4
)


    fig.update_layout(
    polar=dict(
        radialaxis=dict(
            range=[0, 1],
            showticklabels=False,
            ticks=""
        ),
        angularaxis=dict(
            tickfont=dict(size=11)
        )
    ),
    showlegend=False,
    width=320,
    height=300,
    margin=dict(l=20, r=20, t=20, b=20),
    template="plotly_dark"
)



    

    return fig


# Compute aggregate DRI score per sector
dri_scores = {
    sector: np.mean(list(values.values()))
    for sector, values in dri_data.items()
}

ordered_sectors = sorted(
    dri_scores.keys(),
    key=lambda s: dri_scores[s],
    reverse=True
)

cols = st.columns(3)
i = 0

for sector in ordered_sectors:
    fig_radar = make_sector_radar(
        sector_name=sector,
        values=dri_data[sector],
        color=sector_colors.get(sector, "#6A5AE0")
    )

    cols[i].plotly_chart(
        fig_radar,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    i = (i + 1) % 3

st.caption(
    "Each radar summarizes a sector’s Decarbonization Readiness Index across four dimensions. "
    "Scores are normalized to [0,1] within each dimension."
)


