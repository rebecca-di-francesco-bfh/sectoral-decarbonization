import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import plotly.graph_objects as go

METRIC_DESCRIPTIONS = {
    # --------------------------
    # ROOM FOR MANEUVER
    # --------------------------
    "slope_2pct": (
        "Slope @ 2% TE:\n"
        "Local sensitivity of carbon reduction to small changes in tracking error "
        "around the 2% TE point."
    ),
    "auc_5pct": (
        "AUC @ 5% TE:\n"
        "Area under the TE–Carbon frontier up to 5% TE. Measures total carbon-"
        "reduction potential available within a 5% TE budget."
    ),
    "max_cut_5pct": (
        "Max Cut @ 5% TE:\n"
        "Maximum achievable carbon reduction (%) when allowing a 5% tracking "
        "error budget."
    ),
    "min_te_50pct_cut": (
        "Min TE for 50% Cut:\n"
        "Minimum tracking error (bps) required to achieve at least a 50% carbon "
        "reduction relative to the benchmark."
    ),

    # --------------------------
    # FLEXIBILITY
    # --------------------------
    "avg_bandwidth": (
        "Average ε-Bandwidth:\n"
        "Average width of allowable weight intervals for stocks that still satisfy "
        "TE and carbon constraints. Higher = more portfolio flexibility."
    ),
    "median_bandwidth": (
        "Median ε-Bandwidth:\n"
        "Median width of the ε-bands across all stocks. A central measure of "
        "portfolio flexibility."
    ),
    "max_bandwidth": (
        "Max ε-Bandwidth:\n"
        "Maximum allowable variation range for any individual stock’s weight. "
        "Higher = at least one stock has substantial flexibility."
    ),
    "l2_lower_bound": (
        "L2 Lower Bound (Same Objective):\n"
        "Minimum L₂ distance between the optimal portfolio and any other equally-"
        "optimal portfolio that respects constraints. Higher = more flexibility."
    ),
    "turnover_upper_bound": (
        "Turnover UB (√N × L₂):\n"
        "Upper bound on the turnover needed to move within the set of optimal "
        "portfolios. Higher = more flexibility."
    ),

    # --------------------------
    # ROBUSTNESS
    # --------------------------
    "annualized_te": (
        "Annualized Tracking Error:\n"
        "Annualized deviation of the optimized portfolio from the benchmark. "
        "Lower = more stable relative to benchmark movements."
    ),
    "annualized_volatility": (
        "Annualized Volatility:\n"
        "Volatility of the sector benchmark. Used to normalize tracking error."
    ),
    "robustness_ratio": (
        "Robustness Ratio:\n"
        "TE divided by sector volatility (adjusted with exponent α). "
        "Lower values indicate more robust portfolios."
    ),
    "robustness_score": (
        "Robustness Score (0–1):\n"
        "Normalized measure of stability. Higher = more robust relative to other "
        "sectors within the same period."
    ),

    # --------------------------
    # SENSITIVITY
    # --------------------------
    "median_turnover": (
        "Median Turnover (%):\n"
        "Median turnover required to adjust to perturbed optimal portfolios. "
        "Higher = more sensitivity to input changes."
    ),
    "median_cosine_inv": (
        "Inverted Median Cosine Similarity:\n"
        "Cosine similarity measures alignment between baseline and perturbed "
        "portfolios. After inversion, higher values = greater sensitivity."
    ),
    "p95_carbon_loss": (
        "P95 Carbon Loss (pp):\n"
        "95th percentile loss in carbon reduction performance under noisy inputs. "
        "Higher = more sensitivity in climate outcome."
    ),
    "p95_te_drift": (
        "P95 TE Drift (bps):\n"
        "95th percentile change in realized TE under perturbations. "
        "Higher = more sensitivity in tracking error stability."
    ),
    "composition_sensitivity": (
        "Composition Sensitivity:\n"
        "Measures how much optimal weights change under small input perturbations. "
        "Higher = more composition instability."
    ),
    "outcome_sensitivity": (
        "Outcome Sensitivity:\n"
        "Measures how carbon reduction and TE performance change under "
        "perturbations. Higher = less stable outcomes."
    ),
    "sensitivity_score": (
        "Sensitivity Score (0–1):\n"
        "Combined normalized score (inverted) capturing composition and outcome "
        "sensitivity. Higher = more stable, less sensitive."
    ),
}

METRIC_DESCRIPTIONS.update({
    "room_norm": (
        "Room for Maneuver (normalized 0–1): "
        "Position of the sector on the Room-for-Maneuver scale after global "
        "min–max normalization across sectors."
    ),
    "flex_norm": (
        "Flexibility (normalized 0–1): "
        "Position of the sector on the Flexibility scale after global "
        "min–max normalization across sectors."
    ),
    "sens_norm": (
        "Sensitivity (normalized 0–1): "
        "Position of the sector on the Sensitivity scale after global "
        "min–max normalization across sectors. Higher = less sensitive."
    ),
    "robust_norm": (
        "Robustness (normalized 0–1): "
        "Position of the sector on the Robustness scale after global "
        "min–max normalization across sectors."
    ),
    "dri": (
        "Decarbonization Readiness Index (DRI): "
        "Simple average of the four normalized dimensions (Room, Flexibility, "
        "Sensitivity, Robustness). Higher = greater overall readiness."
    ),
})

def info_metric(label, value, help_text="", delta=None):
    # convert \n to <br> for HTML tooltips
    help_html = (help_text or "").replace("\n", "<br>")

    tooltip_html = """
    <style>
    .tooltip-wrapper {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .tooltip-icon {
        background-color: #d0d0d0;
        color: #333;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: help;
        position: relative;
    }
    .tooltip-icon:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .tooltip-text {
        visibility: hidden;
        opacity: 0;
        width: 220px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 100;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        font-size: 12px;
        line-height: 1.3;
        transition: opacity 0.25s ease-in-out;
    }
    .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #555 transparent transparent transparent;
    }
    </style>
    """

    html_label = f"""
    {tooltip_html}
    <div class="tooltip-wrapper">
        <span>{label}</span>
        <span class="tooltip-icon">i
            <span class="tooltip-text">{help_html}</span>
        </span>
    </div>
    """

    # top row: label + tooltip
    st.markdown(html_label, unsafe_allow_html=True)

    # metric row (label hidden but non-empty for Streamlit)
    st.metric(
        label="value",
        value=value,
        delta=delta,
        label_visibility="collapsed",
    )

# ---------------------------------------------------------
# Streamlit CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="TE–Carbon Dashboard — Consumer Discretionary", layout="wide")

# ---------------------------------------------------------
# STYLE FIX — keep sidebar purple, make text white
# ---------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] label { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stSelectbox div[role="combobox"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] svg { fill: #FFFFFF !important; }

/* New compact radar card */


.radar-title {
    font-size: 14px;
    font-weight: 600;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("## TE–Carbon Dashboard — Consumer Discretionary")
st.caption("TE–Carbon frontier and marginal carbon gains for the Consumer Discretionary sector.")

from pathlib import Path

def format_period(p):
    month = p[:2]
    year = "20" + p[2:]
    return f"{month}/{year}"

# Load raw periods (e.g., "0321")
available_periods_raw = sorted([
    p.stem.split("_")[-1]
    for p in Path("results/optimal_portfolios").glob("optimal_portfolios_all_te_*.pkl")
])

# Display-friendly labels (e.g., "03/2021")
available_periods_display = [format_period(p) for p in available_periods_raw]

# Radio selector
selected_display = st.radio(
    "Select analysis period",
    options=available_periods_display,
    horizontal=True,
    index=len(available_periods_display) - 1
)

# Map back to raw tag
selected_period_raw = available_periods_raw[
    available_periods_display.index(selected_display)
]

st.write(f"### Showing results for period **{selected_display}**")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
sectors = pd.read_excel("data/datasets/dataset_comp_1223.xlsx")['GICS Sector'].unique()
with st.sidebar:
    st.header("Settings")
    sector_name = st.selectbox("Sector", sectors, index=0)


# ---------------------------------------------------------
# LOAD FRONTIER DATA for selected period
# ---------------------------------------------------------
pickle_path = Path("results/optimal_portfolios") / f"optimal_portfolios_all_te_{selected_period_raw}.pkl"

if not pickle_path.exists():
    st.error(f"No TE–Carbon frontier file found for period {selected_display}.")
    st.stop()

with open(pickle_path, "rb") as f:
    all_periods_data = pickle.load(f)

if sector_name not in all_periods_data:
    st.error(f"No data available for sector {sector_name} in period {selected_display}.")
    st.stop()

# Extract sector-specific frontier
data = all_periods_data[sector_name]

# TE in bps, CR in %
te = np.array(data["tracking_errors"])
cr = np.array(data["carbon_reductions"])

# Marginal gains (ΔCR / ΔTE)
marginal = np.gradient(cr, te)


# ---------------------------------------------------------
# PLOTLY STYLE TO MIMIC ACADEMIC LOOK
# ---------------------------------------------------------
line_color_frontier = "#3417d7"
line_color_marginal = "#dc29c4"
grid_color = "rgba(150,150,150,0.3)"

# ---------------------------------------------------------
# 1) TE–CARBON FRONTIER
# ---------------------------------------------------------
fig_frontier = go.Figure()
fig_frontier.add_trace(go.Scatter(
    x=te,
    y=cr,
    mode="lines+markers",
    line=dict(color=line_color_frontier, width=2),
    marker=dict(size=6),
    hovertemplate=(
        "<span style='font-size:14px'>"
        "TE (bps): %{x:.2f}<br>"
        "Carbon Reduction: %{y:.2f}%"
        "</span><extra></extra>"
    ),
))

fig_frontier.update_layout(
    title=f"{sector_name} — TE–Carbon Frontier",
    xaxis_title="Tracking Error (bps)",
    yaxis_title="Carbon Reduction (%)",
    template="simple_white",
    hoverlabel=dict(
        font_size=14,       # <<< Bigger hover text
        bgcolor="white",    # clean academic look
        font_color="black"
    )
)


# ---------------------------------------------------------
# 2) MARGINAL CARBON GAIN CURVE
# ---------------------------------------------------------
fig_marginal = go.Figure()
fig_marginal.add_trace(go.Scatter(
    x=te,
    y=marginal,
    mode="lines+markers",
    line=dict(color=line_color_marginal, width=2),
    marker=dict(size=6),
    hovertemplate=(
        "<span style='font-size:14px'>"
        "TE (bps): %{x:.2f}<br>"
        "Marginal Gain: %{y:.2f}"
        "</span><extra></extra>"
    ),
))

fig_marginal.update_layout(
    title=f"{sector_name} — Marginal Carbon Gain",
    xaxis_title="Tracking Error (bps)",
    yaxis_title="Marginal Gain (ΔCR / ΔTE)",
    template="simple_white",
    hoverlabel=dict(
        font_size=14,       # <<< Bigger hover text
        bgcolor="white",
        font_color="black"
    )
)


# ---------------------------------------------------------
# DISPLAY PLOTS
# ---------------------------------------------------------
col1, col2 = st.columns(2)
col1.plotly_chart(fig_frontier, use_container_width=True)
col2.plotly_chart(fig_marginal, use_container_width=True)

# ---------------------------------------------------------
# DRI SECTION (score + radar underneath title)
# ---------------------------------------------------------


# Load DRI table (normalized dimensions + DRI)
dri_df = pd.read_excel("results/DRI/decarbonization_readiness_index.xlsx")

st.markdown("### Decarbonization Readiness Index (DRI)")

# Load row for sector
row = dri_df.loc[dri_df["Sector"] == sector_name].iloc[0]

room_norm   = float(row["Room_norm"])
flex_norm   = float(row["Flex_norm"])
sens_norm   = float(row["Sens_norm"])
robust_norm = float(row["Robust_norm"])
dri_score   = float(row["DRI"])

# ---------- THREE COLUMNS ----------
col1, col2, col3 = st.columns([1.2, 0.8, 1.2])

# ---------------------------------------------------------
# 1) Radar Chart
# ---------------------------------------------------------
with col1:
    labels = ["Room for Maneveur", "Flexibility", "Sensitivity", "Robustness"]
    values = [room_norm, flex_norm, sens_norm, robust_norm] + [room_norm]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        fill="toself",
        line=dict(color="#6A5AE0", width=2),
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0,1], showticklabels=False, ticks=""),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        width=500,
        height=360,
        margin=dict(l=30, r=30, t=30, b=30),
        template="plotly_dark"
    )

    st.plotly_chart(fig_radar, config={"displayModeBar": False})


# ---------------------------------------------------------
# 2) Center table: DRI Score only
# ---------------------------------------------------------
with col2:
    st.markdown("#### Overall DRI Score")
    st.markdown(
        f"""
        <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
            <tr>
                <td style="font-size:16px; font-weight:600; color:white;">DRI</td>
                <td style="font-size:20px; font-weight:700; color:#9BE7FF; text-align:right;">
                    {dri_score:.3f}
                </td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 3) Table of the four dimension scores
# ---------------------------------------------------------
with col3:
    st.markdown("#### Dimension Scores")

    st.markdown(
        f"""
        <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
            <tr><th style='text-align:left;'>Dimension</th><th style='text-align:right;'>Score</th></tr>
            <tr><td>Room for Maneuver</td><td style='text-align:right; color:#A2FFAA;'>{room_norm:.3f}</td></tr>
            <tr><td>Flexibility</td><td style='text-align:right; color:#A2FFAA;'>{flex_norm:.3f}</td></tr>
            <tr><td>Sensitivity</td><td style='text-align:right; color:#A2FFAA;'>{sens_norm:.3f}</td></tr>
            <tr><td>Robustness</td><td style='text-align:right; color:#A2FFAA;'>{robust_norm:.3f}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# ROOM FOR MANEUVER METRICS
# ---------------------------------------------------------
st.markdown("### Room for Maneuver — Key Metrics")

# Convert TE from bps → percent for consistency with formulas
te_pct = te / 100.0
cr_pct = cr

# --- Helper functions ---
def interp_at_x(x, xgrid, ygrid):
    if x <= xgrid.min():
        return ygrid[0]
    if x >= xgrid.max():
        return ygrid[-1]
    idx = np.searchsorted(xgrid, x)
    x0, x1 = xgrid[idx-1], xgrid[idx]
    y0, y1 = ygrid[idx-1], ygrid[idx]
    return y0 + (y1-y0)*(x - x0)/(x1 - x0 + 1e-12)

def finite_slope(x, xgrid, ygrid, h=0.05):
    y_plus = interp_at_x(x + h, xgrid, ygrid)
    y_minus = interp_at_x(x - h, xgrid, ygrid)
    return (y_plus - y_minus) / (2*h)

def elasticity_at_x(x, xgrid, ygrid):
    y = interp_at_x(x, xgrid, ygrid)
    slope = finite_slope(x, xgrid, ygrid)
    if y == 0:
        return np.nan
    return slope * (x / y)

def auc_to_xmax(xgrid, ygrid, xmax=5.0):
    mask = xgrid <= xmax
    X = xgrid[mask]
    Y = ygrid[mask]
    if len(X) < 2:
        return np.nan
    return np.trapz(Y, X)

# --- Compute KPIs ---
slope_2pct     = finite_slope(2.0, te_pct, cr_pct)
elasticity     = elasticity_at_x(2.0, te_pct, cr_pct)
auc_5pct       = auc_to_xmax(te_pct, cr_pct, xmax=5.0)
max_cut_5pct   = interp_at_x(5.0, te_pct, cr_pct)

# TE required for 50% reduction
target = 50.0
indices = np.where(cr_pct >= target)[0]
if len(indices) == 0:
    te50_label = "Not reached"
else:
    k = indices[0]
    te50 = te_pct[k] * 100.0   # convert back to bps
    te50_label = f"{te50:.0f} bps"


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    info_metric(
        label="Slope @ 2% TE",
        value=f"{slope_2pct:.2f}",
        help_text=METRIC_DESCRIPTIONS["slope_2pct"]
    )

with col2:
    info_metric(
        label="Elasticity @ 2% TE",
        value=f"{elasticity:.2f}",
        help_text=(
            "Elasticity of carbon reduction at 2% TE: "
            "percentage change in carbon reduction per 1% change in TE."
        )
    )

with col3:
    info_metric(
        label="AUC ≤ 5% TE",
        value=f"{auc_5pct:.2f}",
        help_text=METRIC_DESCRIPTIONS["auc_5pct"]
    )

with col4:
    info_metric(
        label="Max Cut @ 5% TE",
        value=f"{max_cut_5pct:.1f}%",
        help_text=METRIC_DESCRIPTIONS["max_cut_5pct"]
    )

with col5:
    info_metric(
        label="Min TE for 50% Cut",
        value=te50_label,
        help_text=METRIC_DESCRIPTIONS["min_te_50pct_cut"]
    )

# ---------------------------------------------------------
# FLEXIBILITY METRICS (period 1223)
# ---------------------------------------------------------
st.markdown("### Flexibility — Key Metrics")

# Load flexibility panel
try:
    flex_panel = pd.read_excel(
        "results/flexibility/sector_flexibility_raw.xlsx",
        dtype={"Period": str}
    )
except:
    st.error("❌ Could not load results/flexibility/sector_flexibility_raw.xlsx")
    st.stop()

flex_cd = flex_panel[
    (flex_panel["Sector"] == sector_name) &
    (flex_panel["Period"] == selected_period_raw)
]


if flex_cd.empty:
    st.warning(f"⚠️ No flexibility data found for {sector_name} in {selected_display}.")
else:
    row = flex_cd.iloc[0]

    avg_bw        = float(row["Avg_bandwidth"])
    med_bw        = float(row["Median_bandwidth"])
    max_bw        = float(row["Max_bandwidth"])
    pct_zero_min  = float(row["Pct_wmin_zero"])
    l2_lb         = float(row["L2_lower_bound_same_obj"])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        info_metric(
            label="Avg ε-bandwidth",
            value=f"{avg_bw:.2%}",
            help_text=METRIC_DESCRIPTIONS["avg_bandwidth"]
        )

    with col2:
        info_metric(
            label="Median ε-bandwidth",
            value=f"{med_bw:.2%}",
            help_text=METRIC_DESCRIPTIONS["median_bandwidth"]
        )

    with col3:
        info_metric(
            label="Max ε-bandwidth",
            value=f"{max_bw:.2%}",
            help_text=METRIC_DESCRIPTIONS["max_bandwidth"]
        )

    with col4:
        info_metric(
            label="% weights at min=0",
            value=f"{pct_zero_min:.1f}%",
            help_text="Percentage of stocks whose lower ε-band is exactly zero (fully flexible downward)."
        )

    with col5:
        info_metric(
            label="L2 lower bound",
            value=f"{l2_lb:.3f}",
            help_text=METRIC_DESCRIPTIONS["l2_lower_bound"]
        )

# ---------------------------------------------------------
# SENSITIVITY METRICS (period 1223)
# ---------------------------------------------------------
st.markdown("### Sensitivity — Key Metrics")

# Load sensitivity panel
try:
    sens_panel = pd.read_excel(
        "results/sensitivity/sensitivity_scores_by_period.xlsx",
        dtype={"Period": str}
    )
except:
    st.error("❌ Could not load results/sensitivity/sensitivity_scores_by_period.xlsx")
    st.stop()

sens_cd = sens_panel[
    (sens_panel["Sector"] == sector_name) &
    (sens_panel["Period"] == selected_period_raw)
]

if sens_cd.empty:
    st.warning(f"⚠️ No sensitivity data found for {sector_name} in {selected_display}.")
else:
    row = sens_cd.iloc[0]

    median_turnover = float(row["Median_Turnover_pct"])
    median_cosine   = float(row["Median_Cosine"])
    p95_carbloss    = float(row["P95_CarbonLoss_pp"])
    p95_tedrift     = float(row["P95_TE_Drift_bps"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        info_metric(
            label="Median Turnover (%)",
            value=f"{median_turnover:.2f}%",
            help_text=METRIC_DESCRIPTIONS["median_turnover"]
        )

    with col2:
        info_metric(
            label="Median Cosine Similarity",
            value=f"{median_cosine:.3f}",
            help_text=METRIC_DESCRIPTIONS["median_cosine_inv"]
        )

    with col3:
        info_metric(
            label="P95 Carbon Loss (pp)",
            value=f"{p95_carbloss:.2f}",
            help_text=METRIC_DESCRIPTIONS["p95_carbon_loss"]
        )

    with col4:
        info_metric(
            label="P95 TE Drift (bps)",
            value=f"{p95_tedrift * 10000:.1f} bps",
            help_text=METRIC_DESCRIPTIONS["p95_te_drift"]
        )
# ---------------------------------------------------------
# ROBUSTNESS METRICS (period 1223)
# ---------------------------------------------------------
st.markdown("### Robustness — Key Metrics")

try:
    robustness_panel = pd.read_excel(
        "results/robustness/robustness_scores_by_period.xlsx",
        dtype={"period": str}
    )
except Exception as e:
    st.error("❌ Could not load results/robustness/robustness_scores_by_period.xlsx")
    st.stop()

# Clean up column names
robustness_panel.columns = robustness_panel.columns.str.strip()

# Rename for consistency with other panels
if "sector" in robustness_panel.columns:
    robustness_panel = robustness_panel.rename(columns={"sector": "Sector"})
if "period" in robustness_panel.columns:
    robustness_panel = robustness_panel.rename(columns={"period": "Period"})

# Filter for selected sector + period 1223
robust_cd = robustness_panel[
    (robustness_panel["Sector"] == sector_name) &
    (robustness_panel["Period"] == selected_period_raw)
]

if robust_cd.empty:
    st.warning(f"⚠️ No robustness data found for {sector_name} in {selected_display}.")
else:
    row = robust_cd.iloc[0]

    te_ann       = float(row["annualized_TE"])          # decimal (e.g. 0.02)
    vol_ann      = float(row["annualized_volatility"])  # decimal
    rob_ratio    = float(row["Robustness_Ratio"])
    rob_score    = float(row["Robustness_Score"])

    te_bps       = te_ann * 10000.0
    vol_pct      = vol_ann * 100.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        info_metric(
            label="Annualized TE (bps)",
            value=f"{te_bps:.1f}",
            help_text=METRIC_DESCRIPTIONS["annualized_te"]
        )

    with col2:
        info_metric(
            label="Annualized Volatility (%)",
            value=f"{vol_pct:.2f}%",
            help_text=METRIC_DESCRIPTIONS["annualized_volatility"]
        )

   