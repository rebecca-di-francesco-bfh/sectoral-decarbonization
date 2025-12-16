
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import plotly.graph_objects as go
import json

OUT_OF_SAMPLE_FREQ = 'daily'

with open("metric_descriptions.json", "r") as f:
    METRIC_DESCRIPTIONS = json.load(f)

st.set_page_config(page_title="TE–Carbon Dashboard for", layout="wide")

st.markdown("""
<style>
div.block-container {
    max-width: 1500px;
    padding-left: 3rem;
    padding-right: 3rem;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


def info_metric(label, value, help_text="", delta=None):
    help_html = (help_text or "").replace("\n", "<br>")

    st.markdown("""
    <style>
    .metric-card {
        border: 1px solid #555;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 4px;
        background-color: #1e1e1e;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #ddd;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: white;
    }
    .delta-pos { color: #00d26a; font-size: 14px; }
    .delta-neg { color: #ff4b4b; font-size: 14px; }
    .tooltip-icon {
        background-color: #d0d0d0;
        color: #333;
        border-radius: 50%;
        width: 15px;
        height: 15px;
        font-size: 10px;
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
    </style>
    """, unsafe_allow_html=True)

    # Delta formatted
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            cls = "delta-pos" if delta >= 0 else "delta-neg"
            sign = "+" if delta >= 0 else ""
            delta_html = f"<span class='{cls}'>{sign}{delta}</span>"

    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">
            {label}
            <span class="tooltip-icon">i
                <span class="tooltip-text">{help_html}</span>
            </span>
        </div>
        <div class="metric-value">{value} {delta_html}</div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
st.markdown("""
<style>

    /* ---- SELECTBOX: glowing selected text ---- */
    div[data-baseweb="select"] span {
        color: #F2F2F2 !important;
        font-weight: 300 !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.45) !important;
    }

    /* ---- RADIO: glowing selected period ---- */
    div[role="radio"][aria-checked="true"] ~ label {
        color: #F2F2F2 !important;
        font-weight: 300 !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.45) !important;
    }

</style>
""", unsafe_allow_html=True)




st.markdown("""
<style>
.section-title {
    font-size: 22px;
    font-weight: 700;
    padding: 10px 18px;
    background-color: #111;  /* matches dark theme */
    border-left: 4px solid #6A5AE0;  /* accent color */
    border-radius: 6px;
    margin-top: 30px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


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
# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
sectors = pd.read_excel("data/datasets/dataset_comp_1223.xlsx")['GICS Sector'].unique()

container = st.container(border=True)

with container:
    st.markdown('<div class="settings-box">', unsafe_allow_html=True)

    st.markdown("### Settings")

    sector_name = st.selectbox(
        "Select sector:",
        sectors,
        index=0,
        key="sector_selector"
    )

    selected_display = st.radio(
        "Select analysis period:",
        options=available_periods_display,
        horizontal=True,
        index=len(available_periods_display) - 1
    )

    st.markdown('</div>', unsafe_allow_html=True)




# Map back to raw tag
selected_period_raw = available_periods_raw[
    available_periods_display.index(selected_display)
]



HIGHLIGHT = "#F2F2F2"   # soft luminous white

st.markdown(
    f"""
    <div style="margin: 5px 0 20px 0;">
        <h1 style="font-weight:700; margin-bottom:6px;">
            TE–Carbon Dashboard for 
            <span style="
                color: #F2F2F2;
                font-weight: 300;
                text-shadow: 0px 0px 20px rgba(255, 215, 0, 0.8);
            ">
                {sector_name}
            </span>
            · 
            <span style="
                color: #F2F2F2;
                font-weight: 300;
                text-shadow: 0px 0px 20px rgba(255, 215, 0, 0.8);
            ">
                {selected_display}
            </span>
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)








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
# COMPUTE ELBOW POINT (L-curve method: max distance to chord)
# ---------------------------------------------------------
# Endpoints of frontier
x0, y0 = te[0], cr[0]
x1, y1 = te[-1], cr[-1]

# Perpendicular distance of each point to the line AB
distances = np.abs((te - x0)*(y1 - y0) - (cr - y0)*(x1 - x0)) / np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

# Index of elbow
elbow_idx = np.argmax(distances)
elbow_te = te[elbow_idx]
elbow_cr = cr[elbow_idx]


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
    showlegend=False)
)
# Add elbow marker
# Add elbow point WITHOUT showing it in the legend
fig_frontier.add_trace(go.Scatter(
    x=[elbow_te],
    y=[elbow_cr],
    mode="markers+text",
    marker=dict(color="red", size=12, symbol="circle"),
    text=[f"Elbow ({elbow_te:.1f} bps)"],
    textposition="top center",
    hovertemplate="<b>Elbow Point</b><br>TE: %{x:.2f} bps<br>CR: %{y:.2f}%<extra></extra>",
    showlegend=False  # ← IMPORTANT: hides from legend
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


# # -----------------------------------------------------------
# # 3) SECOND DERIVATIVE CURVATURE PLOT -> DISABLED: VERY NOISY
# # -----------------------------------------------------------
# second_derivative = np.gradient(np.gradient(cr, te), te)

# # Normalize for plotting (optional)
# second_derivative_norm = (second_derivative - second_derivative.min()) / \
#                          (second_derivative.max() - second_derivative.min())
# curv_idx = np.argmax(second_derivative)
# curv_te  = te[curv_idx]
# curv_cr  = cr[curv_idx]
# curv_val = second_derivative[curv_idx]
# fig_second = go.Figure()

# fig_second.add_trace(go.Scatter(
#     x=te,
#     y=second_derivative,
#     mode="lines+markers",
#     line=dict(color="#ff9900", width=2),
#     marker=dict(size=6),
#     name="Second Derivative",
#     hovertemplate=(
#         "<span style='font-size:14px'>"
#         "TE (bps): %{x:.2f}<br>"
#         "Second Derivative: %{y:.4f}"
#         "</span><extra></extra>"
#     ),
# ))

# # Mark maximum curvature
# fig_second.add_trace(go.Scatter(
#     x=[curv_te],
#     y=[curv_val],
#     mode="markers+text",
#     marker=dict(size=12, color="red", symbol="diamond"),
#     text=["max curvature"],
#     textposition="top center",
#     name="Max Curvature",
# ))

# fig_second.update_layout(
#     title=f"{sector_name} — Second Derivative of the TE–Carbon Curve",
#     xaxis_title="Tracking Error (bps)",
#     yaxis_title="Second Derivative (Curvature)",
#     template="simple_white",
#     hoverlabel=dict(
#         font_size=14,
#         bgcolor="white",
#         font_color="black"
#     )
# )
# st.plotly_chart(fig_second, use_container_width=True)


# ---------------------------------------------------------
# DISPLAY PLOTS
# ---------------------------------------------------------
col1, col2 = st.columns(2)
col1.plotly_chart(fig_frontier, use_container_width=True)
col2.plotly_chart(fig_marginal, use_container_width=True)
st.caption("TE–Carbon frontier and marginal carbon gains for the Consumer Discretionary sector.")


# ---------------------------------------------------------
# ROOM FOR MANEUVER METRICS
# ---------------------------------------------------------
st.markdown('<div class="section-title">Room for Maneuver — Key Metrics</div>', unsafe_allow_html=True)

rfm_panel = pd.read_excel(
    "results/room_for_maneuver/room_for_maneuver_scores_by_period.xlsx",
    dtype={"Period": str}
)

rfm_row = rfm_panel[
    (rfm_panel["Sector"] == sector_name) &
    (rfm_panel["Period"] == selected_period_raw)
].iloc[0]

c_at_1pct   = rfm_row["C_at_1pct"]
auc_2pct    = rfm_row["AUC_to_2pctTE"]

te50_raw = rfm_row["TE_for_50pctCut"]
te50_label = (
    f"{te50_raw * 10000:.0f} bps" if pd.notna(te50_raw) else "Not reached"
)

room_for_maneuver_score = rfm_row["Room_for_Maneuver_Score"]

col1, col2, col3  = st.columns(3)

with col1:
    info_metric("Early carbon reduction (1% TE)", f"{c_at_1pct*100:.0f}%",
                help_text=METRIC_DESCRIPTIONS["c_at_1pct"])

with col2:
    info_metric("AUC ≤ 2% TE", f"{auc_2pct:.2f}",
                help_text=METRIC_DESCRIPTIONS["auc_2pct"])

with col3:
    info_metric("TE for 50% reduction", te50_label,
                help_text=METRIC_DESCRIPTIONS["te_50pct"])

with st.expander("📘 Interpretation of the Room for Maneuver metrics"):
    st.markdown(f"""
    <div style='line-height:1.55; font-size:15px;'>

    <p><b>Early carbon reduction (1% TE)</b><br>
    Sector <b>{sector_name}</b> can achieve 
    <b>{c_at_1pct*100:.0f}%</b> of its total achievable decarbonization 
    (<b>{cr[-1]:.0f}%</b> at 5% TE) while staying within a very tight 
    tracking-error budget of 1%. This reflects how much decarbonization is 
    available with <i>minimal</i> active risk.</p>

    <p><b>Early decarbonization space (AUC 0–2% TE)</b><br>
    The sector's AUC score of <b>{auc_2pct:.2f}</b> in the 0–2% TE region indicates 
    the amount of decarbonization potential available before TE constraints become 
    restrictive. Higher values imply greater room to maneuver under tight TE budgets.</p>

    <p><b>TE required for 50% of maximum decarbonization</b><br>
    The sector requires <b>{te50_raw *10000:.0f} bps</b> of tracking error to 
    achieve half of its maximum attainable decarbonization. 
    Lower TE thresholds indicate easier access to mid-level decarbonization 
    within realistic benchmark-relative risk budgets.</p>

    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# FLEXIBILITY METRICS (period 1223)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Flexibility — Key Metrics</div>', unsafe_allow_html=True)


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
st.markdown('<div class="section-title">Sensitivity — Key Metrics</div>', unsafe_allow_html=True)


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


    col1, col2, col3  = st.columns(3)

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

 
# ---------------------------------------------------------
# ROBUSTNESS METRICS (period 1223)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Robustness — Key Metrics</div>', unsafe_allow_html=True)



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

    te_ann       = float(row[f"{OUT_OF_SAMPLE_FREQ}_TE"])          # decimal (e.g. 0.02)
    vol_ann      = float(row[f"{OUT_OF_SAMPLE_FREQ}_volatility"])  # decimal
    rob_ratio    = float(row["Robustness_Ratio"])
    rob_score    = float(row["Robustness_Score"])

    te_bps       = te_ann * 10000.0
    vol_pct      = vol_ann * 100.0

    col1, col2 = st.columns(2)

    with col1:
        info_metric(
            label=f"Out-of-sample Tracking Error (bps)",
            value=f"{te_bps:.1f}",
            help_text=METRIC_DESCRIPTIONS[f"{OUT_OF_SAMPLE_FREQ}_te"]
        )

    with col2:
        info_metric(
            label=f"Sector Benchmark Volatility",
            value=f"{vol_pct:.2f}%",
            help_text=METRIC_DESCRIPTIONS[f"{OUT_OF_SAMPLE_FREQ}_volatility"]
        )


# Add vertical spacing before DRI
st.markdown("<br><br>", unsafe_allow_html=True)

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

# ---------- THREE COLUMNS (Radar • DRI Table • Compare) ----------
col1, col2, col3 = st.columns([1.2, 0.8, 0.8])

# ---------------------------------------------------------
# 1) Compare selector (col3)
# ---------------------------------------------------------
with col3:
    # Remove the currently selected sector from the list
    compare_options = ["None"] + [s for s in sectors if s != sector_name]

    compare_sector = st.selectbox(
        "Compare with:",
        options=compare_options,
        index=0,
        key="compare_selector"
    )


# ---------------------------------------------------------
# Load comparison sector data (if chosen)
# ---------------------------------------------------------
if compare_sector != "None":
    comp_row = dri_df.loc[dri_df["Sector"] == compare_sector].iloc[0]

    comp_room_norm   = float(comp_row["Room_norm"])
    comp_flex_norm   = float(comp_row["Flex_norm"])
    comp_sens_norm   = float(comp_row["Sens_norm"])
    comp_robust_norm = float(comp_row["Robust_norm"])
    comp_dri_score   = float(comp_row["DRI"])

# ---------------------------------------------------------
# 2) RADAR CHART with overlay if comparing
# ---------------------------------------------------------
with col1:
    labels = ["Room for Maneveur", "Flexibility", "Sensitivity", "Robustness"]
    
    # Main sector polygon
    r_main = [room_norm, flex_norm, sens_norm, robust_norm, room_norm]
    
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=r_main,
        theta=labels + [labels[0]],
        fill="toself",
        name=sector_name,
        line=dict(color="#6A5AE0", width=3),
        opacity=0.85
    ))

    # Add comparison polygon
    if compare_sector != "None":
        r_comp = [
            comp_room_norm,
            comp_flex_norm,
            comp_sens_norm,
            comp_robust_norm,
            comp_room_norm
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=r_comp,
            theta=labels + [labels[0]],
            fill="toself",
            name=compare_sector,
            line=dict(color="#FF9F1C", width=3),
            opacity=0.55
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0,1], showticklabels=False, ticks=""),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        width=500,
        height=400,
        margin=dict(l=30, r=30, t=40, b=40),
        template="plotly_dark"
    )

    st.plotly_chart(fig_radar, config={"displayModeBar": False})

# ---------------------------------------------------------
# 3) DRI SCORE TABLE + DIMENSION SCORE TABLE (col2)
#     → Fully compares both sectors if selected
# ---------------------------------------------------------
with col2:

    # ---------- DRI SCORE ----------
    st.markdown("#### Overall DRI Score")

    if compare_sector == "None":
        # Single-sector table
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
            """, unsafe_allow_html=True)
    else:
        # Two-column comparison table
        st.markdown(
            f"""
            <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                <tr>
                    <th></th>
                    <th style='text-align:right;'>{sector_name}</th>
                    <th style='text-align:right;'>{compare_sector}</th>
                </tr>
                <tr>
                    <td>DRI</td>
                    <td style='text-align:right; color:#9BE7FF;'>{dri_score:.3f}</td>
                    <td style='text-align:right; color:#FFCE9B;'>{comp_dri_score:.3f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

    # ---------- DIMENSION SCORES ----------
    st.markdown("#### Dimension Scores")

    if compare_sector == "None":
        # Single-sector table
        st.markdown(
            f"""
            <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                <tr><th>Dimension</th><th style='text-align:right;'>Score</th></tr>
                <tr><td>Room for Maneuver</td><td style='text-align:right; color:#A2FFAA;'>{room_norm:.3f}</td></tr>
                <tr><td>Flexibility</td><td style='text-align:right; color:#A2FFAA;'>{flex_norm:.3f}</td></tr>
                <tr><td>Sensitivity</td><td style='text-align:right; color:#A2FFAA;'>{sens_norm:.3f}</td></tr>
                <tr><td>Robustness</td><td style='text-align:right; color:#A2FFAA;'>{robust_norm:.3f}</td></tr>
            </table>
            """, unsafe_allow_html=True)

    else:
        # Two-column comparison table
        st.markdown(
            f"""
            <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                <tr>
                    <th>Dimension</th>
                    <th style='text-align:right;'>{sector_name}</th>
                    <th style='text-align:right;'>{compare_sector}</th>
                </tr>
                <tr>
                    <td>Room for Maneuver</td>
                    <td style='text-align:right; color:#A2FFAA;'>{room_norm:.3f}</td>
                    <td style='text-align:right; color:#FFD8A2;'>{comp_room_norm:.3f}</td>
                </tr>
                <tr>
                    <td>Flexibility</td>
                    <td style='text-align:right; color:#A2FFAA;'>{flex_norm:.3f}</td>
                    <td style='text-align:right; color:#FFD8A2;'>{comp_flex_norm:.3f}</td>
                </tr>
                <tr>
                    <td>Sensitivity</td>
                    <td style='text-align:right; color:#A2FFAA;'>{sens_norm:.3f}</td>
                    <td style='text-align:right; color:#FFD8A2;'>{comp_sens_norm:.3f}</td>
                </tr>
                <tr>
                    <td>Robustness</td>
                    <td style='text-align:right; color:#A2FFAA;'>{robust_norm:.3f}</td>
                    <td style='text-align:right; color:#FFD8A2;'>{comp_robust_norm:.3f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
