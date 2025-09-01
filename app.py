# Streamlit dashboard for TE–Carbon optimization by sector
# --------------------------------------------------------
# Features:
# - Select sector and annualized tracking error (TE)
# - Original weights vs optimized weights per stock
# - Carbon intensity reduction
# - Market concentration vs optimized portfolio concentration (HHI)
#
# Note: this file reuses the logic of your script. Make sure the Excel file paths
# are correct as in your project.

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Config Streamlit
# ---------------------------
st.set_page_config(page_title="TE–Carbon Dashboard", layout="wide")

# ---- UI tweaks: headings ----
st.markdown(
    """
<style>
.big-title { font-size: 2.0rem; font-weight: 700; margin: 0.75rem 0 0.25rem 0; }
.sub-title { font-size: 2.0rem; font-weight: 700; margin: 0.75rem 0 0.25rem 0; }
.section-gap { margin-top: 0.75rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("TE–Carbon Dashboard")
st.caption(
    "Select a sector and a tracking error level to compare weights, carbon intensity, and concentration."
)

# ---------------------------
# Utility functions
# ---------------------------

def compute_hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index (sum of squared weights)."""
    w = np.asarray(weights).flatten()
    return float(np.sum(np.square(w)))


def sigma_reg_fn(R_clean: pd.DataFrame):
    """Ledoit–Wolf shrinkage + small λI ridge."""
    lw = LedoitWolf().fit(R_clean)
    Sigma_shrink = lw.covariance_
    lambda_I = 1e-5
    Sigma_reg = Sigma_shrink + lambda_I * np.eye(Sigma_shrink.shape[0])
    return Sigma_reg, float(lw.shrinkage_)


def nearest_psd(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def optimize_sector(sector_df: pd.DataFrame, R_sector: pd.DataFrame, te_annual: float):
    """
    Solve carbon-minimization under TE cap for a single sector.
    te_annual in decimal (e.g., 0.02 = 2%).
    Returns dict with weights and diagnostics.
    """
    # Drop the 'Date' column — keep only numeric log returns
    if "Date" in R_sector.columns:
        R_numeric = R_sector.drop(columns=["Date"])
    else:
        R_numeric = R_sector.copy()
    R_clean = R_numeric.dropna()
    stock_labels = R_clean.columns.values

    # Sanity alignment: ensure same ordering as in sector_df
    assert (
        sector_df["SYMBOL"].values == stock_labels
    ).all(), "SYMBOL alignment between data and returns does not match"

    Sigma_sector, shrinkage_alpha = sigma_reg_fn(R_clean)
    Sigma_sector = nearest_psd(Sigma_sector)

    w_b_vec = sector_df["weight_in_sector"].values.astype(float)
    c_vec = sector_df["Carbon Intensity"].values.astype(float)

    N = len(w_b_vec)
    w = cp.Variable(N)
    te_cap_var_monthly = (te_annual / np.sqrt(12)) ** 2

    tracking_error = cp.quad_form(w - w_b_vec, cp.psd_wrap(Sigma_sector))
    constraints = [tracking_error <= te_cap_var_monthly, cp.sum(w) == 1, w >= 0]
    objective = cp.Minimize(cp.sum(cp.multiply(w, c_vec)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        raise RuntimeError(f"Optimization failed: status {prob.status}")

    w_opt = w.value

    diff = w_opt - w_b_vec
    te_real = float(np.sqrt(diff.T @ Sigma_sector @ diff) * np.sqrt(12))  # annualized

    carbon_b = float(w_b_vec @ c_vec)
    carbon_opt = float(w_opt @ c_vec)
    reduction_pct = (carbon_b - carbon_opt) / carbon_b * 100.0 if carbon_b != 0 else 0.0

    return {
        "w_b_vec": w_b_vec,
        "w_opt": w_opt,
        "stock_labels": stock_labels,
        "te_real_bps": te_real * 10000.0,
        "carbon_b": carbon_b,
        "carbon_opt": carbon_opt,
        "reduction_pct": reduction_pct,
        "hhi_bench": compute_hhi(w_b_vec),
        "hhi_opt": compute_hhi(w_opt),
        "shrinkage_alpha": shrinkage_alpha,
    }


# ---------------------------
# Data loading (cache)
# ---------------------------

@st.cache_data(show_spinner=True)
def load_core_data():
    # Adjust paths as needed
    df = pd.read_excel("Data/dataset_comp_1222_without_ceg_ogn.xlsx")
    return df


@st.cache_data(show_spinner=False)
def load_sector_returns(sector_name: str) -> pd.DataFrame:
    R = pd.read_excel(
        "Data/sector_log_returns_comp_1222_without_ceg_ogn.xlsx", sheet_name=sector_name
    )
    return R


# ---------- Frontier + KPIs helpers ----------

def solve_min_carbon_given_te(
    sector_df: pd.DataFrame, R_sector: pd.DataFrame, te_annual: float
):
    """One solve of your existing problem, reused for frontier building."""
    if "Date" in R_sector.columns:
        R_numeric = R_sector.drop(columns=["Date"])
    else:
        R_numeric = R_sector.copy()
    R_clean = R_numeric.dropna()
    Sigma_sector, _ = sigma_reg_fn(R_clean)
    Sigma_sector = nearest_psd(Sigma_sector)

    w_b_vec = sector_df["weight_in_sector"].values.astype(float)
    c_vec = sector_df["Carbon Intensity"].values.astype(float)

    N = len(w_b_vec)
    w = cp.Variable(N)
    te_var_monthly_cap = (te_annual / np.sqrt(12)) ** 2
    tracking_error = cp.quad_form(w - w_b_vec, cp.psd_wrap(Sigma_sector))

    constraints = [tracking_error <= te_var_monthly_cap, cp.sum(w) == 1, w >= 0]
    objective = cp.Minimize(cp.sum(cp.multiply(w, c_vec)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None

    w_opt = w.value
    diff = w_opt - w_b_vec
    # realized TE (annualized)
    te_real = float(np.sqrt(diff.T @ Sigma_sector @ diff) * np.sqrt(12))
    carbon_b = float(w_b_vec @ c_vec)
    carbon_opt = float(w_opt @ c_vec)
    reduction_pct = (carbon_b - carbon_opt) / carbon_b * 100.0 if carbon_b != 0 else 0.0
    return te_real, reduction_pct


@st.cache_data(show_spinner=False)
def build_frontier(
    sector_df: pd.DataFrame, R_sector: pd.DataFrame, te_min=0.002, te_max=0.05, n_pts=80
):
    """
    Sweep TE in [te_min, te_max] (annual %) and compute realized TE and carbon reduction (%) at each step.
    Returns a DataFrame with columns: TE_pct, Reduction_pct (both floats).
    """
    te_grid = np.linspace(te_min, te_max, n_pts)  # in decimals
    rows = []
    for te_a in te_grid:
        res = solve_min_carbon_given_te(sector_df, R_sector, te_a)
        if res is None:
            continue
        te_real, red_pct = res
        rows.append((te_real * 100.0, red_pct))  # TE in percent for plotting

    if not rows:
        return pd.DataFrame(columns=["TE_pct", "Reduction_pct"])
    df = pd.DataFrame(rows, columns=["TE_pct", "Reduction_pct"]).dropna()
    # ensure monotone TE
    df = df.sort_values("TE_pct").drop_duplicates(subset=["TE_pct"])
    return df.reset_index(drop=True)


def interp_at_x(x, xgrid, ygrid):
    """Linear interpolation y(x) given monotone xgrid."""
    if len(xgrid) == 0:
        return np.nan
    xgrid = np.asarray(xgrid)
    ygrid = np.asarray(ygrid)
    if x <= xgrid.min():
        return float(ygrid[0])
    if x >= xgrid.max():
        return float(ygrid[-1])
    idx = np.searchsorted(xgrid, x)
    x0, x1 = xgrid[idx - 1], xgrid[idx]
    y0, y1 = ygrid[idx - 1], ygrid[idx]
    w = (x - x0) / (x1 - x0 + 1e-12)
    return float(y0 + w * (y1 - y0))


def finite_slope(x, xgrid, ygrid, h=0.05):
    """
    Central finite difference slope dy/dx at x.
    x, xgrid are in TE percent (e.g., 2.0), ygrid in % reduction.
    """
    y_plus = interp_at_x(x + h, xgrid, ygrid)
    y_minus = interp_at_x(x - h, xgrid, ygrid)
    if np.isnan(y_plus) or np.isnan(y_minus):
        return np.nan
    return (y_plus - y_minus) / (2 * h)


def elasticity_at_x(x, xgrid, ygrid):
    """
    Elasticity = (dY/Y) / (dX/X) = slope * (X / Y), with X=TE%, Y=Reduction%.
    """
    y = interp_at_x(x, xgrid, ygrid)
    s = finite_slope(x, xgrid, ygrid)
    if y is None or np.isnan(y) or y == 0 or np.isnan(s):
        return np.nan
    return s * (x / y)


def auc_to_xmax(xgrid, ygrid, xmax=5.0):
    """
    Trapezoidal area under curve from min(xgrid) to min(xmax, max(xgrid)).
    X, Y are in percent units; returns "percent^2" units (fine for ranking).
    """
    if len(xgrid) < 2:
        return np.nan
    # clip range
    mask = xgrid <= xmax + 1e-12
    X = xgrid[mask]
    Y = ygrid[mask]
    if len(X) < 2:
        return np.nan
    return float(np.trapz(Y, X))


def min_te_for_target_cut(xgrid, ygrid, target_pct=50.0):
    """
    Smallest TE% where Reduction_pct >= target_pct. Returns NaN if not achievable.
    """
    X = np.asarray(xgrid)
    Y = np.asarray(ygrid)
    if len(X) == 0:
        return np.nan
    # if already above target at the start:
    if Y[0] >= target_pct:
        return float(X[0])
    # find first crossing
    above = np.where(Y >= target_pct)[0]
    if len(above) == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return float(X[0])
    # linear interpolate between k-1 and k
    x0, x1 = X[k - 1], X[k]
    y0, y1 = Y[k - 1], Y[k]
    if y1 == y0:
        return float(x1)
    w = (target_pct - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


# ---------------------------
# Sidebar controls
# ---------------------------
try:
    data = load_core_data()
except Exception as e:
    st.error("Error loading main file. Check path: Data/dataset_comp_1222_without_ceg_ogn.xlsx")
    st.exception(e)
    st.stop()

sectors = sorted(data["GICS Sector"].dropna().unique())

with st.sidebar:
    st.header("Settings")
    sector_name = st.selectbox("Sector", sectors, index=0)
    te_percent = st.slider(
        "Annualized Tracking Error (%)", min_value=0.2, max_value=5.0, value=2.0, step=0.1
    )
    te_decimal = te_percent / 100.0
    st.caption("Tip: 2% ≈ 200 bps")

# Filter sector data
sector_df = data.loc[data["GICS Sector"] == sector_name].copy()

# Checks
if not np.isclose(sector_df["weight_in_sector"].sum(), 1.0):
    st.warning("Warning: benchmark weights in the sector do not sum to 1. Check data.")

if sector_df["Carbon Intensity"].isna().any():
    st.warning("Warning: there are NaNs in the Carbon Intensity of the sector.")

# Load returns for selected sector
try:
    R_sector = load_sector_returns(sector_name)
except Exception as e:
    st.error("Error loading sector log-returns. Check second Excel file and sheet name.")
    st.exception(e)
    st.stop()

# Run optimization
try:
    result = optimize_sector(sector_df, R_sector, te_decimal)
except Exception as e:
    st.error("Optimization failed. Check data, symbol alignment, and solver ECOS.")
    st.exception(e)
    st.stop()

# ---------------------------
# Layout
# ---------------------------
st.markdown(f'<div class="big-title">{sector_name}</div>', unsafe_allow_html=True)

# First row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Realized TE (bps)", f"{result['te_real_bps']:.1f}")
with col2:
    st.metric("Carbon Intensity – Benchmark", f"{result['carbon_b']:.2f}")
with col3:
    st.metric("Carbon Intensity – Optimized", f"{result['carbon_opt']:.2f}")

# Second row (aligned under the first)
col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Reduction (%)", f"{result['reduction_pct']:.2f}")
with col5:
    st.metric("HHI – Benchmark", f"{result['hhi_bench']:.4f}")
with col6:
    st.metric("HHI – Optimized", f"{result['hhi_opt']:.4f}")

# ---------- Room for Maneuver KPIs + Frontier ----------

@st.cache_data(show_spinner=False)
def compute_frontier_points(
    sector_df: pd.DataFrame, R_sector: pd.DataFrame, te_min=0.002, te_max=0.05, n=40
):
    te_grid = np.linspace(te_min, te_max, n)
    xs_bps, ys_pct = [], []
    for te in te_grid:
        try:
            res = optimize_sector(sector_df, R_sector, te)  # your function above
            xs_bps.append(res["te_real_bps"])  # x in bps
            ys_pct.append(res["reduction_pct"])  # y in %
        except Exception:
            continue
    return np.array(xs_bps), np.array(ys_pct)

# --- FRONTIER FIRST ---
x_bps, y_pct = compute_frontier_points(
    sector_df, R_sector, te_min=0.002, te_max=0.05, n=36
)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown("**TE–Carbon Frontier (with key markers)**")

# markers: selected TE, 5% TE
sel_te_bps = result["te_real_bps"]
idx_sel = int(np.argmin(np.abs(x_bps - sel_te_bps))) if len(x_bps) else None
idx_5 = int(np.argmin(np.abs(x_bps - 500))) if len(x_bps) else None

fig_frontier = go.Figure()
fig_frontier.add_trace(
    go.Scatter(x=x_bps, y=y_pct, mode="lines+markers", name=f"{sector_name} frontier")
)
if idx_sel is not None:
    fig_frontier.add_trace(
        go.Scatter(
            x=[x_bps[idx_sel]],
            y=[y_pct[idx_sel]],
            mode="markers",
            name="Selected TE",
            marker=dict(size=10, symbol="diamond"),
        )
    )
if idx_5 is not None:
    fig_frontier.add_trace(
        go.Scatter(
            x=[x_bps[idx_5]],
            y=[y_pct[idx_5]],
            mode="markers",
            name="5% TE",
            marker=dict(size=10, symbol="x"),
        )
    )
fig_frontier.update_layout(
    xaxis_title="Tracking Error (bps)",
    yaxis_title="Carbon Reduction (%)",
    legend_title="",
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_frontier, use_container_width=True)

st.markdown(f'<div class="sub-title">Room for Maneuver</div>', unsafe_allow_html=True)

# ---- Compute Room-for-Maneuver KPIs from the frontier ----
import math

if len(x_bps) == 0 or len(y_pct) == 0:
    slope_2pct = np.nan
    elasticity_2pct = np.nan
    auc_5pct = np.nan
    max_cut_5pct = np.nan
    min_te_50_pct = np.nan
else:
    # convert x from bps -> percent
    x_pct = x_bps / 100.0

    # Slope and elasticity at 2% TE
    slope_2pct = finite_slope(2.0, x_pct, y_pct)  # dy/dx at TE=2%
    elasticity_2pct = elasticity_at_x(2.0, x_pct, y_pct)  # (%ΔY/%ΔX) at TE=2%

    # AUC up to 5% TE (units: %-reduction * %-TE)
    auc_5pct = auc_to_xmax(x_pct, y_pct, xmax=5.0)

    # Max achievable cut at 5% TE
    max_cut_5pct = interp_at_x(5.0, x_pct, y_pct)  # % reduction at TE=5%

    # Minimum TE to reach 50% cut (in % TE)
    min_te_50_pct = min_te_for_target_cut(x_pct, y_pct, target_pct=50.0)

# Pretty label for the last KPI
if np.isnan(min_te_50_pct):
    min_te_50_label = "Not reached"
else:
    min_te_50_label = f"{min_te_50_pct*100:.0f} bps"  # convert % TE -> bps for display

# Safe fallbacks for display
def _fmt_num(x, decimals=2):
    return (
        "—"
        if (x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))))
        else f"{x:.{decimals}f}"
    )

slope_2pct = slope_2pct if np.isfinite(slope_2pct) else np.nan
elasticity_2pct = elasticity_2pct if np.isfinite(elasticity_2pct) else np.nan
auc_5pct = auc_5pct if np.isfinite(auc_5pct) else np.nan
max_cut_5pct = max_cut_5pct if np.isfinite(max_cut_5pct) else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Slope @ 2% TE", f"{slope_2pct:.2f}")
k2.metric("Elasticity @ 2% TE", f"{elasticity_2pct:.2f}")
k3.metric("AUC ≤ 5% TE", f"{auc_5pct:,.3f}")
k4.metric("Max Cut @ 5% TE", f"{max_cut_5pct:.1f}%")
k5.metric("Min TE for 50% Cut", min_te_50_label)  # "Not reached" or "235 bps"

# ---------------------------
# Weights: benchmark vs optimized
# ---------------------------
bench_df = pd.DataFrame(
    {
        "symbol": result["stock_labels"],
        "weight": result["w_b_vec"],
        "Portfolio": "Benchmark",
    }
)
opt_df = pd.DataFrame(
    {"symbol": result["stock_labels"], "weight": result["w_opt"], "Portfolio": "Optimized"}
)
weights_long = pd.concat([bench_df, opt_df], axis=0)

# Sort by benchmark descending for a tidy display
order = bench_df.sort_values("weight", ascending=False)["symbol"].tolist()

st.markdown("### Original vs optimized weights (per stock)")
fig_weights = px.bar(
    weights_long,
    x="symbol",
    y="weight",
    color="Portfolio",
    barmode="group",
    category_orders={"symbol": order},
)
fig_weights.update_layout(xaxis_title="Stock", yaxis_title="Weight", legend_title="Portfolio")
st.plotly_chart(fig_weights, use_container_width=True)

# ---------------------------
# Carbon reduction gauge-like chart (simple)
# ---------------------------
st.markdown("### Carbon Intensity Reduction")
fig_carbon = go.Figure()
fig_carbon.add_trace(
    go.Indicator(
        mode="number+delta",
        value=result["carbon_opt"],
        number={"valueformat": ",.2f"},
        delta={"reference": result["carbon_b"], "relative": False, "valueformat": ",.2f"},
        title={"text": "Carbon Intensity (Optimized vs Benchmark)"},
    )
)
st.plotly_chart(fig_carbon, use_container_width=True)

# ---------------------------
# Concentration comparison
# ---------------------------
st.markdown("### Market concentration vs optimized portfolio (HHI)")
fig_hhi = go.Figure(
    data=[
        go.Bar(name="Benchmark", x=["HHI"], y=[result["hhi_bench"]]),
        go.Bar(name="Optimized", x=["HHI"], y=[result["hhi_opt"]]),
    ]
)
fig_hhi.update_layout(barmode="group", yaxis_title="HHI (Σ w²)")
st.plotly_chart(fig_hhi, use_container_width=True)

# ---------------------------
# Table of weights
# ---------------------------
table_df = pd.DataFrame(
    {
        "Symbol": result["stock_labels"],
        "w_bench": result["w_b_vec"],
        "w_opt": result["w_opt"],
        "Δw": result["w_opt"] - result["w_b_vec"],
    }
).sort_values("w_bench", ascending=False)

st.markdown("### Weight details per stock")
st.dataframe(table_df, use_container_width=True)

# ---------------------------
# Helpful notes
# ---------------------------
with st.expander("Technical details"):
    st.write(
        r"""
- Covariance estimated with **Ledoit–Wolf** and regularization \(\lambda I\) with \(\lambda=1e{-5}\), then projected to the nearest PSD matrix.
- Optimization: minimize sector **carbon intensity** subject to **TE** \(w - w_b\) and constraints **long-only** and **full-investment**.
- Solver: **ECOS** (cvxpy). Make sure it is installed.
- TE input is **annualized** in %, while the constraint is translated into **monthly** variance as in your script.
"""
    )

# ---------------------------
# Footer / How-to
# ---------------------------
st.markdown(
    """
**How to run**
1. Install packages: `pip install streamlit numpy pandas cvxpy scikit-learn plotly openpyxl`
2. Run: `streamlit run app.py`
3. Place the Excel files in the `Data/` folder as in your project.
"""
)
