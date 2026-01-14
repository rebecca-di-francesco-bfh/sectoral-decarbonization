import streamlit as st

from app_layout import set_page_config, apply_global_styles
from app_individual_view_functions import (
    plot_frontier_percent,
    plot_marginal_gains,
    render_dimension_metrics,
    render_individual_radar,
    render_risk_return_profiles,
    label_with_info,
    render_composition_section
)

from app_aggregate_view_functions import (
    plot_aggregate_frontier_absolute,
    plot_aggregate_frontier_percent,
    plot_aggregate_radars_sorted,
    render_aggregate_risk_return_distributions,
    render_room_for_maneuver_bars,
    render_flexibility_bars,
    render_sensitivity_bars,
    render_robustness_bars,
    render_aggregate_composition_tables
)


## ---------------------------------------------------------
# MUST BE FIRST
# ---------------------------------------------------------
set_page_config()

# ---------------------------------------------------------
# Apply global CSS once
# ---------------------------------------------------------
apply_global_styles()

st.markdown(
    """
    <style>
    .view-toggle-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 10px;
    }

    .toggle-text {
        font-size: 14px;
        color: #ddd;
        white-space: nowrap;
    }

    /* Keep toggle size unchanged */
    div[data-testid="stToggle"] {
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# App logic starts here
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### View mode")

    st.markdown(
    """
    <div class="view-toggle-row">
        <span class="toggle-text">Individual/Aggregate:</span>
    """,
    unsafe_allow_html=True
)
    view_toggle = st.toggle("", value=False, key="view_toggle")
    

# Map toggle → view mode
view_mode = "Aggregate" if view_toggle else "Individual"


# ---------------------------------------------------------
sectors = ['Consumer Discretionary', 'Health Care', 'Utilities',
       'Information Technology', 'Real Estate', 'Materials', 'Financials',
       'Industrials', 'Energy', 'Communication Services',
       'Consumer Staples']

available_periods_raw =  ['0321', '0621', '0921', '1221', '0322', '0622', '0922','1222', '0323', '0623',  '0923', '1223']
# Display-friendly labels (e.g., "03/2021")
available_periods_display = ['03/2021', '06/2021', '09/2021', '12/2021', '03/2022', '06/2022', '09/2022','12/2022',  '03/2023',  '06/2023' , '09/2023', '12/2023']
# =====================================================
# INDIVIDUAL VIEW
# =====================================================
if view_mode == "Individual":
    st.markdown("# Individual View for Each Sector")

    container = st.container(border=True)

    with container:
        st.markdown('<div class="settings-box">', unsafe_allow_html=True)

        st.markdown("### Settings")

        sector_name = st.selectbox(
            "Select GICS sector:",
            sectors,
            index=0,
            key="sector_selector"
        )

        analysis_period_tooltip = (
    "Each analysis period represents a quarter-end snapshot of the S&P 500. "
    "Portfolio optimization uses only information available at that date,"
    " including sector benchmark weights and firm-level carbon intensities."
    " Tracking-error risk is measured using a return covariance matrix estimated from the previous two "
    "years of monthly data and enters directly into the tracking-error constraint."
)       

        label_with_info("Select analysis period", analysis_period_tooltip)



        selected_display = st.radio(
            "",
            options=available_periods_display,
            horizontal=True,
            index=len(available_periods_display) - 1,
            label_visibility="collapsed"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Map back to raw tag
    selected_period_raw = available_periods_raw[
        available_periods_display.index(selected_display)
    ]

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        plot_frontier_percent(sector_name, selected_period_raw)

    with col2:
        plot_marginal_gains(sector_name, selected_period_raw)
    
    render_dimension_metrics(sector_name, selected_period_raw)
    render_individual_radar(sector_name, sectors)
    render_risk_return_profiles(sector_name, selected_period_raw)
    render_composition_section(sector_name, selected_period_raw, target_te_bps=200)


# =====================================================
# AGGREGATE VIEW
# ---------------------------------------------------------

elif view_mode == "Aggregate":
    st.markdown("# Aggregate View Across Sectors")
    # -----------------------------------------------------
    # AGGREGATE SETTINGS (TOP OF PAGE)
    # -----------------------------------------------------

    container = st.container(border=True)
    with container:
        st.markdown('<div class="settings-box">', unsafe_allow_html=True)

        st.markdown("### Settings (Only for Te-Carbon Frontiers)")

    
        label_with_info(
    "Select analysis period",
    (
        "Each analysis period represents a quarter-end snapshot of the S&P 500. "
        
    )
)       

        selected_display = st.radio(
            "",
            options=available_periods_display,
            horizontal=True,
            index=len(available_periods_display) - 1,
            label_visibility='collapsed'
        )

        st.markdown('</div>', unsafe_allow_html=True)

        selected_period_raw = available_periods_raw[
            available_periods_display.index(selected_display)
        ]



    st.markdown("## TE–Carbon Frontiers Across Sectors")
    col_left, col_right = st.columns(2)

    with col_left:
        plot_aggregate_frontier_percent(selected_period_raw)

    with col_right:
        plot_aggregate_frontier_absolute(selected_period_raw)

    plot_aggregate_radars_sorted()
    render_aggregate_risk_return_distributions()
    st.markdown("## Dimension Breakdown")
    render_room_for_maneuver_bars()
    render_flexibility_bars()
    render_sensitivity_bars()
    render_robustness_bars()
    render_aggregate_composition_tables()