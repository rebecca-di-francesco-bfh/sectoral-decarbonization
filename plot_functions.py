import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
import math

def plot_sector_evolution(
    df,
    value_col,
    title,
    ylabel,
    vol_df=None,
    adjust_by_vol=False,
    figsize=(10, 6)
):
    """
    Plot the evolution of a metric over time for all sectors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Sector' and 'Period' columns, plus the value_col
    value_col : str
        Column name of the metric to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    vol_df : pd.DataFrame, optional
        DataFrame with 'Sector' and 'Sector Volatility' columns.
        Used if adjust_by_vol=True
    adjust_by_vol : bool, default False
        Whether to divide the metric by sector volatility
    figsize : tuple, default (10, 6)
        Figure size
    """

    df_plot = df.copy()

    # Optionally adjust by volatility
    if adjust_by_vol and vol_df is not None:
        df_plot = df_plot.merge(vol_df, on="Sector", how="left")
        adjusted_col = f"{value_col}_per_Vol"
        df_plot[adjusted_col] = df_plot[value_col] / df_plot["Sector Volatility"]
        value_col = adjusted_col

    # Sort periods chronologically
    period_order = sorted(df_plot["Period"].unique())
    df_plot["Period"] = pd.Categorical(df_plot["Period"], categories=period_order, ordered=True)

    # Define consistent GICS sector color scheme
    sector_colors = {
        'Communication Services': '#E63946',      # Red
        'Consumer Discretionary': '#F77F00',      # Orange
        'Consumer Staples': '#FCBF49',            # Yellow
        'Energy': '#06FFA5',                      # Mint Green
        'Financials': '#118AB2',                  # Blue
        'Health Care': '#073B4C',                 # Dark Blue
        'Industrials': '#8B5A3C',                 # Brown
        'Information Technology': '#9D4EDD',      # Purple
        'Materials': '#6A994E',                   # Olive Green
        'Real Estate': '#BC4749',                 # Burgundy
        'Utilities': '#264653'                    # Navy
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    for sector, grp in df_plot.groupby("Sector"):
        ax.plot(
            grp["Period"],
            grp[value_col],
            marker="o",
            label=sector,
            color=sector_colors.get(sector, 'gray'),
            alpha=0.85,
            linewidth=2
        )

    # Styling
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Period (Quarter)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title="Sector",
        title_fontsize=9,
        frameon=False
    )

    plt.tight_layout()
    plt.show()


def plot_te_carbon_by_sector(
    tracking_errors,
    carbon_reductions,
    tracking_errors_np,
    absolute_te_points,
    sector_name,
    sector_weights,
    relative_te_points,
    save_plot=True,
    show_plot=False,
    fig_dir="figures/te_carbon",
    tag="raw"   # NEW: distinguish raw vs shrinkage
): 
    # Prepare figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- LEFT PLOT: Absolute change from first point ---
    ax1 = axes[0]
    ax1.plot(tracking_errors, carbon_reductions, 'x-', label=f'{tag} frontier')
    x_base, y_base = tracking_errors[0], carbon_reductions[0]

    for te_target in absolute_te_points:
        idx_target = np.argmin(np.abs(tracking_errors_np - te_target))
        x_target, y_target = tracking_errors[idx_target], carbon_reductions[idx_target]

        ax1.annotate('', xy=(x_target, y_target), xytext=(x_base, y_base),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='#cccccc'))
        abs_delta = y_target - y_base
        ax1.text((x_base + x_target) / 2, (y_base + y_target) / 2,
                 f'{abs_delta:.1f}%', color='#444444', fontsize=8,
                 ha='center', va='center')

    ax1.set_title(f'{sector_name} – Absolute Change (base→target) [{tag}]')
    ax1.set_xlabel('Tracking Error (bps)')
    ax1.set_ylabel('Carbon Reduction (%)')
    ax1.grid(True)

    # --- RIGHT PLOT: Relative change between intervals ---
    ax2 = axes[1]
    ax2.plot(tracking_errors, carbon_reductions, 'x-', alpha=0.3, label=f'{tag} frontier')

    base_y = min(carbon_reductions)
    step_height = 1 if sector_name not in ["Financials", "Industrials"] else 0.1

    for i in range(len(relative_te_points) - 1):
        te_start = relative_te_points[i]
        te_end = relative_te_points[i + 1]

        idx_start = np.argmin(np.abs(tracking_errors_np - te_start))
        idx_end = np.argmin(np.abs(tracking_errors_np - te_end))

        x_start = tracking_errors[idx_start]
        x_end = tracking_errors[idx_end]
        y_arrow = base_y + i * step_height

        y1 = carbon_reductions[idx_start]
        y2 = carbon_reductions[idx_end]
        rel_delta = (y2 - y1) / y1 * 100 if y1 != 0 else 0

        ax2.annotate('', xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

        ax2.text((x_start + x_end) / 2, y_arrow + 0.3,
                 f'{rel_delta:.1f}%', color='green', fontsize=8,
                 ha='center', va='bottom')

        sector_weights[sector_name]['relative_deltas'].append(rel_delta)

    ax2.set_title(f'{sector_name} – Relative Change [{tag}]')
    ax2.set_xlabel('Tracking Error (bps)')
    ax2.set_ylabel('Carbon Reduction (%)')
    ax2.grid(True)

    if sector_name in ["Financials", "Industrials"]:
        ax1.set_ylim(min(carbon_reductions) - 0.5, max(carbon_reductions) + 0.5)
        ax2.set_ylim(min(carbon_reductions) - 0.5, max(carbon_reductions) + 0.5)

    plt.tight_layout()

    if save_plot:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(
            fig_dir,
            f"{sector_name.replace(' ', '_')}_te_carbon_full_{tag}.png"  # <- tag added
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"📈 Saved plot: {fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_te_carbon_heatmap(sector_weights, te_caps_annual):
    # Extract sector names in desired order
    sector_labels = list(sector_weights.keys())

    # Get carbon reductions for each sector
    all_carbon_reductions = [sector_weights[sector]['carbon_reductions'] for sector in sector_labels]

    # Heatmap-compatible matrix
    carbon_matrix = np.array(all_carbon_reductions)
    carbon_matrix_reversed = carbon_matrix[::-1]
    sector_labels_reversed = sector_labels[::-1]
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(carbon_matrix_reversed, aspect='auto', cmap='viridis')

    # Set axis ticks
    plt.xticks(ticks=np.linspace(0, len(te_caps_annual) - 1, 11),
            labels=[f"{int(te*10000)}" for te in np.linspace(0, 0.05, 11)])  # bp
    plt.yticks(ticks=np.arange(len(sector_labels_reversed)), labels=sector_labels_reversed)

    # Axis labels and title
    plt.xlabel("Tracking Error (bp)", fontsize=12)
    plt.ylabel("Sector", fontsize=12)
    plt.title("Heatmap: % Carbon Reduction by Sector vs TE", fontsize=15, weight='bold')

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Carbon Reduction (%)", fontsize=12)

    # Optional: grid and layout
    plt.grid(visible=True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_origin_vs_opt_weights_comparison(w_b_vec, w_opt, stock_labels, sector_name, rel_change_bool=False, weight_comp_bool=False, heatmap_bool=False):

    delta_weights = w_opt - w_b_vec
    x = np.arange(len(stock_labels))
    width = 0.35

    # Plot 1: Delta Weight Bar Plot
    if rel_change_bool:
        plt.figure(figsize=(12, 5))
        plt.bar(stock_labels, delta_weights, color=['green' if dw > 0 else 'red' for dw in delta_weights])
        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45)
        plt.ylabel("Change in Weight")
        plt.title(f"{sector_name or 'Sector'} – Change in Portfolio Composition", weight='bold')
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

    if weight_comp_bool:
        # Sort by descending benchmark weights
        sorted_indices = np.argsort(w_b_vec)[::-1]
        w_b_vec_sorted = w_b_vec[sorted_indices]
        w_opt_sorted = w_opt[sorted_indices]
        stock_labels_sorted = [stock_labels[i] for i in sorted_indices]

        x = np.arange(len(stock_labels))
        plt.figure(figsize=(12, 5))
        plt.bar(x - width/2, w_b_vec_sorted, width, label='Benchmark')
        plt.bar(x + width/2, w_opt_sorted, width, label='Optimized')
        plt.xticks(x, stock_labels_sorted, rotation=45)
        plt.ylabel("Weight")
        plt.title(f"{sector_name or 'Sector'} – Benchmark vs Optimized Weights", weight='bold')
        plt.legend()
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)

        # Compute HHI (Concentration Score)
        hhi_bench = np.sum(w_b_vec ** 2)
        hhi_opt = np.sum(w_opt ** 2)

        # Annotate scores on the plot
        plt.text(0.01, 0.95, f'Benchmark Concentration: {hhi_bench:.3f}', transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', color='black',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.text(0.01, 0.85, f'Optimized Concentration: {hhi_opt:.3f}', transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', color='black',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.show()

    # Plot 3: Heatmap of Weights
    if heatmap_bool:
        weight_df = pd.DataFrame([w_b_vec, w_opt], index=['Bench.', 'Optim.'], columns=stock_labels)
        plt.figure(figsize=(12, 2))
        sns.heatmap(weight_df, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=0.5)
        plt.title(f"{sector_name or 'Sector'} – Portfolio Weights Heatmap", weight='bold')
        plt.tight_layout()
        plt.show()



import matplotlib.cm as cm

def plot_sector_radar_grid(df, cols_to_norm, title, savepath=None):

    # --- Radar labels ---
    labels = [
        "         Room for\n         Maneuver",
        "\nFlexibility",
        "Robustness         ",
        "Sensitivity\n(Inverted)"
    ]
    num_vars = len(labels)

    # Radar angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Sort by DRI
    df_sorted = df.sort_values("DRI", ascending=False).reset_index(drop=True)

    # Grid
    n_sectors = len(df_sorted)
    ncols = 3
    nrows = 4


    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(polar=True),
        figsize=(12, 3.3 * nrows)
    )
    axes = axes.flatten()

    # Colormap for DRI-based intensity
    cmap = cm.get_cmap("Blues")   # Stronger = darker

    # --- Plot ---
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax = axes[i]

        # Values
        values = row[cols_to_norm].tolist()
        values += values[:1]

        # Color based on DRI (0 → light, 1 → dark)
        color = cmap(row["DRI"])

        # Plot + fill
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)

        # Axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, fontweight="medium")
        ax.set_ylim(0, 1)

        # Radial ticks
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=5)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5)

        # Title
        ax.set_title(
            f"{row['Sector']} ({row['DRI']:.2f})",
            fontsize=9,
            fontweight="bold",
            y=1.12
        )

    # # Remove unused subplots
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

        # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # ----------------------------------------------------------------------
    # Add colorbar in the empty subplot (Option C)
    # ----------------------------------------------------------------------
    if len(axes) > n_sectors:
        cbar_ax = axes[n_sectors]     # use the first empty cell

        # Create fake scalar mappable for the colorbar
        norm = plt.Normalize(vmin=df_sorted["DRI"].min(), vmax=df_sorted["DRI"].max())
        cmap = plt.cm.Blues
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # Remove polar frame
        cbar_ax.set_axis_off()

        # Add the colorbar
        cbar = plt.colorbar(
            sm,
            ax=cbar_ax,
            fraction=0.8,
            pad=0.1
        )
        cbar.ax.set_title("DRI", fontsize=9, pad=6)
        cbar.ax.tick_params(labelsize=7)


    # Title & spacing
    # plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Reduced padding (vertical & horizontal)
    plt.tight_layout(h_pad=2, w_pad=1)
    #fig.subplots_adjust(hspace=0.25)

    # Save
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()
    return fig


import matplotlib.pyplot as plt
import numpy as np

def plot_all_dimension_evolution(room_df, flex_df, sens_df, robust_df, savepath):
    """
    Plot the evolution of the four DRI dimensions over time
    in a single figure with 4 horizontal subplots.
    """

    # Consistent sector colors
    sector_colors = {
        'Communication Services': '#E63946',
        'Consumer Discretionary': '#F77F00',
        'Consumer Staples': '#FCBF49',
        'Energy': '#06FFA5',
        'Financials': '#118AB2',
        'Health Care': '#073B4C',
        'Industrials': '#8B5A3C',
        'Information Technology': '#9D4EDD',
        'Materials': '#6A994E',
        'Real Estate': '#BC4749',
        'Utilities': '#264653'
    }

    def _prep_periods(df, period_col="Period"):
        order = sorted(df[period_col].unique())
        df = df.copy()
        df[period_col] = pd.Categorical(df[period_col], categories=order, ordered=True)
        return df

    # Ensure periods are ordered
    room_df_p   = _prep_periods(room_df)
    flex_df_p   = _prep_periods(flex_df)
    sens_df_p   = _prep_periods(sens_df)
    robust_df_p = _prep_periods(robust_df)

    # Create 4 subplots horizontally
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(12, 8),
        sharex=True
    )

    def _plot_dimension(ax, df_dim, value_col, title, ylabel):

        df_dim = df_dim.copy()

        # Format periods: 0621 → 06/21
        df_dim["Period"] = df_dim["Period"].astype(str).str.zfill(4)
        df_dim["Period"] = df_dim["Period"].apply(lambda x: f"{x[:2]}/{x[2:]}")

        # Order periods
        period_order = sorted(df_dim["Period"].unique())
        df_dim["Period"] = pd.Categorical(df_dim["Period"], categories=period_order, ordered=True)

        # Plot
        for sector, grp in df_dim.groupby("Sector"):
            ax.plot(
                grp["Period"],
                grp[value_col],
                marker="o",
                linewidth=2,
                label=sector,
                alpha=0.85,
                color=sector_colors.get(sector, "gray"),
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

        # ---- rotate xtick labels slightly ----
        ax.tick_params(axis="x", rotation=30)


  
    # ---- Plot each dimension ----
    _plot_dimension(
        axes[0,0],
        room_df_p,
        value_col="Room_for_Maneuver_Score",
        title="Room for Maneuver",
        ylabel="Score"
    )

    _plot_dimension(
        axes[0, 1],
        flex_df_p,
        value_col="Flexibility_Score",
        title="Flexibility",
        ylabel="Score"
    )

    _plot_dimension(
         axes[1, 0],
        sens_df_p,
        value_col="Sensitivity_Score",
        title="Sensitivity (Inverted)",
        ylabel="Score"
    )

    _plot_dimension(
        axes[1,1],
        robust_df_p,
        value_col="Robustness_Score",
        title="Robustness",
        ylabel="Score"
    )

    # ---- Single centered legend below all subplots ----
    handles, labels = axes[1,1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Sector",
        fontsize=8,
        title_fontsize=9,
        loc="lower center",
        ncol=6,                    # all sectors in one line (adjust if needed)
        bbox_to_anchor=(0.5, -0.003),   # center, slightly below the figure
    )


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)   # give space for the legend

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


    return fig


def plot_te_carbon_frontiers_all_periods(portfolio_dir, output_path=None):
    """
    Plot TE-Carbon frontiers for all periods in a 6x2 subplot grid.

    Parameters
    ----------
    portfolio_dir : str or Path
        Directory containing pickle files with optimal portfolios
    output_path : str, optional
        Path to save the figure as PDF. If None, doesn't save.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    from pathlib import Path
    import pickle

    portfolio_dir = Path(portfolio_dir)
    pickle_files = sorted(portfolio_dir.glob("optimal_portfolios_all_te_*.pkl"))

    # Custom sector order
    ordered_sectors = [
        "Industrials",
        "Financials",
        "Consumer Discretionary",
        "Health Care",
        "Information Technology",
        "Consumer Staples",
        "Energy",
        "Materials",
        "Real Estate",
        "Utilities",
        "Communication Services"
    ]

    # Sector colors - highly distinguishable on white background
    sector_colors = {
        'Communication Services': '#E41A1C',  # Red
        'Consumer Discretionary': '#FF7F00',  # Orange
        'Consumer Staples': '#FFD92F',  # Yellow
        'Energy': '#4DAF4A',  # Green
        'Financials': '#377EB8',  # Blue
        'Health Care': '#984EA3',  # Purple
        'Industrials': '#A65628',  # Brown
        'Information Technology': '#F781BF',  # Pink
        'Materials': '#1B9E77',  # Teal
        'Real Estate': '#D95F02',  # Dark Orange
        'Utilities': '#666666'  # Dark Grey
    }

    # Create 6x2 subplots
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    axes = axes.flatten()

    # Plot each pickle file in a subplot
    for idx, pickle_file in enumerate(pickle_files):
        with open(pickle_file, "rb") as f:
            sector_weights = pickle.load(f)

        # Extract period from filename (e.g., "1223" from "optimal_portfolios_all_te_1223.pkl")
        period = pickle_file.stem.split("_")[-1]
        # Format period with slash (e.g., "0621" -> "06/21")
        formatted_period = f"{period[:2]}/{period[2:]}"

        ax = axes[idx]

        for sector_name in ordered_sectors:
            if sector_name in sector_weights:
                metrics = sector_weights[sector_name]
                ax.plot(metrics['tracking_errors'], metrics['carbon_reductions'],
                       label=sector_name, color=sector_colors[sector_name])

        ax.set_xlabel('Tracking Error (bps)')
        ax.set_ylabel('Carbon Reduction (%)')
        ax.set_title(f'Period {formatted_period}')
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(pickle_files), len(axes)):
        axes[idx].set_visible(False)

    # Create single legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Sectors", loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # Make room for the legend

    # Save as high-quality PDF for LaTeX/Overleaf
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Figure saved to: {output_path}")

    return fig