import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

def plot_sector_evolution(
    df,
    value_col,
    title,
    ylabel,
    vol_df=None,
    adjust_by_vol=False,
    figsize=(10, 6),
    show=True,            # <- optional
    savepath=None,        # <- optional convenience
    dpi=300               # <- optional
):
    df_plot = df.copy()

    if adjust_by_vol and vol_df is not None:
        df_plot = df_plot.merge(vol_df, on="Sector", how="left")
        adjusted_col = f"{value_col}_per_Vol"
        df_plot[adjusted_col] = df_plot[value_col] / df_plot["Sector Volatility"]
        value_col = adjusted_col

    period_order = sorted(df_plot["Period"].unique())
    df_plot["Period"] = pd.Categorical(df_plot["Period"], categories=period_order, ordered=True)

    sector_colors = {
        'Communication Services': '#E63946',
        'Consumer Discretionary': '#F77F00',
        'Consumer Staples': '#FCBF49',
        'Energy': '#06FFA5',
        'Financials': '#118AB2',
        'Health Care': '#073B4C',
        'Industrials': '#8B5A3C',
        'Information Technology': '#9D4EDD',
        'Materials': '#FF69B4',
        'Real Estate': '#BC4749',
        'Utilities': '#808080'
    }

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

    fig.tight_layout()

    # Save if requested
    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)  # avoids GUI popup / memory buildup

    return fig


def plot_sector_radar_grid(df, cols_to_norm, title, savepath=None):


    labels = [
    "         Sensitivity\n         (Inverted)",   # East
    "Flexibility",               # North
    "Room for         \nManeuver         ",        # West
    "Robustness"                 # South
]

  

    print("\n=== AXIS CHECK ===")
    for label, col in zip(labels, cols_to_norm):
        print(f"{label}  <--->  {col}")
    print("===================\n")

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

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})
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


def plot_all_dimension_evolution(room_df, flex_df, sens_df, robust_df, savepath):
    """
    Plot the evolution of the four DRI dimensions over time
    in a single figure with 4 horizontal subplots.
    """

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


    plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})
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
    plt.subplots_adjust(bottom=0.17)   # give space for the legend

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


    return fig

def extract_period_key(p):
    tag = p.stem.split("_")[-1]   # e.g. "0621"
    month = int(tag[:2])
    year  = int(tag[2:])
    return (year, month)



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

    # Sort pickle files chronologically by period tag
    pickle_files = sorted(
    portfolio_dir.glob("optimal_portfolios_all_te_*.pkl"),
    key=extract_period_key
    )   
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

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.alpha": 0.25,
})

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


def plot_te_carbon_marginal_gains(
    sectors_to_plot=None,
    portfolio_dir="results/optimal_portfolios",
    output_path="results/te_carbon_marginal_gains_last_period_academic.pdf",
    show=True,
):
    """
    Plot TE-Carbon frontier and marginal carbon gain for the latest period.

    Loads the most recent pickle from `portfolio_dir` and produces a 3×2 grid:
    left column = frontier curve, right column = marginal gain curve.

    Parameters
    ----------
    sectors_to_plot : list of str, optional
        Sectors to include. Defaults to
        ["Industrials", "Communication Services", "Financials"].
    portfolio_dir : str or Path
        Directory containing optimal-portfolio pickle files.
    output_path : str or None
        Path to save the figure as PDF. Pass None to skip saving.
    show : bool
        Whether to call plt.show(). Set False when running headlessly.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import pickle
    from pathlib import Path

    if sectors_to_plot is None:
        sectors_to_plot = ["Industrials", "Communication Services", "Financials"]

    # Load latest period
    last_pickle = sorted(Path(portfolio_dir).glob("optimal_portfolios_all_te_*.pkl"))[-1]
    with open(last_pickle, "rb") as f:
        last_period = pickle.load(f)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(len(sectors_to_plot), 2, figsize=(11, 10))

    for row, sector in enumerate(sectors_to_plot):
        data = last_period[sector]
        te = np.array(data["tracking_errors"])
        cr = np.array(data["carbon_reductions"])

        # Frontier
        ax_frontier = axes[row, 0]
        ax_frontier.plot(te, cr, "-", lw=2.0, color="#1f77b4")
        ax_frontier.set_title(f"{sector} — Carbon–TE Frontier", fontsize=12)
        ax_frontier.set_xlabel("Tracking Error (bps)")
        ax_frontier.set_ylabel("Carbon Reduction (%)")
        ax_frontier.set_ylim(0, 102)
        ax_frontier.grid(True)

        # Marginal gains
        marginal = np.gradient(cr, te)
        ax_marg = axes[row, 1]
        ax_marg.plot(te, marginal, "-", lw=2.0, color="#2c7a2c")
        ax_marg.axhline(0, color="black", linewidth=0.8)
        ax_marg.set_title(f"{sector} — Marginal Carbon Gain", fontsize=12)
        ax_marg.set_xlabel("Tracking Error (bps)")
        ax_marg.set_ylabel(r"Marginal Gain ($\Delta CR / \Delta TE$)")
        ax_marg.set_ylim(0, 1.6)
        ax_marg.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig