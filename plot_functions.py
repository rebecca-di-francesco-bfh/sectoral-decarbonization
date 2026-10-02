import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from pathlib import Path
import matplotlib as mpl
import matplotlib.colors as mcolors
import pickle

def plot_sector_radar_grid(df, cols_to_norm, title, savepath=None, formats=("pdf",)):
    """
    Plot one radar chart per sector (sorted by DRI) in a 4x3 grid,
    with a DRI colorbar in the first empty cell.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per sector, with columns "Sector", "DRI" and cols_to_norm.
    cols_to_norm : list of str
        The four normalised dimension columns, in the same order as the
        axis labels (East, North, West, South).
    title : str
        Figure title (currently not drawn; kept for compatibility).
    savepath : str or Path or None
        Base path for the saved figure. Any extension is ignored; one file
        is written per entry in `formats`. Pass None to skip saving.
    formats : tuple of str
        File formats to save, e.g. ("pdf",), ("eps",) or ("pdf", "eps").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    labels = [
        "         Sensitivity\n         (Inverted)",   # East
        "Flexibility",                                 # North
        "Room for         \nManeuver         ",        # West
        "Robustness"                                   # South
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

    # Styling without transparency, so PDF and EPS look identical
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#C8C8C8",
        "grid.linestyle": "--",
        "ps.fonttype": 42,         # embed fonts as real text in EPS
        "pdf.fonttype": 42,        # same for PDF
    })

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(polar=True),
        figsize=(12, 3.3 * nrows)
    )
    axes = axes.flatten()

    # Colormap and normalisation for DRI-based intensity.
    # The SAME norm is used for the radar colours and the colorbar,
    # so the colorbar always matches the plotted colours.
    cmap = mpl.colormaps["Blues"]           # stronger = darker
    norm = mcolors.Normalize(vmin=0, vmax=1)

    def _lighten(color, opacity=0.25):
        """Blend a colour with white to mimic alpha=opacity without transparency."""
        r, g, b = mcolors.to_rgb(color)
        return (1 - opacity + opacity * r,
                1 - opacity + opacity * g,
                1 - opacity + opacity * b)

    # --- Plot ---
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax = axes[i]

        # Values
        values = row[cols_to_norm].tolist()
        values += values[:1]

        # Color based on DRI
        color = cmap(norm(row["DRI"]))

        # Fill first (light, opaque), then the outline on top
        ax.fill(angles, values, color=_lighten(color), zorder=1)
        ax.plot(angles, values, color=color, linewidth=2, zorder=2)

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

    # ----------------------------------------------------------------------
    # Colorbar in the first empty cell; remove any other unused cells
    # ----------------------------------------------------------------------
    if len(axes) > n_sectors:
        cbar_ax = axes[n_sectors]
        cbar_ax.set_axis_off()             # hide the polar frame

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=cbar_ax,
            fraction=0.8,
            pad=0.1
        )
        cbar.ax.set_title("DRI", fontsize=9, pad=6)
        cbar.ax.tick_params(labelsize=7)

    for j in range(n_sectors + 1, len(axes)):
        fig.delaxes(axes[j])

    # Reduced padding (vertical & horizontal)
    plt.tight_layout(h_pad=2, w_pad=1)

    # Save one file per requested format
    if savepath is not None:
        base = Path(savepath).with_suffix("")
        for ext in formats:
            out = base.with_suffix(f".{ext}")
            # dpi only matters for raster formats such as png
            fig.savefig(out, bbox_inches="tight", format=ext, dpi=600)
            print(f"Figure saved to: {out}")

    plt.show()
    return fig

def plot_sector_evolution(
    df,
    value_col,
    title,
    ylabel,
    vol_df=None,
    adjust_by_vol=False,
    figsize=(10, 6),
    show=True,
    savepath=None,
    formats=("pdf",),
    dpi=600,
):
    """
    Plot the evolution of a score over time, one line per sector.
 
    Parameters
    ----------
    df : pandas.DataFrame
        Columns "Period", "Sector" and `value_col`.
    value_col : str
        Column to plot.
    title, ylabel : str
        Plot title and y-axis label.
    vol_df : pandas.DataFrame, optional
        Columns "Sector" and "Sector Volatility"; used if adjust_by_vol=True.
    adjust_by_vol : bool
        If True, plot value_col divided by sector volatility.
    figsize : tuple
        Figure size in inches.
    show : bool
        Whether to call plt.show(). Set False when running headlessly.
    savepath : str or Path or None
        Base path for the saved figure. Any extension is ignored; one file
        is written per entry in `formats`. Pass None to skip saving.
    formats : tuple of str
        File formats to save, e.g. ("pdf",), ("eps",) or ("pdf", "eps").
    dpi : int
        Resolution for raster formats such as png (ignored for pdf/eps/svg).
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    df_plot = df.copy()
 
    if adjust_by_vol and vol_df is not None:
        df_plot = df_plot.merge(vol_df, on="Sector", how="left")
        adjusted_col = f"{value_col}_per_Vol"
        df_plot[adjusted_col] = df_plot[value_col] / df_plot["Sector Volatility"]
        value_col = adjusted_col
 
    # Sort rows chronologically. Matplotlib orders a text x-axis by the order
    # in which values first appear, so the rows themselves must be in order.
    def _chrono_key(period):
        p = str(period)
        if p.isdigit() and len(p) <= 4:        # MMYY, e.g. "0621" or 621
            p = p.zfill(4)
            return (0, int(p[2:]), int(p[:2]))  # year first, then month
        return (1, p, 0)                        # any other format: plain sort
 
    df_plot["_key"] = df_plot["Period"].apply(_chrono_key)
    df_plot = df_plot.sort_values("_key")
    df_plot["Period"] = df_plot["Period"].astype(str)
 
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
 
    # Styling without transparency, so PDF and EPS look identical
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#C8C8C8",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "ps.fonttype": 42,         # embed fonts as real text in EPS
        "pdf.fonttype": 42,        # same for PDF
    })
 
    fig, ax = plt.subplots(figsize=figsize)
 
    for sector, grp in df_plot.groupby("Sector"):
        ax.plot(
            grp["Period"],
            grp[value_col],
            marker="o",
            label=sector,
            color=sector_colors.get(sector, 'gray'),
            linewidth=2
        )
 
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Period (Quarter)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        title="Sector",
        title_fontsize=9,
        frameon=False
    )
 
    fig.tight_layout()
 
    # Save one file per requested format
    if savepath is not None:
        base = Path(savepath).with_suffix("")
        for ext in formats:
            out = base.with_suffix(f".{ext}")
            fig.savefig(out, bbox_inches="tight", format=ext, dpi=dpi)
            print(f"Figure saved to: {out}")
 
    if show:
        plt.show()
    else:
        plt.close(fig)  # avoids GUI popup / memory buildup
 
    return fig

def plot_all_dimension_evolution(room_df, flex_df, sens_df, robust_df, savepath,
                                 formats=("pdf",)):
    """
    Plot the evolution of the four DRI dimensions over time
    in a single figure with a 2x2 grid of subplots.

    Parameters
    ----------
    room_df, flex_df, sens_df, robust_df : pandas.DataFrame
        One DataFrame per dimension, each with columns "Period", "Sector"
        and the corresponding score column.
    savepath : str or Path or None
        Base path for the saved figure. Any extension is ignored; one file
        is written per entry in `formats`. Pass None to skip saving.
    formats : tuple of str
        File formats to save, e.g. ("pdf",), ("eps",) or ("pdf", "eps").

    Returns
    -------
    fig : matplotlib.figure.Figure
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

    # Styling without transparency, so PDF and EPS look identical
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#C8C8C8",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "legend.framealpha": 1.0,  # opaque legend box (EPS has no transparency)
        "ps.fonttype": 42,         # embed fonts as real text in EPS
        "pdf.fonttype": 42,        # same for PDF
    })

    def _chrono_key(period):
        """'0621' -> (21, 6), so periods sort by year first, then month."""
        return (int(period[2:]), int(period[:2]))

    def _plot_dimension(ax, df_dim, value_col, title, ylabel):

        df_dim = df_dim.copy()

        # Normalise periods to 4-digit MMYY strings (e.g. 621 -> "0621")
        df_dim["Period"] = df_dim["Period"].astype(str).str.zfill(4)

        # Sort rows chronologically (year, then month) before plotting,
        # so the x-axis runs 03/21, 06/21, ..., 12/23
        df_dim["_key"] = df_dim["Period"].apply(_chrono_key)
        df_dim = df_dim.sort_values("_key")

        # Format periods: 0621 -> 06/21
        df_dim["Period"] = df_dim["Period"].apply(lambda x: f"{x[:2]}/{x[2:]}")

        # Plot
        for sector, grp in df_dim.groupby("Sector", sort=False):
            ax.plot(
                grp["Period"],
                grp[value_col],
                marker="o",
                linewidth=2,
                label=sector,
                color=sector_colors.get(sector, "gray"),
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Period")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # Rotate x tick labels slightly
        ax.tick_params(axis="x", rotation=30)

    # Create 2x2 subplots
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(12, 8),
        sharex=True
    )

    # ---- Plot each dimension ----
    _plot_dimension(
        axes[0, 0],
        room_df,
        value_col="Room_for_Maneuver_Score",
        title="Room for Maneuver",
        ylabel="Score"
    )

    _plot_dimension(
        axes[0, 1],
        flex_df,
        value_col="Flexibility_Score",
        title="Flexibility",
        ylabel="Score"
    )

    _plot_dimension(
        axes[1, 0],
        sens_df,
        value_col="Sensitivity_Score",
        title="Sensitivity (Inverted)",
        ylabel="Score"
    )

    _plot_dimension(
        axes[1, 1],
        robust_df,
        value_col="Robustness_Score",
        title="Robustness",
        ylabel="Score"
    )

    # ---- Single centered legend below all subplots ----
    handles, labels = axes[1, 1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Sector",
        fontsize=8,
        title_fontsize=9,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.003),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)   # give space for the legend

    # Save one file per requested format
    if savepath is not None:
        base = Path(savepath).with_suffix("")
        for ext in formats:
            out = base.with_suffix(f".{ext}")
            # dpi only matters for raster formats such as png
            fig.savefig(out, bbox_inches="tight", format=ext, dpi=600)
            print(f"Figure saved to: {out}")

    plt.show()

    return fig


def extract_period_key(p):
    tag = p.stem.split("_")[-1]   # e.g. "0621"
    month = int(tag[:2])
    year  = int(tag[2:])
    return (year, month)


def plot_te_carbon_frontiers_all_periods(portfolio_dir, output_path=None, formats=("pdf",)):
    """
    Plot TE-Carbon frontiers for all periods in a 6x2 subplot grid.
 
    Parameters
    ----------
    portfolio_dir : str or Path
        Directory containing pickle files with optimal portfolios
    output_path : str or Path, optional
        Base path for the saved figure. Any extension is ignored; one file
        is written per entry in `formats`. If None, doesn't save.
    formats : tuple of str
        File formats to save, e.g. ("pdf",), ("eps",), ("svg",)
        or ("pdf", "eps", "svg").
 
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
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
 
    # Styling without transparency, so PDF, EPS and SVG look identical
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#C8C8C8",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "legend.framealpha": 1.0,  # opaque legend box (EPS has no transparency)
        "ps.fonttype": 42,         # embed fonts as real text in EPS
        "pdf.fonttype": 42,        # same for PDF
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
    fig.legend(handles, labels, title="Sectors", loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02))
 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # Make room for the legend
 
    # Save one file per requested format
    if output_path is not None:
        base = Path(output_path).with_suffix("")
        for ext in formats:
            out = base.with_suffix(f".{ext}")
            fig.savefig(out, bbox_inches="tight", format=ext)
            print(f"Figure saved to: {out}")
 
    return fig
def plot_te_carbon_marginal_gains(
    sectors_to_plot=None,
    portfolio_dir="results/optimal_portfolios",
    output_path="results/te_carbon_marginal_gains_last_period_academic",
    formats=("pdf",),
    show=True,
):
    """
    Plot TE-Carbon frontier and marginal carbon gain for the latest period.

    Loads the most recent pickle from `portfolio_dir` and produces a grid
    with one row per sector: left column = frontier curve,
    right column = marginal gain curve.

    Parameters
    ----------
    sectors_to_plot : list of str, optional
        Sectors to include. Defaults to
        ["Industrials", "Communication Services", "Financials"].
    portfolio_dir : str or Path
        Directory containing optimal-portfolio pickle files.
    output_path : str or Path or None
        Base path for the saved figure. Any extension is ignored; one file
        is written per entry in `formats`. Pass None to skip saving.
    formats : tuple of str
        File formats to save, e.g. ("pdf",), ("eps",) or ("pdf", "eps").
    show : bool
        Whether to call plt.show(). Set False when running headlessly.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sectors_to_plot is None:
        sectors_to_plot = ["Industrials", "Communication Services", "Financials"]

    # Load latest period (sorted chronologically)
    last_pickle = sorted(
        Path(portfolio_dir).glob("optimal_portfolios_all_te_*.pkl"),
        key=extract_period_key
    )[-1]
    with open(last_pickle, "rb") as f:
        last_period = pickle.load(f)

    # Styling without transparency, so PDF and EPS look identical
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#C8C8C8",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "ps.fonttype": 42,         # embed fonts as real text in EPS
        "pdf.fonttype": 42,        # same for PDF
    })

    # squeeze=False keeps axes 2-D, so a single sector also works
    fig, axes = plt.subplots(len(sectors_to_plot), 2, figsize=(11, 10), squeeze=False)

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

    # Save one file per requested format
    if output_path is not None:
        base = Path(output_path).with_suffix("")
        for ext in formats:
            out = base.with_suffix(f".{ext}")
            fig.savefig(out, bbox_inches="tight", format=ext)
            print(f"Figure saved to: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig