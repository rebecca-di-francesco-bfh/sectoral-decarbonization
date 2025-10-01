import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

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