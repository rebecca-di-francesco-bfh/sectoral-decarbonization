import matplotlib.pyplot as plt
import numpy as np

def plot_te_carbon_by_sector(tracking_errors, carbon_reductions, tracking_errors_np, absolute_te_points, sector_name, sector_weights, relative_te_points):
    # Prepare figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- LEFT PLOT: Absolute change from first point ---
    ax1 = axes[0]
    ax1.plot(tracking_errors, carbon_reductions, 'x-', label='TE–Carbon Frontier')

    # Base point (index 0)
    x_base, y_base = tracking_errors[0], carbon_reductions[0]

    for te_target in absolute_te_points:
        idx_target = np.argmin(np.abs(tracking_errors_np - te_target))
        x_target, y_target = tracking_errors[idx_target], carbon_reductions[idx_target]

        # Arrow from first point to target
        ax1.annotate(
            '', xy=(x_target, y_target), xytext=(x_base, y_base),
            arrowprops=dict(arrowstyle='->', lw=1.5, color= '#cccccc') 
        )

        # Absolute change in % terms
        abs_delta = y_target - y_base
        ax1.text(
            (x_base + x_target) / 2, (y_base + y_target) / 2,
            f'{abs_delta:.1f}%', color='#444444', fontsize=8,
            ha='center', va='center'
        )

    ax1.set_title(f'{sector_name} – Absolute Change from Base (10 bps)')
    ax1.set_xlabel('Tracking Error (bps)')
    ax1.set_ylabel('Carbon Reduction (%)')
    ax1.grid(True)

    # --- RIGHT PLOT: Relative change between intervals (stair-step style) ---
    ax2 = axes[1]
    ax2.plot(tracking_errors, carbon_reductions, 'x-', alpha=0.3)  # faded background frontier

    base_y = min(carbon_reductions) * 1  # start slightly above bottom of data
    step_height = 1

    if sector_name in ["Financials", "Industrials"]:
        step_height = 0.1
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
        if y1 != 0:
            rel_delta = (y2 - y1) / y1 * 100
        else:
            rel_delta = 0

        ax2.annotate(
            '', xy=(x_end, y_arrow), xytext=(x_start, y_arrow),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='green')
        )

        ax2.text(
            (x_start + x_end) / 2, y_arrow + 0.3,
            f'{rel_delta:.1f}%', color='green', fontsize=8,
            ha='center', va='bottom'
        )

        sector_weights[sector_name]['relative_deltas'].append(rel_delta)
    ax2.set_title(f'{sector_name} – Relative Change')
    ax2.set_xlabel('Tracking Error (bps)')
    ax2.set_ylabel('Carbon Reduction (%)')
    ax2.grid(True)

    if sector_name in ["Financials", "Industrials"]:
        # For left plot (real y-axis values)
        ax1.set_ylim(min(carbon_reductions) - 0.5, max(carbon_reductions) + 0.5)
        ax2.set_ylim(min(carbon_reductions) - 0.5, max(carbon_reductions) + 0.5)

    plt.tight_layout()
    plt.show()

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
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd

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

    # Plot 2: Side-by-Side Bar Plot
    if weight_comp_bool:
        plt.figure(figsize=(12, 5))
        plt.bar(x - width/2, w_b_vec, width, label='Benchmark')
        plt.bar(x + width/2, w_opt, width, label='Optimized')
        plt.xticks(x, stock_labels, rotation=45)
        plt.ylabel("Weight")
        plt.title(f"{sector_name or 'Sector'} – Benchmark vs Optimized Weights", weight='bold')
        plt.legend()
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)
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
