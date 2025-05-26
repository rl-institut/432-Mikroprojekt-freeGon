import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
logger = logging.getLogger(__name__)

def generate_parameter_comparison_charts(matches, dataset_name='DLR', output_dir='charts'):
    """Generate charts comparing original network parameters to allocated source parameters"""
    if len(matches) == 0:
        logger.warning(f"No {dataset_name} matches found, cannot generate parameter comparison charts.")
        return

    logger.info(f"Generating parameter comparison charts for {dataset_name}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a DataFrame for visualization, removing any rows with zero or NaN values
    df_compare = matches.copy()
    df_compare = df_compare.replace(0, np.nan).dropna(subset=['network_r', 'network_x', 'network_b',
                                                              'allocated_r', 'allocated_x', 'allocated_b'])

    # Calculate ratios
    df_compare['r_ratio'] = df_compare['allocated_r'] / df_compare['network_r']
    df_compare['x_ratio'] = df_compare['allocated_x'] / df_compare['network_x']
    df_compare['b_ratio'] = df_compare['allocated_b'] / df_compare['network_b']

    # Generate comparison scatter plots
    for param in ['r', 'x', 'b']:
        col_orig = f'network_{param}'
        col_upd = f'allocated_{param}'
        col_ratio = f'{param}_ratio'

        # Skip if not enough data points
        if df_compare[[col_orig, col_upd]].dropna().shape[0] < 2:
            logger.warning(f"Not enough data points for {param} comparison chart.")
            continue

        # Create scatter plot
        plt.figure(figsize=(10, 8))

        cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                                 [(0, 'blue'),  # small ratio
                                                  (0.5, 'white'),  # equal values
                                                  (1.0, 'red')])  # large ratio

        # Cap ratio values for better visualization
        capped_ratios = df_compare[col_ratio].clip(0, 10)

        plt.scatter(
            df_compare[col_orig],
            df_compare[col_upd],
            c=capped_ratios,
            cmap=cmap,
            alpha=0.8,
            vmin=0,
            vmax=10,
            edgecolors='gray',
            s=50
        )

        # Add colorbar
        cbar = plt.colorbar(label="Allocated / Original ratio (capped at 10)")
        cbar.set_ticks([0, 1, 2, 5, 10])
        cbar.set_ticklabels(['0 (smaller)', '1 (equal)', '2x', '5x', '10x+ (larger)'])

        # Add diagonal line
        min_val = min(df_compare[col_orig].min(), df_compare[col_upd].min())
        max_val = max(df_compare[col_orig].max(), df_compare[col_upd].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Equal values (y = x)")

        # Use log scale if values span multiple orders of magnitude
        span_orig = max_val / min_val if min_val > 0 else 10
        if span_orig > 100:
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"{dataset_name} {param.upper()} Parameter Comparison (log scale)")
        else:
            plt.title(f"{dataset_name} {param.upper()} Parameter Comparison")

        # Add labels and grid
        plt.xlabel(f"Original network {param} value")
        plt.ylabel(f"Allocated {dataset_name} {param} value")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add statistics
        median_ratio = df_compare[col_ratio].median()
        mean_ratio = df_compare[col_ratio].mean()
        plt.figtext(0.15, 0.02, f"Median ratio: {median_ratio:.2f}   |   Mean ratio: {mean_ratio:.2f}", fontsize=10)

        # Save figure
        output_file = os.path.join(output_dir, f"{dataset_name.lower()}_{param}_parameter_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Parameter comparison chart for {dataset_name} {param} saved to {output_file}")

    # Create ratio histograms
    plt.figure(figsize=(15, 5))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{dataset_name} Parameter Ratio Distributions', fontsize=16)

    bins = np.logspace(-2, 2, 41)  # Logarithmic bins from 0.01 to 100

    for i, param in enumerate(['r', 'x', 'b']):
        ratio_col = f'{param}_ratio'
        ratios = df_compare[ratio_col].dropna().clip(0.01, 100)

        if len(ratios) == 0:
            continue

        axs[i].hist(ratios, bins=bins, alpha=0.7, color=f'C{i}')
        axs[i].axvline(x=1, color='red', linestyle='--', label='Equal (ratio=1)')
        axs[i].set_xscale('log')
        axs[i].set_title(f'{param.upper()} Ratio Distribution')
        axs[i].set_xlabel('Allocated / Original ratio')
        axs[i].set_ylabel('Count')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlim(0.01, 100)

        median_ratio = ratios.median()
        axs[i].annotate(f'Median: {median_ratio:.2f}', xy=(0.7, 0.9), xycoords='axes fraction')

    plt.tight_layout()
    ratio_hist_file = os.path.join(output_dir, f"{dataset_name.lower()}_parameter_ratio_histograms.png")
    plt.savefig(ratio_hist_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"{dataset_name} parameter ratio histograms saved to {ratio_hist_file}")

    return os.path.abspath(output_dir)