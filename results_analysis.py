import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from main import delta_ts_physical, delta_ts_virtual, datasets_physical, datasets_virtual
from utils import seconds_to_human_readable


def plot_comparison_grid(results_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_csv)

    # Define the datasets and delta_t values for physical and virtual types in the desired order
    datasets_physical_names = list(map(lambda G: G.name, datasets_physical))
    datasets_virtual_names = list(map(lambda G: G.name, datasets_virtual))
    metrics = ['MSE', 'MAE', 'RMSE', 'AUPRC']

    # Store the sorted order of datasets for each dataset type after sorting by the smallest delta_t
    sorted_order_physical = {}
    sorted_order_virtual = {}

    # Create a 3x2 grid for each metric (3 aggregation times x 2 dataset types)
    for metric in metrics:
        fig, axs = plt.subplots(len(delta_ts_physical), 2, figsize=(14, 18))

        # Map the dataset type to each column
        for col_idx, (dataset_type, datasets, delta_ts, sorted_order) in enumerate([
            ('Physical', datasets_physical_names, delta_ts_physical, sorted_order_physical),
            ('Virtual', datasets_virtual_names, delta_ts_virtual, sorted_order_virtual)
        ]):
            for row_idx, delta_t in enumerate(delta_ts):
                # Filter data for the current dataset type and delta_t
                subset = df[(df['Dataset name'].isin(datasets)) & (df['delta_t'] == delta_t)]

                # Ensure there is data to plot
                if subset.empty:
                    axs[row_idx, col_idx].text(0.5, 0.5, 'No data available',
                                               ha='center', va='center', fontsize=12)
                    axs[row_idx, col_idx].set_title(f'{dataset_type} - delta_t={delta_t} seconds')
                    axs[row_idx, col_idx].axis('off')
                    continue

                # Determine the baseline, SD, and SCD metric columns
                baseline_col = f'Baseline {metric}'
                sd_col = f'SDModel {metric}'
                scd_col = f'SCDModel {metric}'

                if baseline_col not in subset.columns or sd_col not in subset.columns or scd_col not in subset.columns:
                    axs[row_idx, col_idx].text(0.5, 0.5, f'{metric} data not available',
                                               ha='center', va='center', fontsize=12)
                    axs[row_idx, col_idx].axis('off')
                    continue

                if row_idx == 0:
                    subset = subset.sort_values(by=baseline_col, ascending=False).round(2)
                    sorted_order.update({dataset_type: subset['Dataset name'].tolist()})
                else:
                    # Use the stored sorted order for other delta_t values
                    sorted_labels = sorted_order.get(dataset_type, [])
                    subset.loc[:, 'sort_key'] = subset['Dataset name'].apply(
                        lambda x: sorted_labels.index(x) if x in sorted_labels else float('inf'))
                    subset = subset.sort_values('sort_key').round(2)

                # Define positions and values for the bars
                labels = subset['Dataset name']
                baseline_values = subset[baseline_col]
                sd_values = subset[sd_col]
                scd_values = subset[scd_col]

                # Ensure that all values are finite
                if not (baseline_values.isnull().any() or sd_values.isnull().any() or scd_values.isnull().any()):
                    # Define the x positions for each dataset
                    x = np.arange(len(labels))
                    width = 0.25  # Width of each bar

                    # Create the bar plot
                    ax = axs[row_idx, col_idx]
                    ax.bar(x - width, baseline_values, width, label=f'Baseline {metric}', color='blue')
                    ax.bar(x, sd_values, width, label=f'SD {metric}', color='orange')
                    ax.bar(x + width, scd_values, width, label=f'SCD {metric}', color='green')

                    # Add values on top of each bar for better readability
                    for i, v in enumerate(baseline_values):
                        ax.text(i - width, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)
                    for i, v in enumerate(sd_values):
                        ax.text(i, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)
                    for i, v in enumerate(scd_values):
                        ax.text(i + width, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)

                    # Set title and labels for each subplot
                    ax.set_title(f'{dataset_type} - Δt = {seconds_to_human_readable(delta_t)}')
                    ax.set_xlabel('Dataset')
                    ax.set_xticks(ticks=x)
                    ax.set_xticklabels(labels, rotation=45, ha='right')

                    if col_idx == 0:
                        ax.set_ylabel(metric)  # Set the y-label only for the first column
                else:
                    ax.text(0.5, 0.5, 'Non-finite values',
                            ha='center', va='center', fontsize=12)
                    ax.set_title(f'{dataset_type} - Δt = {seconds_to_human_readable(delta_t)}')
                    ax.axis('off')

        # Add a single legend for the entire figure outside of the subplots
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)

        # Adjust layout and show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(f'Comparison of {metric} Across Different Datasets and Delta_t Values', fontsize=16)
        plt.show()

# Call the plotting function with the path to the results CSV
plot_comparison_grid("results/results.csv")
