import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_mse_comparison_grid(results_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_csv)

    # Define the datasets and delta_t values for physical and virtual types in the desired order
    datasets_physical = ['SFHH graph', 'Hypertext graph']
    datasets_virtual = ['Socio-sms graph', 'College graph 1', 'College graph 2', 'Socio-calls graph']
    delta_ts_physical = [600, 1800, 3600]  # delta_t values in seconds (10 minutes, 30 minutes, 1 hour)
    delta_ts_virtual = [3600, 86400, 259200]  # delta_t values in seconds (1 hour, 1 day, 3 days)

    # Create a 3x2 grid for the plots (3 aggregation times x 2 dataset types)
    fig, axs = plt.subplots(len(delta_ts_physical), 2, figsize=(14, 18))

    # Map the dataset type to each column
    for col_idx, (dataset_type, datasets, delta_ts) in enumerate([
        ('Physical', datasets_physical, delta_ts_physical),
        ('Virtual', datasets_virtual, delta_ts_virtual)
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

            # Sort datasets by their Baseline MSE values in descending order
            subset = subset.sort_values(by='Baseline MSE', ascending=False).round(2)

            # Define positions and values for the bars
            labels = subset['Dataset name']
            baseline_mse = subset['Baseline MSE']
            sd_mse = subset['SDModel MSE']
            scd_mse = subset['SCDModel MSE']

            # Ensure that all MSE values are finite
            if not (baseline_mse.isnull().any() or sd_mse.isnull().any() or scd_mse.isnull().any()):
                # Define the x positions for each dataset
                x = np.arange(len(labels))
                width = 0.25  # Width of each bar

                # Create the bar plot
                ax = axs[row_idx, col_idx]
                ax.bar(x - width, baseline_mse, width, label='Baseline MSE', color='blue')
                ax.bar(x, sd_mse, width, label='SD MSE', color='orange')
                ax.bar(x + width, scd_mse, width, label='SCD MSE', color='green')

                # Add values on top of each bar for better readability
                for i, v in enumerate(baseline_mse):
                    ax.text(i - width, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)
                for i, v in enumerate(sd_mse):
                    ax.text(i, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)
                for i, v in enumerate(scd_mse):
                    ax.text(i + width, v + 0.01, round(v, 3), ha='center', va='bottom', fontsize=8)

                # Set title and labels for each subplot
                ax.set_title(f'{dataset_type} - delta_t={delta_t} seconds')
                ax.set_xlabel('Dataset')
                ax.set_xticks(ticks=x)
                ax.set_xticklabels(labels, rotation=45, ha='right')

                if col_idx == 0:
                    ax.set_ylabel('MSE')  # Set the y-label only for the first column
            else:
                ax.text(0.5, 0.5, 'Non-finite MSE values',
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{dataset_type} - delta_t={delta_t} seconds')
                ax.axis('off')

    # Add a single legend for the entire figure outside of the subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Call the plotting function with the path to the results CSV
plot_mse_comparison_grid("results/results.csv")
