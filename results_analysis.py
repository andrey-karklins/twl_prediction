import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from get_data import *
from utils import *


def plot_comparison_grid(results_csv):
    datasets_physical = [
        load_or_fetch_dataset(get_hypertext, 'pickles/hypertext.pkl'),
        load_or_fetch_dataset(get_SFHH, 'pickles/SFHH.pkl')
    ]
    datasets_virtual = [
        load_or_fetch_dataset(get_college_1, 'pickles/college_1.pkl'),
        load_or_fetch_dataset(get_college_2, 'pickles/college_2.pkl'),
        load_or_fetch_dataset(get_socio_calls, 'pickles/socio_calls.pkl'),
        load_or_fetch_dataset(get_socio_sms, 'pickles/socio_sms.pkl')
    ]
    delta_ts_physical = [10 * M, 30 * M, 1 * H]
    delta_ts_virtual = [1 * H, 1 * D, 3 * D]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_csv)

    # Define the datasets and delta_t values for physical and virtual types in the desired order
    datasets_physical_names = list(map(lambda G: G.name, datasets_physical))
    datasets_virtual_names = list(map(lambda G: G.name, datasets_virtual))
    metrics = ['MSE', 'AUPRC']

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


def find_best_results(file_path='results/results.csv', output_file_path='results/best_results.csv'):
    # Load the CSV file
    # Replace with your file path
    data = pd.read_csv(file_path)

    # Define the columns for the output
    output_columns = [
        "Dataset name", "delta_t",
        "Baseline MSE", "Baseline AUPRC",
        "SDModel MSE", "SDModel AUPRC", "SDModel tau", "SDModel L",
        "SCDModel MSE", "SCDModel AUPRC", "SCDModel tau", "SCDModel L", "SCDModel coefs",
        "SCDOModel MSE", "SCDOModel AUPRC", "SCDOModel tau", "SCDOModel L", "SCDOModel coefs"
    ]

    # Initialize an empty list for results
    results = []

    # Group by Dataset name and delta_t
    grouped = data.groupby(["Dataset name", "delta_t"])

    # Iterate through each group
    for (dataset_name, delta_t), group in grouped:
        # Find the row with the minimum MSE for each model type
        baseline_row = group.loc[group["Baseline MSE"].idxmin()]
        sdmodel_row = group.loc[group["SDModel MSE"].idxmin()]
        scdmodel_row = group.loc[group["SCDModel MSE"].idxmin()]
        scdomodel_row = group.loc[group["SCDOModel MSE"].idxmin()]

        # Append the best values and corresponding parameters
        results.append({
            "Dataset name": dataset_name,
            "delta_t": delta_t,
            "Baseline MSE": baseline_row["Baseline MSE"],
            "Baseline AUPRC": baseline_row["Baseline AUPRC"],
            "SDModel MSE": sdmodel_row["SDModel MSE"],
            "SDModel AUPRC": sdmodel_row["SDModel AUPRC"],
            "SDModel tau": sdmodel_row["tau"],
            "SDModel L": sdmodel_row["L"],
            "SCDModel MSE": scdmodel_row["SCDModel MSE"],
            "SCDModel AUPRC": scdmodel_row["SCDModel AUPRC"],
            "SCDModel tau": scdmodel_row["tau"],
            "SCDModel L": scdmodel_row["L"],
            "SCDModel coefs": scdmodel_row["SCDModel coefs"],
            "SCDOModel MSE": scdomodel_row["SCDOModel MSE"],
            "SCDOModel AUPRC": scdomodel_row["SCDOModel AUPRC"],
            "SCDOModel tau": scdomodel_row["tau"],
            "SCDOModel L": scdomodel_row["L"],
            "SCDOModel coefs": scdomodel_row["SCDOModel coefs"],
        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=output_columns)

    # Save the results to a CSV file
    results_df.to_csv(output_file_path, index=False)


def improvement_table(csv_name):
    # Load the CSV file
    df = pd.read_csv(csv_name)  # Load without headers as we access by index

    # Calculate percentage improvements between models for each row using specified indices
    # Improvement of SD compared to Baseline
    df['Improvement_Baseline_to_SD'] = ((df["Baseline MSE"] - df["SDModel MSE"]) / df["Baseline MSE"]) * 100

    # Improvement of SCD compared to SD and Baseline
    df['Improvement_SD_to_SCD'] = ((df["SDModel MSE"] - df["SCDModel MSE"]) / df["SDModel MSE"]) * 100
    df['Improvement_Baseline_to_SCD'] = ((df["Baseline MSE"] - df["SCDModel MSE"]) / df["Baseline MSE"]) * 100

    # Improvement of SCDO compared to SCD, SD, and Baseline
    df['Improvement_SCD_to_SCDO'] = ((df["SCDModel MSE"] - df["SCDOModel MSE"]) / df["SCDModel MSE"]) * 100
    df['Improvement_SD_to_SCDO'] = ((df["SDModel MSE"] - df["SCDOModel MSE"]) / df["SDModel MSE"]) * 100
    df['Improvement_Baseline_to_SCDO'] = ((df["Baseline MSE"] - df["SCDOModel MSE"]) / df["Baseline MSE"]) * 100

    # Calculate the average improvement for each step across all rows
    avg_improvements = {
        'Baseline to SD': str(round(df['Improvement_Baseline_to_SD'].mean(), 2)) + "%",
        'Baseline to SCD': str(round(df['Improvement_Baseline_to_SCD'].mean(), 2)) + "%",
        'Baseline to SCDO': str(round(df['Improvement_Baseline_to_SCDO'].mean(), 2)) + "%",
        'SD to SCD': str(round(df['Improvement_SD_to_SCD'].mean(), 2)) + "%",
        'SD to SCDO': str(round(df['Improvement_SD_to_SCDO'].mean(), 2)) + "%",
        'SCD to SCDO': str(round(df['Improvement_SCD_to_SCDO'].mean(), 2)) + "%"
    }

    # Write the LaTeX table to a text file
    with open("tmp.txt", "w") as file:
        for key, value in avg_improvements.items():
            file.write(f"{key} & {value} \\\\\n")


def autocorrelate_all_table(data_csv_names, dataset_properties_csv_name):
    # Load the CSV files and combine them
    df = pd.concat([pd.read_csv(csv_name) for csv_name in data_csv_names], ignore_index=True)
    df = break_down_coefs(df, 'SCDModel')
    df = break_down_coefs(df, 'SCDOModel')

    # Load the dataset properties CSV file
    df_properties = pd.read_csv(dataset_properties_csv_name)

    df = pd.merge(df, df_properties, on=['Dataset name', 'delta_t'])

    # Extract dataset properties column names
    dataset_properties_cols = df_properties.columns.difference(['Dataset name', 'delta_t']).tolist()
    dataset_properties_cols.extend([x + " " + y for x in ["SDModel", "SCDModel", "SCDOModel"] for y in ["L", "tau"]])

    # Remove unnecessary columns
    df = df.drop(columns=['Dataset name', 'delta_t'])

    # Compute the correlation of dataset_properties columns with all other columns
    correlation_df = df.corr(method='pearson').loc[dataset_properties_cols]

    # Plot heatmap
    plt.figure(figsize=(24, len(dataset_properties_cols)))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation of Dataset Properties with All Columns")
    plt.show()


def autocorrelate_table(data_csv_names, dataset_properties_csv_name):
    # Load the CSV files and combine them
    df = pd.concat([pd.read_csv(csv_name) for csv_name in data_csv_names], ignore_index=True)
    df = break_down_coefs(df, 'SCDModel')

    # Load the dataset properties CSV file
    df_properties = pd.read_csv(dataset_properties_csv_name)

    df = pd.merge(df, df_properties, on=['Dataset name', 'delta_t'])

    # ceil the average edge weight
    df["average_edge_weight"] = np.ceil(df["average_edge_weight"])
    df["average_edge_weight"] = df["average_edge_weight"] ** 2

    df["SCDModel MSE"] = df["SCDModel MSE"] / df["average_edge_weight"]

    # Select only the columns of interest
    properties = ['average_clustering', 'average_percentage_of_links_per_snapshot',
                  'average_weighted_interaction_entropy']
    scd_betas = ["SCDModel MSE", "SCDModel AUPRC", 'SCDModel coef 1', 'SCDModel coef 2', 'SCDModel coef 3']
    df_subset = df[properties + scd_betas]

    # Calculate the correlation matrix for the selected columns
    correlations = df_subset.corr(method='spearman')

    # Filter the correlation matrix for SCD and SCDO separately
    correlations_scd = correlations.loc[properties, scd_betas]

    # Rename columns and rows for LaTeX-style formatting
    correlations_scd.columns = ['MSE', 'AUPRC', r'$\beta_1$', r'$\beta_2$', r'$\beta_3$']
    correlations_scd.index = [
        r'$\mu_{cc}$',
        r'$\mu_{\text{E}, \%}$',
        r'$\mu_{\text{H}, \%}$'
    ]

    # Plot correlation heatmap for SCD betas
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations_scd, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.savefig('results/plots/correlation_heatmap.png')
    plt.show()


# plot scatterplot of average link weight to mse
def plot_scatterplot_weight_to_mse(data_csv_name, dataset_properties_csv_name):
    # Load the CSV files
    df = pd.read_csv(data_csv_name)
    df_properties = pd.read_csv(dataset_properties_csv_name)

    # Merge the two dataframes on 'Dataset name' and 'delta_t'
    df = pd.merge(df, df_properties, on=['Dataset name', 'delta_t'])

    # ceil the average edge weight
    df["average_edge_weight"] = np.ceil(df["average_edge_weight"])
    df["average_edge_weight"] = df["average_edge_weight"] ** 2

    # filter out the rows with average edge weight > 200
    df = df[df["average_edge_weight"] <= 200]

    df["SCDModel MSE"] = df["SCDModel MSE"] / df["average_edge_weight"]

    # fit a line to the data
    sns.lmplot(data=df, x='average_edge_weight', y='SCDModel MSE', height=6.5, aspect=1.5)

    # Plot a scatterplot of average edge weight vs. SCDModel MSE
    sns.scatterplot(data=df, x='average_edge_weight', y='SCDModel MSE')

    # Set the zoom limits for x and y axes (adjust as necessary)
    plt.xlim(0, 150)  # Set the limit for the x-axis
    plt.ylim(0, 15)  # Set the limit for the y-axis

    plt.xlabel('Average squared link weight')
    plt.ylabel('Normalized SCDModel MSE')
    plt.savefig('results/plots/weight_vs_mse_normalized.png')
    plt.show()


# autocorrelate_table(['results/best_results_grid.csv'],
#                         'results/aggregated_properties.csv')
plot_scatterplot_weight_to_mse('results/best_results_grid.csv', 'results/aggregated_properties.csv')
