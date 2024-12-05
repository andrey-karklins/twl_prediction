import ast
import logging
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds


def seconds_to_human_readable(seconds):
    """
    Convert a given number of seconds into a human-readable format (days, hours, minutes, seconds).

    Args:
    seconds (int): Number of seconds.

    Returns:
    str: Human-readable string representing the time in days, hours, minutes, and seconds.
    """
    # Define time units
    day = 86400  # seconds in a day
    hour = 3600  # seconds in an hour
    minute = 60  # seconds in a minute

    # Calculate the number of days, hours, minutes, and seconds
    days = seconds // day
    seconds %= day
    hours = seconds // hour
    seconds %= hour
    minutes = seconds // minute
    seconds %= minute

    # Create a human-readable string
    human_readable = []
    if days > 0:
        human_readable.append(f"{days}d")
    if hours > 0:
        human_readable.append(f"{hours}h")
    if minutes > 0:
        human_readable.append(f"{minutes}m")
    if seconds > 0 or not human_readable:  # Include seconds even if zero or if the list is empty
        human_readable.append(f"{seconds}s")
    human_readable = " ".join(human_readable)
    return human_readable


def load_or_fetch_dataset(fetch_func, pickle_filename):
    """Loads dataset from a pickle file if it exists; otherwise, fetches, pickles, and returns it."""
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as file:
            dataset = pickle.load(file)
    else:
        dataset = fetch_func()
        with open(pickle_filename, 'wb') as file:
            pickle.dump(dataset, file)
    return dataset

def results_table(csv_name):
    # Update the formatted output directly in the code with "\\" at the end of each row
    # Prepare formatted rows with the final formatting as specified

    # Sort by dataset name in specific order:
    dataset_order = ['Hypertext graph', 'SFHH graph', 'College graph 1', 'College graph 2', 'Socio-calls graph', 'Socio-sms graph']
    remove_columns = ['SDModel tau', 'SDModel L', 'SCDModel tau', 'SCDModel L', 'SCDModel coefs', 'SCDOModel tau', 'SCDOModel L', 'SCDOModel coefs']

    # Load the data
    data = pd.read_csv(csv_name)
    data = data.drop(columns=remove_columns)
    data['Dataset name'] = pd.Categorical(data['Dataset name'], categories=dataset_order, ordered=True)
    data = data.sort_values('Dataset name')

    formatted_rows_final_with_backslashes = []

    # Iterate over each row to apply formatting as specified
    for i, (_, row) in enumerate(data.iterrows()):
        # Copy the row's values
        row_values = list(row)

        # Columns to format for smallest and largest values
        min_columns = [2, 4, 6, 8]  # Columns 3, 5, 7, 9 in 0-based index
        max_columns = [3, 5, 7, 9]  # Columns 4, 6, 8, 10 in 0-based index

        # Find the smallest and largest values in the specified columns
        min_value = min(round(row_values[col], 2) for col in min_columns)
        max_value = max(round(row_values[col], 2) for col in max_columns)

        # Wrap all instances of the minimum values in min_columns in \textbf{} format
        for col in min_columns:
            if round(row_values[col], 2) == min_value:
                row_values[col] = f"\\textbf{{{round(row_values[col], 2)}}}"
            else:
                row_values[col] = f"{round(row_values[col], 2)}"

        # Wrap all instances of the maximum values in max_columns in \textbf{} format
        for col in max_columns:
            if round(row_values[col], 2) == max_value:
                row_values[col] = f"\\textbf{{{round(row_values[col], 2)}}}"
            else:
                row_values[col] = f"{round(row_values[col], 2)}"

        # Apply multirow formatting every 3 rows for the first column
        if i % 3 == 0:
            row_values[0] = f"\\multirow{{3}}{{*}}{{{row_values[0]}}}"
        else:
            row_values[0] = ""  # Leave the first column empty for the other 2 rows in each group

        row_values[1] = f"{seconds_to_human_readable(row_values[1])}"
        # Convert row to " & " separated format and add "\\" at the end
        formatted_rows_final_with_backslashes.append(' & '.join(map(str, row_values)) + " \\\\")


    # write to txt file
    with open("results_table.txt", "w") as file:
        for row in formatted_rows_final_with_backslashes:
            file.write(row + "\n")

def params_table(csv_name):
    # Update the formatted output directly in the code with "\\" at the end of each row
    # Prepare formatted rows with the final formatting as specified

    ## Sort by dataset name in specific order:
    dataset_order = ['Hypertext graph', 'SFHH graph', 'College graph 1', 'College graph 2', 'Socio-calls graph', 'Socio-sms graph']
    remove_columns = ['SDModel MSE', 'SDModel AUPRC', 'SCDModel MSE', 'SCDModel AUPRC', 'SCDOModel MSE', 'SCDOModel AUPRC']

    # Load the data
    data = pd.read_csv(csv_name)
    data = data.drop(columns=remove_columns)
    data['Dataset name'] = pd.Categorical(data['Dataset name'], categories=dataset_order, ordered=True)
    data = data.sort_values('Dataset name')

    formatted_rows_final_with_backslashes = []

    values = np.zeros((data.shape[0], 12))
    # Iterate over each row to apply formatting as specified
    for i, (_, row) in enumerate(data.iterrows()):
        # Copy the row's values
        row_values = list(row)
        # Apply multirow formatting every 3 rows for the first column
        if i % 3 == 0:
            row_values[0] = f"\\multirow{{3}}{{*}}{{{row_values[0]}}}"
        else:
            row_values[0] = ""  # Leave the first column empty for the other 2 rows in each group

        row_values[1] = f"{seconds_to_human_readable(row_values[1])}"
        # unwrap the values in the 5th and 8th columns which are tuples of 3
        coef1, coef2, coef3 = ast.literal_eval(row_values[6])
        row_values[6] = coef1
        row_values.insert(7, coef2)
        row_values.insert(8, coef3)
        coef1, coef2, coef3 = ast.literal_eval(row_values[-1])
        row_values[-1] = coef1
        row_values.append(coef2)
        row_values.append(coef3)
        # Convert row to " & " separated format and add "\\" at the end
        formatted_rows_final_with_backslashes.append(' & '.join(map(str, row_values)) + " \\\\")
        values[i] = row_values[2:]

    mean_values = np.mean(values, axis=0)
    mean_values = np.round(mean_values, 2)
    formatted_rows_final_with_backslashes.append(' & '.join(map(str, mean_values)) + " \\\\")
    formatted_rows_final_with_backslashes[-1] = "\multicolumn{2}{c|}{Mean}" + formatted_rows_final_with_backslashes[-1]
    # write to txt file
    with open("tmp.txt", "w") as file:
        for row in formatted_rows_final_with_backslashes:
            file.write(row + "\n")

params_table("results/best_results.csv")

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


def autocorrelate_all_table(csv_name):
    # Load the CSV file
    df = pd.read_csv(csv_name)

    # Safely evaluate each entry in 'SCDModel coefs' and 'SCDOModel coefs' columns as a tuple/list
    df['SCDModel coefs'] = df['SCDModel coefs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['SCDOModel coefs'] = df['SCDOModel coefs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Breakdown coefficient columns into separate columns
    df[['SCD_beta1', 'SCD_beta2', 'SCD_beta3']] = pd.DataFrame(df['SCDModel coefs'].tolist(), index=df.index)
    df[['SCDO_beta1', 'SCDO_beta2', 'SCDO_beta3']] = pd.DataFrame(df['SCDOModel coefs'].tolist(), index=df.index)

    # Drop the original coefficient columns
    df = df.drop(columns=['SCDModel coefs', 'SCDOModel coefs'])

    # plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(method='pearson'), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.show()


def autocorrelate_table(csv_name):
    # Load the CSV file
    df = pd.read_csv(csv_name)

    # Safely evaluate each entry in 'SCDModel coefs' and 'SCDOModel coefs' columns as a tuple/list
    df['SCDModel coefs'] = df['SCDModel coefs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['SCDOModel coefs'] = df['SCDOModel coefs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Breakdown coefficient columns into separate columns
    df[['SCD_beta1', 'SCD_beta2', 'SCD_beta3']] = pd.DataFrame(df['SCDModel coefs'].tolist(), index=df.index)
    df[['SCDO_beta1', 'SCDO_beta2', 'SCDO_beta3']] = pd.DataFrame(df['SCDOModel coefs'].tolist(), index=df.index)

    # Drop the original coefficient columns
    df = df.drop(columns=['SCDModel coefs', 'SCDOModel coefs'])

    # Select only the columns of interest
    properties = ['delta_t', 'transitivity', 'average_clustering', 'mean_common_neighbors', 'mean_distinct_neighbors']
    scd_betas = ['SCD_beta1', 'SCD_beta2', 'SCD_beta3', "SCDModel MSE", "SCDModel AUPRC"]
    scdo_betas = ['SCDO_beta1', 'SCDO_beta2', 'SCDO_beta3', "SCDOModel MSE", "SCDOModel AUPRC"]
    df_subset = df[properties + scd_betas + scdo_betas]

    # Calculate the correlation matrix for the selected columns
    correlations = df_subset.corr(method='pearson')

    # Filter the correlation matrix for SCD and SCDO separately
    correlations_scd = correlations.loc[properties, scd_betas]
    correlations_scdo = correlations.loc[properties, scdo_betas]

    # Rename columns and rows for LaTeX-style formatting
    correlations_scd.columns = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', 'MSE', 'AUPRC']
    correlations_scdo.columns = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', 'MSE', 'AUPRC']
    correlations_scd.index = [r'$\Delta t$', 'Transitivity', 'Average Clustering', r'$\mu_{common}$',
                              r'$\mu_{distinct}$']
    correlations_scdo.index = [r'$\Delta t$', 'Transitivity', 'Average Clustering', r'$\mu_{common}$',
                               r'$\mu_{distinct}$']

    # Plot correlation heatmap for SCD betas
    plt.figure(figsize=(6, 6))
    sns.heatmap(correlations_scd, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.show()

    # Plot correlation heatmap for SCDO betas
    plt.figure(figsize=(6, 6))
    sns.heatmap(correlations_scdo, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.show()


def plot_normalized_mse(file_path='results/results_L2.csv'):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Group data by 'Dataset name' and 'delta_t'
    grouped = data.groupby(['Dataset name', 'delta_t'])

    # Calculate normalized MSE for SDModel MSE
    data['Worst MSE SD'] = grouped['SDModel MSE'].transform('max')
    data['Best MSE SD'] = grouped['SDModel MSE'].transform('min')
    data['Normalized MSE SD'] = (data['Worst MSE SD'] - data['SDModel MSE']) / (data['Worst MSE SD'] - data['Best MSE SD'])

    # Calculate normalized MSE for SCDModel MSE
    data['Worst MSE SCD'] = grouped['SCDModel MSE'].transform('max')
    data['Best MSE SCD'] = grouped['SCDModel MSE'].transform('min')
    data['Normalized MSE SCD'] = (data['Worst MSE SCD'] - data['SCDModel MSE']) / (data['Worst MSE SCD'] - data['Best MSE SCD'])

    datasets = data['Dataset name'].unique()
    delta_ts = data['delta_t'].unique()
    physical_datasets = ['tatsets', 'hypertext', 'sfhh']

    # Plot for SDModel MSE
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        for delta_t in delta_ts:
            subset = data[(data['Dataset name'] == dataset) & (data['delta_t'] == delta_t)].sort_values(by='tau')
            if not subset.empty:
                plt.plot(subset['tau'], subset['Normalized MSE SD'], label=f"{dataset}, delta_t={delta_t}")

    plt.title("Normalized MSE vs Tau (SDModel MSE)")
    plt.xlabel("Tau")
    plt.ylabel("Normalized MSE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', color='gray', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot for SCDModel MSE (Physical Datasets)
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        for delta_t in delta_ts:
            subset = data[(data['Dataset name'] == dataset) & (data['delta_t'] == delta_t)].sort_values(by='tau')
            if not subset.empty and any(phys in dataset.lower() for phys in physical_datasets):
                plt.plot(subset['tau'], subset['Normalized MSE SCD'], label=f"{dataset}, delta_t={delta_t}")

    plt.title("Normalized MSE vs Tau (Physical Datasets - SCDModel MSE)")
    plt.xlabel("Tau")
    plt.ylabel("Normalized MSE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', color='gray', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot for SCDModel MSE (Virtual Datasets)
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        for delta_t in delta_ts:
            subset = data[(data['Dataset name'] == dataset) & (data['delta_t'] == delta_t)].sort_values(by='tau')
            if not subset.empty and not any(phys in dataset.lower() for phys in physical_datasets):
                plt.plot(subset['tau'], subset['Normalized MSE SCD'], label=f"{dataset}, delta_t={delta_t}")

    plt.title("Normalized MSE vs Tau (Virtual Datasets - SCDModel MSE)")
    plt.xlabel("Tau")
    plt.ylabel("Normalized MSE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', color='gray', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()




# Use the function
# plot_normalized_mse()

def geo_mean(iterable):
    a = np.array(iterable, dtype=np.float64)

    # Filter out zeros and negative values to prevent issues with log
    a = a[a > 0]
    if len(a) == 0:
        return 0  # or np.nan, depending on your use case

    # Use log transformation to avoid overflow, then take the mean
    return np.exp(np.mean(np.log(a)))
