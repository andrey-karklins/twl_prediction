import ast
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
    dataset_order = ['Hypertext graph', 'SFHH graph', 'College graph 1', 'College graph 2', 'Socio-calls graph',
                     'Socio-sms graph']
    remove_columns = ['SDModel tau', 'SDModel L', 'SCDModel tau', 'SCDModel L', 'SCDModel coefs', 'SCDOModel tau',
                      'SCDOModel L', 'SCDOModel coefs']

    # Load the data
    data = pd.read_csv(csv_name)
    data = data.drop(columns=remove_columns)

    data['Dataset name'] = pd.Categorical(data['Dataset name'], categories=dataset_order, ordered=True)
    data['delta_t'] = data['delta_t'].astype(int)
    data = data.sort_values(['Dataset name', 'delta_t'])

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
    dataset_order = ['Hypertext graph', 'SFHH graph', 'College graph 1', 'College graph 2', 'Socio-calls graph',
                     'Socio-sms graph']
    remove_columns = ['Baseline MSE', 'Baseline AUPRC', 'SDModel MSE', 'SDModel AUPRC', 'SCDModel MSE',
                      'SCDModel AUPRC', 'SCDOModel MSE', 'SCDOModel AUPRC']

    # Load the data
    data = pd.read_csv(csv_name)
    data = data.drop(columns=remove_columns)
    data['Dataset name'] = pd.Categorical(data['Dataset name'], categories=dataset_order, ordered=True)
    data['delta_t'] = data['delta_t'].astype(int)
    data = data.sort_values(['Dataset name', 'delta_t'])

    formatted_rows_final_with_backslashes = []

    values = np.zeros((data.shape[0], 12))
    scd_params_index = data.columns.get_loc("SCDModel coefs")
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
        # Parsing the string into a rounded NumPy array
        def parse_and_round_array(string, decimals=2):
            # Remove square brackets and split the elements
            elements = string.strip("[]").split()
            # Convert to NumPy array of floats
            array = np.array(elements, dtype=float)
            # Round the array to the specified number of decimals
            return np.round(array, decimals=decimals)

        coefs = parse_and_round_array(row_values[scd_params_index])
        row_values[scd_params_index] = coefs[1]
        row_values.insert(scd_params_index + 1, coefs[2])
        row_values.insert(scd_params_index + 2, coefs[3])
        coefs = parse_and_round_array(row_values[-1])
        row_values[-1] = coefs[1]
        row_values.append(coefs[2])
        row_values.append(coefs[3])
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


def plot_normalized_mse(file_path='results/results_L10.csv'):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Group data by 'Dataset name' and 'delta_t'
    grouped = data.groupby(['Dataset name', 'delta_t'])

    # Calculate normalized MSE for SDModel MSE
    data['Worst MSE SD'] = grouped['SDModel MSE'].transform('max')
    data['Best MSE SD'] = grouped['SDModel MSE'].transform('min')
    data['Normalized MSE SD'] = (data['Worst MSE SD'] - data['SDModel MSE']) / (
            data['Worst MSE SD'] - data['Best MSE SD'])

    # Calculate normalized MSE for SCDModel MSE
    data['Worst MSE SCD'] = grouped['SCDModel MSE'].transform('max')
    data['Best MSE SCD'] = grouped['SCDModel MSE'].transform('min')
    data['Normalized MSE SCD'] = (data['Worst MSE SCD'] - data['SCDModel MSE']) / (
            data['Worst MSE SCD'] - data['Best MSE SCD'])

    # find tau with best MSE scd model
    best_tau = data.loc[data.groupby(['Dataset name', 'delta_t'])['SCDModel MSE'].idxmin()]

    # group dataset name and delta_t by best tau being <5 and >=5
    best_tau['tau < 5'] = best_tau['tau'] < 3

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

    # Function to plot based on tau condition
    def plot_scd_normalized(tau_condition, condition_name):
        plt.figure(figsize=(10, 6))
        for dataset in datasets:
            for delta_t in delta_ts:
                # Get rows where 'Dataset name' and 'delta_t' match and the condition on tau holds
                tau_values = best_tau[
                    (best_tau['Dataset name'] == dataset) & (best_tau['delta_t'] == delta_t) & tau_condition]
                if not tau_values.empty:
                    # Use the entire data to plot values across all tau for matching dataset and delta_t
                    subset = data[(data['Dataset name'] == dataset) & (data['delta_t'] == delta_t)].sort_values(
                        by='tau')
                    plt.plot(subset['tau'], subset['Normalized MSE SCD'], label=f"{dataset}, delta_t={delta_t}")
        plt.title(f"Normalized MSE vs Tau (SCDModel MSE) - {condition_name}")
        plt.xlabel("Tau")
        plt.ylabel("Normalized MSE")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', linestyle='--', color='gray', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Plot for tau < 5
    plot_scd_normalized(best_tau['tau < 5'], "Best tau < 3")
    # Plot for tau >= 5
    plot_scd_normalized(~best_tau['tau < 5'], "Best tau >= 3")


def break_down_coefs(df, model_name):
    coefs = df[f'{model_name} coefs']

    def parse_and_round_array(string, decimals=2):
        # Remove square brackets and split the elements
        elements = string.strip("[]").split()
        # Convert to NumPy array of floats
        array = np.array(elements, dtype=float)
        # Round the array to the specified number of decimals
        return np.round(array, decimals=decimals)

    coefs = coefs.apply(lambda x: parse_and_round_array(x))
    df[f'{model_name} coef 1'] = coefs.apply(lambda x: x[0])
    df[f'{model_name} coef 2'] = coefs.apply(lambda x: x[1])
    df[f'{model_name} coef 3'] = coefs.apply(lambda x: x[2])
    df = df.drop(columns=[f'{model_name} coefs'])
    return df


def combine_dataset_name_and_delta_t(df):
    df['Dataset name'] = df['Dataset name'] + '_' + df['delta_t'].astype(str)
    df = df.drop(columns=['delta_t'])
    return df


def geo_mean(iterable):
    a = np.array(iterable, dtype=np.float64)

    # Filter out zeros and negative values to prevent issues with log
    a = a[a > 0]
    if len(a) == 0:
        return 0  # or np.nan, depending on your use case

    # Use log transformation to avoid overflow, then take the mean
    return np.exp(np.mean(np.log(a)))