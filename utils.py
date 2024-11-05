import logging
import os
import pickle

import numpy as np
import pandas as pd
from numba import float64

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

def load_completed_tasks(filename='results/results.csv'):
    try:
        results_df = pd.read_csv(filename)
        return set(zip(results_df['Dataset name'], results_df['delta_t']))
    except FileNotFoundError:
        logging.info("No existing results file found, proceeding with all tasks.")
        return set()

def results_table(csv_name):
    # Update the formatted output directly in the code with "\\" at the end of each row
    # Prepare formatted rows with the final formatting as specified

    # Load the data
    data = pd.read_csv(csv_name)

    formatted_rows_final_with_backslashes = []

    # Iterate over each row to apply formatting as specified
    for i, row in data.iterrows():
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
            else :
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


import pandas as pd


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
    with open("improvement_table.txt", "w") as file:
        for key, value in avg_improvements.items():
            file.write(f"{key} & {value} \\\\\n")

improvement_table('results/results_sorted.csv')

def geo_mean(iterable):
    a = np.array(iterable, dtype=np.float64)

    # Filter out zeros and negative values to prevent issues with log
    a = a[a > 0]
    if len(a) == 0:
        return 0  # or np.nan, depending on your use case

    # Use log transformation to avoid overflow, then take the mean
    return np.exp(np.mean(np.log(a)))