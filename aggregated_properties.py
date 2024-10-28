from data_analysis import plot_lag_autocorrelation, plot_lag_weighted_jaccard_similarity, plot_autocorrelation_jaccard
from get_data import *

# Getting the datasets
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [
    get_college_1(),
    get_college_2(),
    get_socio_calls(),
    get_socio_sms()
]
# Time intervals (delta_t)
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]

import csv
import numpy as np

# Function to calculate and write aggregated properties
def get_aggregated_properties(matrix, global_G, dataset_name, delta_t, output_file):
    num_snapshots = matrix.shape[1]  # Number of time intervals
    all_edges = len(global_G.edges_list)  # Total number of edges in the global graph
    edges_per_snapshot = np.count_nonzero(matrix, axis=0)  # Count of active edges per snapshot
    interactions_per_snapshot = matrix.sum(axis=0)  # Total interactions (edge weights) per snapshot

    # Calculate averages and standard deviations
    avg_edges_per_snapshot = np.mean(edges_per_snapshot / all_edges)  # Percentage of edges per snapshot
    std_edges_per_snapshot = np.std(edges_per_snapshot / all_edges)
    avg_interactions_per_snapshot = np.mean(interactions_per_snapshot / interactions_per_snapshot.sum())
    std_interactions_per_snapshot = np.std(interactions_per_snapshot / interactions_per_snapshot.sum())

    # Create result dictionary (without total_number_interactions)
    results = {
        "Dataset name": dataset_name,
        "delta_t": delta_t,
        "total_number_snapshots": num_snapshots,  # Changed to total number of snapshots
        "average_percentage_of_links_per_snapshot": round(avg_edges_per_snapshot * 100, 2),
        "std_percentage_of_links_per_snapshot": round(std_edges_per_snapshot * 100, 2),
        "average_percentage_of_interactions_per_snapshot": round(avg_interactions_per_snapshot * 100, 2),
        "std_percentage_of_interactions_per_snapshot": round(std_interactions_per_snapshot * 100, 2),
    }

    # Write the data to a CSV file
    write_to_csv(output_file, results)

# Function to write data to CSV
def write_to_csv(output_file, results):
    # Check if the file exists
    file_exists = False
    try:
        with open(output_file, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    # Write data to CSV
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ["Dataset name", "delta_t", "total_number_snapshots",
                      "average_percentage_of_links_per_snapshot", "std_percentage_of_links_per_snapshot",
                      "average_percentage_of_interactions_per_snapshot", "std_percentage_of_interactions_per_snapshot"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is being created for the first time
        if not file_exists:
            writer.writeheader()

        # Write the actual data
        writer.writerow(results)

# Function to aggregate data for datasets and delta_t
def aggregate_data(datasets, delta_ts, output_file):
    for dataset in datasets:
        for delta_t in delta_ts:
            # Apply the new aggregate_to_matrix function
            matrix, global_G = aggregate_to_matrix(dataset, delta_t=delta_t)

            # Get the aggregated properties and write them to the CSV file
            get_aggregated_properties(matrix, global_G, dataset.name, delta_t, output_file)

# Example usage:
# Aggregating data for physical datasets and virtual datasets
aggregate_data(datasets_physical, delta_ts_physical, "physical_datasets_output.csv")
aggregate_data(datasets_virtual, delta_ts_virtual, "virtual_datasets_output.csv")
