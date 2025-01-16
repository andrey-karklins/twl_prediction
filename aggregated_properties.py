import csv

import numpy as np

from get_data import *
from utils import load_or_fetch_dataset, M, H, D

# Getting the datasets
# Load datasets
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
# Time intervals (delta_t)
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]


# Function to calculate and write aggregated properties
def get_aggregated_properties(matrix, global_G, dataset_name, delta_t, output_file):
    num_snapshots = matrix.shape[1]  # Number of time intervals
    all_edges = len(global_G.edges_list)  # Total number of edges in the global graph
    edges_per_snapshot = np.count_nonzero(matrix, axis=0)  # Count of active edges per snapshot
    interactions_per_snapshot = matrix.sum(axis=0)  # Total interactions (edge weights) per snapshot

    average_edge_weight = np.mean(interactions_per_snapshot / len(global_G.edges_list))

    # Calculate averages and standard deviations
    avg_edges_per_snapshot = np.mean(edges_per_snapshot / all_edges)  # Percentage of edges per snapshot
    std_edges_per_snapshot = np.std(edges_per_snapshot / all_edges)

    # Calculate maximum entropy for normalization
    max_entropy = np.log(edges_per_snapshot)  # Log of the number of edges

    # For every snapshot, calculate the weighted interaction entropy
    weighted_interaction_entropy = np.zeros(num_snapshots)
    for i in range(num_snapshots):
        if np.isnan(max_entropy[i]) or max_entropy[i] == 0:
            weighted_interaction_entropy[i] = 0
            continue
        p = matrix[:, i] / interactions_per_snapshot[i]
        weighted_interaction_entropy[i] = -np.nansum(p * np.log(p)) / max_entropy[i]

    avg_weighted_interaction_entropy = np.mean(weighted_interaction_entropy)
    # calculate average transitivity per snapshot
    clustering_per_snapshot = np.zeros(num_snapshots)
    transitivity_per_snapshot = np.zeros(num_snapshots)
    for i in range(num_snapshots):
        indices = np.where(matrix[:, i] > 0)[0]
        edges = [global_G.edges_list[j] for j in indices]
        G = nx.Graph()
        G.add_edges_from(edges)
        clustering_per_snapshot[i] = nx.average_clustering(G, weight='weight')
        transitivity_per_snapshot[i] = nx.transitivity(G)

    avg_clustering = np.mean(clustering_per_snapshot)

    # Create result dictionary (without total_number_interactions)
    results = {
        "Dataset name": dataset_name,
        "delta_t": delta_t,
        "total_number_snapshots": num_snapshots,  # Changed to total number of snapshots
        "average_percentage_of_links_per_snapshot": round(avg_edges_per_snapshot * 100, 2),
        "std_percentage_of_links_per_snapshot": round(std_edges_per_snapshot * 100, 2),
        "average_edge_weight": round(average_edge_weight, 2),
        "average_weighted_interaction_entropy": round(avg_weighted_interaction_entropy * 100, 2),
        "average_clustering": round(avg_clustering, 2),
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
                      "average_edge_weight", "average_weighted_interaction_entropy", "average_clustering"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file is being created for the first time
        if not file_exists:
            writer.writeheader()

        # Write the actual data
        writer.writerow(results)


# Function to aggregate data for datasets and delta_t
def aggregate_data(output_file='results/aggregated_properties.csv'):
    tasks = []
    for dataset in datasets_physical:
        for delta_t in delta_ts_physical:
            data, G = aggregate_to_matrix(dataset, delta_t)
            tasks.append((data, G, dataset.name, delta_t))
    for dataset in datasets_virtual:
        for delta_t in delta_ts_virtual:
            data, G = aggregate_to_matrix(dataset, delta_t)
            tasks.append((data, G, dataset.name, delta_t))
    for data, G, dataset_name, delta_t in tasks:
        get_aggregated_properties(data, G, dataset_name, delta_t, output_file)


# aggregate_data()
