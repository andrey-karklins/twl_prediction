import numpy as np
import pandas as pd

from get_data import *

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds

delta_ts_physical = [10 * M, 20 * M, 40 * M, 1 * H]
delta_ts_virtual = [12 * H, 1 * D, 3 * D, 7 * D]
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [get_college_1(), get_college_2(), get_socio_calls(), get_socio_sms()]


def get_aggregated_properties(snapshots, links):
    num_snapshots = len(snapshots)
    all_edges = len(links)
    edges_per_snapshot = [len(g.edges()) for g in snapshots]
    interactions_per_snapshot = [sum(data['weight'] for _, _, data in g.edges(data=True)) for g in snapshots]
    weights_per_snapshot = [sum(data['weight'] for _, _, data in g.edges(data=True)) / edges_per_snapshot[i] for i, g in
                            enumerate(snapshots)]

    # To store minimum and maximum weights per snapshot
    min_weights = []
    max_weights = []

    for g in snapshots:
        if g.edges(data=True):  # Check if there are edges in the snapshot
            snapshot_weights = [data['weight'] for _, _, data in g.edges(data=True)]
            min_weights.append(min(snapshot_weights))
            max_weights.append(max(snapshot_weights))

    # Calculate averages and standard deviations
    all_interactions = sum(interactions_per_snapshot)
    avg_edges_per_snapshot = np.mean([edges / all_edges for edges in edges_per_snapshot])
    std_edges_per_snapshot = np.std([edges / all_edges for edges in edges_per_snapshot])
    avg_interactions_per_snapshot = np.mean([interactions / all_interactions for interactions in interactions_per_snapshot])
    std_interactions_per_snapshot = np.std([interactions / all_interactions for interactions in interactions_per_snapshot])
    min_interactions = min(interactions_per_snapshot)/all_interactions
    max_interactions = max(interactions_per_snapshot)/all_interactions

    return {
        "n": num_snapshots,
        "total_links": all_edges,
        "total_interaction": all_interactions,
        "mu_links": round(avg_edges_per_snapshot * 100, 2),
        "std_links": round(std_edges_per_snapshot * 100, 2),
        "mu_contacts": round(avg_interactions_per_snapshot * 100, 2),
        "std_contacts": round(std_interactions_per_snapshot * 100, 2),
        "min_contacts": round(min_interactions * 100, 8),
        "max_contacts": round(max_interactions * 100, 2),
        # "mu_min_w": round(avg_min_weight, 2),
        # "mu_w": round(avg_weight_per_snapshot, 2),
        # "mu_max_w": round(avg_max_weight, 2),
        # "std_w": round(std_weight_per_snapshot, 2),
        # "var_coef_w": round(std_weight_per_snapshot/avg_weight_per_snapshot, 2)
    }


def grid_search(datasets, delta_ts):
    results = []
    for dataset in datasets:
        links = set([(u, v) for (u, v) in dataset.edges()])
        for delta_t in delta_ts:
            data = aggregate_into_snapshots(dataset, delta_t=delta_t)
            result = {'Name': dataset.name, '△t': delta_t}
            result = result | get_aggregated_properties(data, links)
            results.append(result)
    return results


res = grid_search(datasets_physical, delta_ts_physical)
# infect = {'Name': "Infectious", '△t': 1 * D}
# infect = infect | get_aggregated_properties(get_infectious())
# res.append(infect)
df = pd.DataFrame(res)
df.to_csv('grid_search_physical.csv', index=False)

df = pd.DataFrame(grid_search(datasets_virtual, delta_ts_virtual))
df.to_csv('grid_search_virtual.csv', index=False)
