import numpy as np
import pandas as pd

from get_data import *

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds

delta_ts_physical = [5 * M, 10 * M, 20 * M, 40 * M, 1 * H, 2 * H, 3 * H, 1 * D]
delta_ts_virtual = [3 * H, 6 * H, 12 * H, 1 * D, 2 * D, 3 * D, 7 * D, 14 * D]
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [get_college_1(), get_college_2(), get_socio_calls(), get_socio_sms()]


def get_aggregated_properties(snapshots):
    num_snapshots = len(snapshots)

    # Compute the number of interactions and total weight per snapshot
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
    avg_edges_per_snapshot = np.mean(edges_per_snapshot)
    avg_interactions_per_snapshot = np.mean(interactions_per_snapshot)
    std_interactions_per_snapshot = np.std(interactions_per_snapshot)
    avg_weight_per_snapshot = np.mean(weights_per_snapshot)
    std_weight_per_snapshot = np.std(weights_per_snapshot)
    avg_min_weight = np.mean(min_weights)
    avg_max_weight = np.mean(max_weights)

    return {
        "n_snapshots": num_snapshots,
        "avg_edges": round(avg_edges_per_snapshot, 2),
        "avg_interactions": round(avg_interactions_per_snapshot, 2),
        "std_interactions": round(std_interactions_per_snapshot, 2),
        "avg_min_weight": round(avg_min_weight, 2),
        "avg_weight": round(avg_weight_per_snapshot, 2),
        "avg_max_weight": round(avg_max_weight, 2),
        "std_weight": round(std_weight_per_snapshot, 2),
        "weight_variation_coeff": round(std_weight_per_snapshot/avg_weight_per_snapshot, 2)
    }


def grid_search(datasets, delta_ts):
    results = []
    for dataset in datasets:
        for delta_t in delta_ts:
            data = aggregate_into_snapshots(dataset, delta_t=delta_t)
            if len(data) < 50:
                break
            result = {'Name': dataset.name, '△t': delta_t}
            result = result | get_aggregated_properties(data)
            results.append(result)
    return results


res = grid_search(datasets_physical, delta_ts_physical)
infect = {'Name': "Infectious", '△t': 1 * D}
infect = infect | get_aggregated_properties(get_infectious())
res.append(infect)
df = pd.DataFrame(res)
df.to_csv('grid_search_physical.csv', index=False)

df = pd.DataFrame(grid_search(datasets_virtual, delta_ts_virtual))
df.to_csv('grid_search_virtual.csv', index=False)
