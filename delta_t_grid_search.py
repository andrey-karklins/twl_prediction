from data_analysis import plot_lag_autocorrelation, plot_lag_weighted_jaccard_similarity, plot_autocorrelation_jaccard
from get_data import *

delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 12 * H, 1 * D]
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [
    get_college_1(),
    get_college_2(),
    get_socio_calls(),
    get_socio_sms()
]


def get_aggregated_properties(snapshots, links):
    num_snapshots = len(snapshots)
    all_edges = len(links)
    edges_per_snapshot = [len(g.edges()) for g in snapshots]
    interactions_per_snapshot = [sum(data['weight'] for _, _, data in g.edges(data=True)) for g in snapshots]


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
    avg_interactions_per_snapshot = np.mean(
        [interactions / all_interactions for interactions in interactions_per_snapshot])
    std_interactions_per_snapshot = np.std(
        [interactions / all_interactions for interactions in interactions_per_snapshot])


    return {
        "n": num_snapshots,
        "total_links": all_edges,
        "total_interaction": all_interactions,
        "mu_links": round(avg_edges_per_snapshot * 100, 2),
        "std_links": round(std_edges_per_snapshot * 100, 2),
        "mu_contacts": round(avg_interactions_per_snapshot * 100, 2),
        "std_contacts": round(std_interactions_per_snapshot * 100, 2),
    }


def aggregate_data(datasets, delta_t):
    results = []
    for dataset in datasets:
        data = aggregate_to_matrix(dataset, delta_t=delta_t)
        results.append(data)
    return results


plot_autocorrelation_jaccard(datasets_virtual, delta_ts_virtual)

