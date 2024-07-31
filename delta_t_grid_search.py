import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from utils import *
from get_data import *

delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 12 * H, 1 * D]
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [get_college_1(), get_college_2(), get_socio_calls(), get_socio_sms()]


def weighted_jaccard_similarity(G1, G2):
    # Get the edge sets with weights
    edges_G1 = {edge: G1[edge[0]][edge[1]]['weight'] for edge in G1.edges()}
    edges_G2 = {edge: G2[edge[0]][edge[1]]['weight'] for edge in G2.edges()}

    # Calculate the intersection and union of the edge sets with weights
    intersection_weight = 0.0
    union_weight = 0.0

    all_edges = set(edges_G1.keys()).union(set(edges_G2.keys()))

    for edge in all_edges:
        weight_G1 = edges_G1.get(edge, 0)
        weight_G2 = edges_G2.get(edge, 0)

        intersection_weight += min(weight_G1, weight_G2)
        union_weight += max(weight_G1, weight_G2)

    if union_weight == 0:
        return 0.0

    return intersection_weight / union_weight


def calculate_jaccard_similarities(snapshots):
    similarities = []
    for i in range(len(snapshots) - 1):
        similarity = weighted_jaccard_similarity(snapshots[i], snapshots[i + 1])
        similarities.append(similarity)
    return similarities


def calculate_autocorrelation(similarities):
    if len(similarities) > 1:
        return pearsonr(similarities[:-1], similarities[1:])[0]
    return 0


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

    # Calculate Jaccard similarities and autocorrelation
    jaccard_similarities = calculate_jaccard_similarities(snapshots)
    avg_jaccard_similarity = np.mean(jaccard_similarities)
    std_jaccard_similarity = np.std(jaccard_similarities)
    autocorrelation = calculate_autocorrelation(jaccard_similarities)

    return {
        "n": num_snapshots,
        # "total_links": all_edges,
        # "total_interaction": all_interactions,
        # "mu_links": round(avg_edges_per_snapshot * 100, 2),
        # "std_links": round(std_edges_per_snapshot * 100, 2),
        # "mu_contacts": round(avg_interactions_per_snapshot * 100, 2),
        # "std_contacts": round(std_interactions_per_snapshot * 100, 2),
        "avg_jaccard_similarity": round(avg_jaccard_similarity, 4),
        "std_jaccard_similarity": round(std_jaccard_similarity, 4),
        "autocorrelation": round(autocorrelation, 4),
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
