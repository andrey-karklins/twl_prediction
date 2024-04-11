import networkx as nx

import get_data

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds


def extract_network_features(snapshots):
    """
    Extracts an expanded set of network features for each snapshot in the array of nx.Graph() objects.
    This includes additional node-level metrics (betweenness centrality, closeness centrality, PageRank)
    and edge-level features (average edge weight, maximum edge weight), alongside the previously included metrics.

    :param snapshots: List of nx.Graph() objects ordered by timestamp.
    :return: Dictionary with snapshot index as keys and extracted features as values.
    """
    features_per_snapshot = {}

    for index, graph in enumerate(snapshots):
        # Node-level features
        degrees = dict(graph.degree())
        clustering_coefficients = nx.clustering(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)

        # Graph-level features
        avg_clustering_coefficient = nx.average_clustering(graph)
        density = nx.density(graph)

        # Edge-level features
        weights = nx.get_edge_attributes(graph, 'weight')
        average_edge_weight = sum(weights.values()) / len(weights) if weights else 0
        max_edge_weight = max(weights.values()) if weights else 0

        # Storing extracted features
        features_per_snapshot[index] = {
            'degrees': degrees,
            'clustering_coefficients': clustering_coefficients,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'pagerank': pagerank,
            'avg_clustering_coefficient': avg_clustering_coefficient,
            'density': density,
            'average_edge_weight': average_edge_weight,
            'max_edge_weight': max_edge_weight,
        }

    return features_per_snapshot


print(extract_network_features(get_data.aggregate_into_snapshots(get_data.get_hypertext(), delta_t=H)))
