


# datasets = [(get_data.get_hypertext(), 20 * M),
#             # (get_data.get_SFHH(), 10 * M),
#             # (get_data.get_infectious(), None),
#             # (get_data.get_college_1(), D),
#             # (get_data.get_college_2(), 2 * D),
#             # (get_data.get_socio_calls(), 2 * D),
#             # (get_data.get_socio_sms(), 6 * H),
#             ]
#
# datasets = list(map(lambda x: (get_data.aggregate_into_snapshots(x[0], delta_t=x[1]), x[0].name) if
# x[1] is not None else x[0], datasets))

import networkx as nx
import pandas as pd
from tsfresh import extract_features

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds

def extract_edge_features_undirected(snapshots, u, v, i):
    rows = []
    for G in snapshots:
        t = G.graph['t']
        degree_u = G.degree(u) if u in G else 0
        degree_v = G.degree(v) if v in G else 0
        strength_u = sum([G[u][n]['weight'] for n in G[u]]) if u in G else 0
        strength_v = sum([G[v][n]['weight'] for n in G[v]]) if v in G else 0
        betweenness = nx.betweenness_centrality(G)
        features = {
            'time': t,
            'id': i,
            'edge_weight': G[u][v]['weight'] if G.has_edge(u, v) else 0,
            'degree_avg': (degree_u + degree_v) / 2,
            'strength_avg': (strength_u + strength_v) / 2,
            # 'clustering_avg': (nx.clustering(G, u) + nx.clustering(G, v)) / 2 if u in G and v in G else 0,
            # 'betweenness_avg': (betweenness.get(u, 0) + betweenness.get(v, 0)) / 2,
            # 'common_neighbors': len(list(nx.common_neighbors(G, u, v))) if u in G and v in G else 0,
            # 'jaccard_coefficient': list(nx.jaccard_coefficient(G, [(u, v)]))[0][2] if u in G and v in G else 0
        }
        rows.append(features)
    return pd.DataFrame(rows)


def extract_graph_features(snapshots):
    rows = []
    for G in snapshots:
        features = {
            'time': G.graph['t'],
            'n_interactions': sum([w for (_, _, w) in G.edges.data("weight", default=0)]),
            'average_degree': sum(dict(G.degree()).values()) / float(
                G.number_of_nodes()) if G.number_of_nodes() > 0 else 0,
            'density': nx.density(G),
            'transitivity': nx.transitivity(G)
        }
        rows.append(features)
    return pd.DataFrame(rows)


def extract_train_data(snapshots, test_snapshot):
    # Initialize empty lists to collect data
    X_data = []
    Y_data = []

    # Extracting graph-level features once since it doesn't depend on the individual edge
    graph_features = extract_graph_features(snapshots)

    for (i, (u, v, w)) in enumerate(test_snapshot.edges.data("weight", default=0)):
        edge_features = extract_edge_features_undirected(snapshots, u, v, i)
        Y_data.append(w)
        edge_features = pd.merge(edge_features, graph_features, on='time', how='inner')
        X_data.append(edge_features)

    X = pd.concat(X_data, ignore_index=True)
    Y = pd.Series(Y_data)
    return extract_features(X, column_id="id"), Y
