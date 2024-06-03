import networkx as nx
import pandas as pd
from tsfresh import extract_features
from utils import *
from get_data import *

datasets = [(get_hypertext(), 20 * M),
            (get_SFHH(), 10 * M),
            (get_infectious(), None),
            (get_college_1(), D),
            (get_college_2(), 2 * D),
            (get_socio_calls(), 2 * D),
            (get_socio_sms(), 6 * H),
            ]

datasets = list(map(lambda x: (aggregate_into_snapshots(x[0], delta_t=x[1]), x[0].name) if
x[1] is not None else x[0], datasets))


def extract_edge_features_undirected(snapshots, u, v, i):
    rows = []
    for G in snapshots:
        t = G.graph['t']
        degree_u = G.degree(u) if u in G else 0
        degree_v = G.degree(v) if v in G else 0
        strength_u = sum([G[u][n]['weight'] for n in G[u]]) if u in G else 0
        strength_v = sum([G[v][n]['weight'] for n in G[v]]) if v in G else 0
        features = {
            'time': t,
            'id': i,
            'edge_weight': G[u][v]['weight'] if G.has_edge(u, v) else 0,
            'degree_avg': (degree_u + degree_v) / 2,
            'strength_avg': (strength_u + strength_v) / 2,
            'common_neighbors': len(list(nx.common_neighbors(G, u, v))) if u in G and v in G else 0,
            'jaccard_coefficient': list(nx.jaccard_coefficient(G, [(u, v)]))[0][2] if u in G and v in G else 0
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
            'density': nx.density(G)
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


def get_folds(data, n_folds, observation_size):
    folds = []
    fold_size = len(data) // n_folds
    if observation_size > fold_size:
        raise ValueError("Observation size should be less than the fold size")
    for i in range(n_folds):
        snapshots = data[i * fold_size:i * fold_size + fold_size]
        observation_windows = []
        for j in range(len(snapshots) - observation_size + 1):
            observation_windows.append(snapshots[j:j + observation_size])
        folds.append(observation_windows)
    return folds


def train_test_feature_extraction():
    dataset = get_socio_sms()
    delta_t = 1 * D
    n_folds = 5
    observation_size = 10

    data = aggregate_into_snapshots(dataset, delta_t=delta_t)
    folds = get_folds(data, n_folds, observation_size)
    for i, fold in enumerate(folds):
        X_trains = []
        Y_trains = []
        for observation_window in fold[:-1]:
            X_train, Y_train = extract_train_data(observation_window[:-1], observation_window[-1])
            X_trains.append(X_train)
            Y_trains.append(Y_train)
        X_test, Y_test = extract_train_data(fold[-1][:-1], fold[-1][-1])
        X_test.to_csv(f'train_test_data/X_test_{i + 1}.csv', index=False)
        Y_test.to_csv(f'train_test_data/Y_test_{i + 1}.csv', index=False)
        pd.concat(X_trains).to_csv(f'train_test_data/X_train_{i + 1}.csv', index=False)
        pd.concat(Y_trains).to_csv(f'train_test_data/Y_train_{i + 1}.csv', index=False)

if __name__ == '__main__':
    train_test_feature_extraction()