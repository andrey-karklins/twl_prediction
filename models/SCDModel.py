import numpy as np


def _get_neighbors_sum(X, cache):
    neighbors_sum = np.zeros(X.shape[1])
    for edge in range(X.shape[1]):
        if len(cache[edge]) == 0:
            continue
        neighbors_sum[edge] += X[:, cache[edge]].sum() / len(cache[edge])
    return neighbors_sum


class SCDModel:
    def __init__(self, tau, L, alpha, beta, gamma, G_global):
        self.tau = tau  # Decay factor
        self.L = L  # Number of past time steps to consider
        self.alpha = alpha  # Weight for self-driven component
        self.beta = beta  # Weight for neighbor-driven component
        self.gamma = gamma  # Weight for common neighbor-driven component
        self.G_global = G_global  # The global graph structure
        # Map both edge (1, 2) and (2, 1) to the same ID
        self.edge_to_id = {tuple(sorted(edge)): i for i, edge in enumerate(self.G_global.edges())}

        # Cache neighbor and common neighbor edges for each edge in the graph
        self.neighbor_edges_cache = {}
        self.common_neighbor_edges_cache = {}

        self._cache_neighbor_edges()
        self._cache_common_neighbor_edges()

    def _cache_neighbor_edges(self):
        """Precompute and cache the neighbor edges for each edge."""
        for edge in self.G_global.edges():
            sorted_edge = tuple(sorted(edge))
            neighbors_0 = list(self.G_global.neighbors(edge[0]))
            neighbors_1 = list(self.G_global.neighbors(edge[1]))

            neighbors_0_edges = [self.edge_to_id[tuple(sorted((edge[0], neighbor)))]
                                 for neighbor in neighbors_0
                                 if neighbor != edge[1] and tuple(sorted((edge[0], neighbor))) in self.edge_to_id]

            neighbors_1_edges = [self.edge_to_id[tuple(sorted((edge[1], neighbor)))]
                                 for neighbor in neighbors_1
                                 if neighbor != edge[0] and tuple(sorted((edge[1], neighbor))) in self.edge_to_id]

            self.neighbor_edges_cache[self.edge_to_id[sorted_edge]] = np.array(neighbors_0_edges + neighbors_1_edges)

    def _cache_common_neighbor_edges(self):
        """Precompute and cache the common neighbor edges for each edge."""
        for edge in self.G_global.edges():
            sorted_edge = tuple(sorted(edge))
            common_neighbors = set(self.G_global.neighbors(edge[0])) & set(self.G_global.neighbors(edge[1]))

            common_neighbors_edges = [self.edge_to_id[tuple(sorted((edge[0], cn)))]
                                      for cn in common_neighbors
                                      if tuple(sorted((edge[0], cn))) in self.edge_to_id] + \
                                     [self.edge_to_id[tuple(sorted((edge[1], cn)))]
                                      for cn in common_neighbors
                                      if tuple(sorted((edge[1], cn))) in self.edge_to_id]

            self.common_neighbor_edges_cache[self.edge_to_id[sorted_edge]] = np.array(common_neighbors_edges)

    def fit(self, X, y=None):
        # No fitting process needed for SCDModel
        pass

    def predict(self, X, indices):
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        for i,t in enumerate(indices):
            start_index = max(0, t - self.L)
            # Calculate the time indices for the current window
            time_indices = np.arange(start_index, t)
            # Calculate the exponential decay factors
            decay_factors = np.exp(-self.tau * (t - time_indices))
            decay_factors /= decay_factors.sum()  # Normalize decay factors
            # Calculate the weighted sum for each feature (link) using the decay factors
            weighted_X = decay_factors[:, np.newaxis] * X[start_index:t]

            # Self-driven component
            self_driven = self.alpha * weighted_X.sum(axis=0)

            # Neighbor-driven component
            neighbor_driven = self.beta * _get_neighbors_sum(weighted_X, self.neighbor_edges_cache)

            # Common neighbor-driven component
            common_neighbor_driven = self.gamma * _get_neighbors_sum(weighted_X, self.common_neighbor_edges_cache)

            # Total prediction for each link at time t
            total_driven = self_driven + neighbor_driven + common_neighbor_driven

            predictions[i] = total_driven

        return predictions
