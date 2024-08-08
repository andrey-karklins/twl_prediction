import numpy as np


class SCDModel:
    def __init__(self, tau, L, alpha, beta, gamma, G_global):
        self.tau = tau
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.G_global = G_global

    def fit(self, X, y=None):
        pass  # No fitting process needed for SCDModel

    def predict(self, X):
        n_samples, n_features = X.shape
        predictions = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            start_index = max(0, i - self.L + 1)
            decay_factors = np.exp(-self.tau * np.arange(i - start_index, -1, -1))
            for j in range(n_features):
                edge = list(self.G_global.edges())[j]

                # Self-driven component
                weighted_sum = np.sum(X[start_index:i + 1, j] * decay_factors)
                self_driven = self.alpha * weighted_sum

                neighbors_1 = list(self.G_global.neighbors(edge[0]))
                neighbors_1.remove(edge[1])
                neighbors_2 = list(self.G_global.neighbors(edge[1]))
                neighbors_2.remove(edge[0])

                distinct_neighbors_edges = []
                for neighbor in neighbors_1:
                    if neighbor not in neighbors_2:
                        distinct_neighbors_edges.append(self.G_global.edges[edge[0], neighbor]["id"])
                for neighbor in neighbors_2:
                    if neighbor not in neighbors_1:
                        distinct_neighbors_edges.append(self.G_global.edges[edge[1], neighbor]["id"])
                neighbor_sum = 0
                for neighbor_edge in distinct_neighbors_edges:
                    neighbor_sum += np.sum(X[start_index:i + 1, neighbor_edge] * decay_factors)
                neighbor_driven = self.beta * neighbor_sum

                # Common neighbor-driven component
                common_neighbors = list(set(neighbors_1) & set(neighbors_2))
                common_neighbors_sum = 0
                for common_neighbor in common_neighbors:
                    common_neighbors_sum += np.sum(
                        X[start_index:i + 1, self.G_global.edges[edge[0], common_neighbor]["id"]] * decay_factors)
                    common_neighbors_sum += np.sum(
                        X[start_index:i + 1, self.G_global.edges[edge[1], common_neighbor]["id"]] * decay_factors)
                common_neighbor_driven = self.gamma * common_neighbors_sum

                # Total prediction
                predictions[i, j] = self_driven + neighbor_driven + common_neighbor_driven

        return predictions
