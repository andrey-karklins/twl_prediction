import numpy as np
from models.SDModel import SDModel

def _get_common_neighbors_sum(sd_predictions, common_neighbor_geometric_cache):
    neighbors_average = np.zeros(len(sd_predictions))

    # Iterate only over indices with non-empty neighbor lists
    for i, neighbors in common_neighbor_geometric_cache.items():
        if len(neighbors) > 0:
            neighbors = np.array(neighbors)  # Convert once for each non-empty entry
            # Vectorized computation of the geometric means of neighbors' predictions
            products = np.sqrt(sd_predictions[neighbors[:, 0]] * sd_predictions[neighbors[:, 1]])
            # Compute the mean of products directly for this index
            neighbors_average[i] = np.mean(products)

    return neighbors_average


def _get_neighbors_sum(sd_predictions, neighbor_edges_cache_1, neighbor_edges_cache_2):
    neighbors_average = np.zeros(len(sd_predictions))

    # Compute the average for each neighbor index sequentially
    for i in range(len(neighbors_average)):
        match (len(neighbor_edges_cache_1[i]) == 0, len(neighbor_edges_cache_2[i]) == 0):
            case (False, False):
                neighbors_average[i] = np.sqrt(
                    np.mean(sd_predictions[neighbor_edges_cache_1[i]]) * np.mean(
                        sd_predictions[neighbor_edges_cache_2[i]]))
                continue
            case (True, False):
                neighbors_average[i] = np.mean(sd_predictions[neighbor_edges_cache_2[i]])
                continue
            case (False, True):
                neighbors_average[i] = np.mean(sd_predictions[neighbor_edges_cache_1[i]])
                continue
            case (True, True):
                continue
    return neighbors_average


class SCDOModel:
    def __init__(self, tau, L, alpha, beta, gamma, G_global):
        self.tau = tau  # Decay factor
        self.L = L  # Number of past time steps to consider
        self.alpha = alpha  # Weight for self-driven component
        self.beta = beta  # Weight for common neighbor-driven component
        self.gamma = gamma  # Weight for distinct neighbor-driven component
        self.G_global = G_global  # The global graph structure
        self.SDModel = SDModel(tau, L)
        self.sd_model_predictions_cache = None

    def fit(self, X, y=None):
        # No fitting process needed for SCDModel
        pass

    import numpy as np

    def predict(self, X, indices):
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        # Cache predictions from SDModel
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)

        # Compute predictions for each time index sequentially
        for i, t in enumerate(indices):
            # Self-driven component
            self_driven = self.alpha * self.sd_model_predictions_cache[i]

            # Common neighbor-driven component
            common_neighbor_driven = self.beta * _get_common_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.common_neighbor_geometric_cache
            )

            # Neighbor-driven component
            neighbor_driven = self.gamma * _get_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.neighbor_edges_cache_1,
                self.G_global.neighbor_edges_cache_2
            )

            # Total prediction for each link at time t
            total_driven = self_driven + neighbor_driven + common_neighbor_driven
            predictions[i] = total_driven

        return predictions
