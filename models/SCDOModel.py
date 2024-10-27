import numpy as np
from scipy.stats import gmean

from models.SDModel import SDModel
from utils import geo_mean


class SCDOModel:
    def __init__(self, tau, L, alpha, beta, gamma, G_global):
        self.tau = tau  # Decay factor
        self.L = L  # Number of past time steps to consider
        self.alpha = alpha  # Weight for self-driven component
        self.beta = beta  # Weight for neighbor-driven component
        self.gamma = gamma  # Weight for common neighbor-driven component
        self.G_global = G_global  # The global graph structure
        self.SDModel = SDModel(tau, L)
        self.sd_model_predictions_cache = None

    def fit(self, X, y=None):
        # No fitting process needed for SCDOModel
        pass

    def _get_neighbors_sum(self, sd_predictions, neighbor_edges_cache_1, neighbor_edges_cache_2):
        neighbors_average = np.zeros(len(sd_predictions))

        # Compute the average for each neighbor index sequentially
        for i in range(len(neighbors_average)):
            neighbors_1 = geo_mean(sd_predictions[neighbor_edges_cache_1[i]]) if len(neighbor_edges_cache_1[i]) > 0 else 0
            neighbors_2 = geo_mean(sd_predictions[neighbor_edges_cache_2[i]]) if len(neighbor_edges_cache_2[i]) > 0 else 0
            neighbors_average[i] = (neighbors_1 + neighbors_2) / 2

        return neighbors_average

    import numpy as np

    def _get_common_neighbors_sum(self, sd_predictions, common_neighbor_geometric_cache):
        neighbors_average = np.zeros(len(sd_predictions))

        # Iterate over each possible index in order
        for i in range(len(neighbors_average)):
            if i in common_neighbor_geometric_cache and len(common_neighbor_geometric_cache[i]) > 0:
                # Convert neighbors to a NumPy array of shape (n, 2) for vectorized access
                neighbors = np.array(common_neighbor_geometric_cache[i])
                # Use NumPy to index sd_predictions and calculate the square root of products
                products = np.sqrt(sd_predictions[neighbors[:, 0]] * sd_predictions[neighbors[:, 1]])
                # Assign the mean of products to the correct index
                neighbors_average[i] = products.mean()

        return neighbors_average

    def predict(self, X, indices):
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        # Cache predictions from SDModel
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)

        # Compute predictions for each time index sequentially
        for i, t in enumerate(indices):
            # Self-driven component
            self_driven = self.alpha * self.sd_model_predictions_cache[i]

            # Neighbor-driven component
            neighbor_driven = self.beta * self._get_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.neighbor_edges_cache_1,
                self.G_global.neighbor_edges_cache_2
            )

            # Common neighbor-driven component
            common_neighbor_driven = self.gamma * self._get_common_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.common_neighbor_geometric_cache
            )

            # Total prediction for each link at time t
            total_driven = self_driven + neighbor_driven + common_neighbor_driven
            predictions[i] = total_driven

        return predictions
