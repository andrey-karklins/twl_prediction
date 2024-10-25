import numpy as np

from models.SDModel import SDModel


def _get_neighbors_sum(sd_predictions, neighbor_edges_cache):
    neighbors_average = np.zeros(len(sd_predictions))
    for i in range(len(neighbors_average)):
        if len(neighbor_edges_cache[i]) == 0:
            continue
        neighbors_average[i] += sd_predictions[list(neighbor_edges_cache[i])].sum() / len(
            neighbor_edges_cache[i])
    return neighbors_average

class OriginalSCDModel:
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
        # No fitting process needed for SCDModel
        pass

    def _get_common_neighbors_sum(self, sd_predictions, common_neighbor_edges_cache):
        neighbors_average = np.zeros(len(sd_predictions))
        for i in range(len(neighbors_average)):
            if len(common_neighbor_edges_cache[i]) == 0:
                continue
            results = list(map(lambda x: np.sqrt(sd_predictions[x[0]]*sd_predictions[x[1]]),common_neighbor_edges_cache))
            neighbors_average[i] += sum(results) / len(results[i])
        return neighbors_average

    def predict(self, X, indices):
        T, M = X.shape
        predictions = np.zeros((len(indices), M))
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)
        for i, t in enumerate(indices):
            # Self-driven component
            self_driven = self.alpha * self.sd_model_predictions_cache[i]

            # Neighbor-driven component
            neighbor_driven = self.beta * _get_neighbors_sum(self.sd_model_predictions_cache[i],
                                                             self.G_global.neighbor_edges_cache)

            # Common neighbor-driven component
            common_neighbor_driven = self.gamma * _get_neighbors_sum(self.sd_model_predictions_cache[i],
                                                                     self.G_global.common_neighbor_edges_cache)

            # Total prediction for each link at time t
            total_driven = self_driven + neighbor_driven + common_neighbor_driven

            predictions[i] = total_driven

        return predictions
