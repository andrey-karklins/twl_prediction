import numpy as np
from concurrent.futures import ThreadPoolExecutor
from models.SDModel import SDModel


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

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
        # No fitting process needed for SCDModel
        pass

    def _get_neighbors_sum(self, sd_predictions, neighbor_edges_cache_1, neighbor_edges_cache_2):
        neighbors_average = np.zeros(len(sd_predictions))

        # Define a function to compute the average for a single neighbor index
        def compute_average(i):
            neighbors_1 = 0
            if len(neighbor_edges_cache_1[i]) != 0:
                neighbors_1 = geo_mean(sd_predictions[list(neighbor_edges_cache_1[i])])
            neighbors_2 = 0
            if len(neighbor_edges_cache_2[i]) != 0:
                neighbors_2 = geo_mean(sd_predictions[list(neighbor_edges_cache_2[i])])

            return (neighbors_1 + neighbors_2) / 2

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            results = executor.map(compute_average, range(len(neighbors_average)))

        # Gather results
        neighbors_average = np.fromiter(results, dtype=np.float64)
        return neighbors_average

    def _get_common_neighbors_sum(self, sd_predictions, common_neighbor_geometric_cache):
        neighbors_average = np.zeros(len(sd_predictions))

        # Define a function to compute the geometric mean for a common neighbor index
        def compute_geometric_average(i):
            if len(common_neighbor_geometric_cache[i]) == 0:
                return 0
            results = list(
                map(lambda x: np.sqrt(sd_predictions[x[0]] * sd_predictions[x[1]]), common_neighbor_geometric_cache[i]))
            return sum(results) / len(results)

        # Parallelize with ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = executor.map(compute_geometric_average, range(len(neighbors_average)))

        # Collect results
        neighbors_average = np.fromiter(results, dtype=np.float64)
        return neighbors_average

    def predict(self, X, indices):
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        # Cache predictions from SDModel
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)

        # Define function to compute predictions for a single time index
        def compute_prediction(i, t):
            # Self-driven component
            self_driven = self.alpha * self.sd_model_predictions_cache[i]

            # Neighbor-driven component
            neighbor_driven = self.beta * self._get_neighbors_sum(
                self.sd_model_predictions_cache[i], self.G_global.neighbor_edges_cache_1, self.G_global.neighbor_edges_cache_2
            )

            # Common neighbor-driven component
            common_neighbor_driven = self.gamma * self._get_common_neighbors_sum(
                self.sd_model_predictions_cache[i], self.G_global.common_neighbor_geometric_cache
            )

            # Total prediction for each link at time t
            total_driven = self_driven + neighbor_driven + common_neighbor_driven
            return total_driven

        # Parallelize the prediction computations for each time index
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda idx_t: compute_prediction(*idx_t), enumerate(indices))

        # Collect the results
        for i, result in enumerate(results):
            predictions[i] = result

        return predictions
