import numpy as np
from sklearn.linear_model import Lasso

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
    def __init__(self, tau, L, G_global, alpha=0.01):
        """
        Initialize the SCDModel with parameters.

        Parameters:
        - tau: Decay factor
        - L: Number of past time steps to consider
        - alpha: Regularization strength for Lasso regression
        - G_global: The global graph structure
        """
        self.tau = tau
        self.L = L
        self.alpha = alpha  # Regularization strength for Lasso
        self.G_global = G_global
        self.SDModel = SDModel(tau, L)
        self.lasso = Lasso(alpha=self.alpha)
        self.coefficients_ = None
        self.sd_model_predictions_cache = None

    def fit(self, X, indices):
        """
        Fit the model using Lasso regression.

        Parameters:
        - X: Input data, shape (T, M)
        - indices: Time indices used as targets for training
        """
        T, M = X.shape
        n_samples = len(indices)
        n_features = 3  # beta1, beta2, beta3

        # Prepare feature matrix and target vector
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)
        feature_matrix = np.zeros((n_samples * M, n_features))
        target_vector = np.repeat(indices, M)  # Use indices as the target

        for i, t in enumerate(indices):
            # Self-driven component
            self_driven = self.sd_model_predictions_cache[i].flatten()

            # Common neighbor-driven component
            common_neighbor_driven = _get_common_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.common_neighbor_geometric_cache
            ).flatten()

            # Neighbor-driven component
            neighbor_driven = _get_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.neighbor_edges_cache_1,
                self.G_global.neighbor_edges_cache_2
            ).flatten()

            # Add features to the matrix
            feature_matrix[i * M:(i + 1) * M, 0] = self_driven
            feature_matrix[i * M:(i + 1) * M, 1] = common_neighbor_driven
            feature_matrix[i * M:(i + 1) * M, 2] = neighbor_driven

        # Fit Lasso model
        self.lasso.fit(feature_matrix, target_vector)
        self.coefficients_ = self.lasso.coef_
        self.beta0 = self.lasso.intercept_

    def predict(self, X, indices):
        """
        Predict using the fitted SCDModel.

        Parameters:
        - X: Input data, shape (T, M)
        - indices: Time indices for prediction

        Returns:
        - predictions: Predicted values, shape (len(indices), M)
        """
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        # Cache predictions from SDModel
        self.sd_model_predictions_cache = self.SDModel.predict(X, indices)

        # Compute predictions for each time index sequentially
        for i, t in enumerate(indices):
            # Self-driven component
            self_driven = self.coefficients_[0] * self.sd_model_predictions_cache[i]

            # Common neighbor-driven component
            common_neighbor_driven = self.coefficients_[1] * _get_common_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.common_neighbor_geometric_cache
            )

            # Neighbor-driven component
            neighbor_driven = self.coefficients_[2] * _get_neighbors_sum(
                self.sd_model_predictions_cache[i],
                self.G_global.neighbor_edges_cache_1,
                self.G_global.neighbor_edges_cache_2
            )

            # Total prediction for each link at time t with bias term
            total_driven = self.beta0 + self_driven + neighbor_driven + common_neighbor_driven
            predictions[i] = total_driven

        return predictions
