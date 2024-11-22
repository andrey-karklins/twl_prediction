import numpy as np
from sklearn.linear_model import Lasso

from models.SCDModel import _get_common_neighbors_sum


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
    def __init__(self, G_global):
        self.G_global = G_global
        self.betas = None
        self.has_converged = False

    def _compute_features(self, sd_predictions):
        """
        Computes the feature matrix components for the model:
        - Intercept term
        - Self-driven component
        - Common neighbor-driven component
        - Distinct neighbor-driven component

        Returns:
        X: numpy array, shape (n_samples, 4)
            Feature matrix including intercept, self-driven, common neighbors, and distinct neighbors.
        """
        intercept = np.ones(sd_predictions.shape[0])  # Intercept term

        self_driven = sd_predictions  # Self-driven component

        common_neighbor_driven = _get_common_neighbors_sum(
            sd_predictions, self.G_global.common_neighbor_geometric_cache
        )

        neighbor_driven = _get_neighbors_sum(
            sd_predictions, self.G_global.neighbor_edges_cache_1, self.G_global.neighbor_edges_cache_2
        )

        # Combine features into a matrix
        X = np.column_stack([intercept, self_driven, common_neighbor_driven, neighbor_driven])
        return X

    def fit(self, sd_predictions, y, alpha=0.05, max_iter=2000, tol=1e-6):
        """
        Fits the model using Lasso regression to minimize Mean Squared Error (MSE).

        Parameters:
        sd_predictions: numpy array, shape (n_samples,)
            The base predictions (self-driven component).
        y: numpy array, shape (n_samples,)
            The target variable (continuous positive values).
        alpha: float, optional (default=0.1)
            The regularization strength for Lasso.
        max_iter: int, optional (default=1000)
            The maximum number of iterations for Lasso optimization.
        """
        # Compute the feature matrix
        X = self._compute_features(sd_predictions)

        # Fit Lasso regression to minimize MSE
        lasso = Lasso(alpha=alpha, max_iter=max_iter, positive=True, tol=tol, fit_intercept=False)
        lasso.fit(X, y)
        self.has_converged = lasso.n_iter_ < max_iter

        # Store the learned coefficients (including intercept)
        self.betas = lasso.coef_

    def predict(self, sd_predictions):
        """
        Predicts the target variable using the learned coefficients.
        """
        # Compute feature matrix
        X = self._compute_features(sd_predictions)

        # Prediction: dot product of X and betas
        return X @ self.betas
