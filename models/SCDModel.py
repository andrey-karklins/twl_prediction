import numpy as np
from sklearn.linear_model import Lasso


def _get_common_neighbors_sum(sd_predictions, common_neighbor_geometric_cache):
    neighbors_average = np.zeros(len(sd_predictions))

    # Iterate only over indices with non-empty neighbor lists
    for i, neighbors in common_neighbor_geometric_cache.items():
        if len(neighbors) > 0:
            # Vectorized computation of the geometric means of neighbors' predictions
            products = np.sqrt(sd_predictions[neighbors[:, 0]] * sd_predictions[neighbors[:, 1]])
            # Compute the mean of products directly for this index
            neighbors_average[i] = np.mean(products)

    return neighbors_average


def _get_neighbors_sum(sd_predictions, neighbor_edges_cache_1, neighbor_edges_cache_2):
    neighbors_average = np.zeros(len(sd_predictions))

    # Iterate only over indices where either cache has neighbors
    for i in range(len(neighbors_average)):
        neighbors_1 = neighbor_edges_cache_1.get(i, [])
        neighbors_2 = neighbor_edges_cache_2.get(i, [])

        # If both are empty, skip this index
        if len(neighbors_1) == 0 and len(neighbors_2) == 0:
            continue

        # Concatenate only non-empty arrays for efficiency
        if len(neighbors_1) > 0 and len(neighbors_2) > 0:
            neighbors = np.concatenate((neighbors_1, neighbors_2))
        elif len(neighbors_1) > 0:
            neighbors = neighbors_1
        else:
            neighbors = neighbors_2

        # Calculate the average for non-empty neighbors
        neighbors_average[i] = sd_predictions[neighbors].mean()

    return neighbors_average


class SCDModel:
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

    def fit(self, sd_predictions, y, alpha=0.05, max_iter=1000, tol=1e-6):
        """
        Fits the model using Lasso regression to minimize Mean Squared Error (MSE).

        Parameters:
        sd_predictions: numpy array, shape (n_samples,n_targets)
            The base predictions (self-driven component).
        y: numpy array, shape (n_samples,n_targets)
            The target variable (continuous positive values).
        alpha: float, optional (default=0.1)
            The regularization strength for Lasso.
        max_iter: int, optional (default=1000)
            The maximum number of iterations for Lasso optimization.
        """
        # Preallocate the feature matrix and target array
        num_samples, num_predictions = sd_predictions.shape
        X = np.zeros((num_samples * num_predictions, 4))  # Assuming 4 features
        Y = y.flatten()  # Flatten if `y` is a 2D array matching `sd_predictions`

        # Compute the feature matrix in a vectorized manner
        features = np.vstack([self._compute_features(sd_pred) for sd_pred in sd_predictions])

        # Assign features and targets directly
        X[:features.shape[0], :] = features
        Y[:features.shape[0]] = y.ravel()  # Ensure y is flattened if it's a 2D array

        # Fit Lasso regression to minimize MSE
        lasso = Lasso(alpha=alpha, max_iter=max_iter, positive=True, tol=tol, fit_intercept=False)
        lasso.fit(X, Y)
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
