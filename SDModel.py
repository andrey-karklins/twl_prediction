import numpy as np


class SDModel:
    def __init__(self, tau, L):
        self.tau = tau
        self.L = L

    def fit(self, X, y=None):
        pass  # No fitting process needed for SDModel

    def predict(self, X):
        n_samples, n_features = X.shape
        predictions = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            start_index = max(0, i - self.L + 1)
            decay_factors = np.exp(-self.tau * np.arange(i - start_index, -1, -1))
            weighted_sum = np.sum(X[start_index:i+1] * decay_factors[:, np.newaxis], axis=0)
            predictions[i] = weighted_sum

        return predictions