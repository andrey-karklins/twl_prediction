import numpy as np


class SDModel:
    def __init__(self, tau, L):
        self.tau = tau  # Decay factor
        self.L = L  # Number of past time steps to consider

    def fit(self, X, y=None):
        # No fitting process needed for SDModel
        pass

    def predict(self, X, indices):
        """
        Predict the future states of links based on the past states using the self-driven model.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (T, M) where T is the number of time steps and M is the number of links.

        Returns:
        numpy.ndarray: A 2D array of shape (T, M) containing the predicted states for each time step.
        """
        T, M = X.shape
        predictions = np.zeros((len(indices), M))

        for i,t in enumerate(indices):
            # Determine the start of the window
            start_index = max(0, t - self.L)
            # Calculate the time indices for the current window
            time_indices = np.arange(start_index, t)
            # Calculate the exponential decay factors
            decay_factors = np.exp(-self.tau * (t - time_indices))
            decay_factors /= decay_factors.sum()  # Normalize decay factors
            # Calculate the weighted sum for each feature (link) using the decay factors
            weighted_sum = np.dot(decay_factors, X[start_index:t])
            predictions[i] = weighted_sum

        return predictions