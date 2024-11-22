import numpy as np


class SDModel:
    def __init__(self, tau):
        self.tau = tau  # Decay factor

    def predict(self, X):
        """
        Predict the future states of links based on the past states using the self-driven model.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (T, M) where T is the number of time steps and M is the number of links.

        Returns:
        numpy.ndarray: A 2D array of shape (T, M) containing the predicted states for each time step.
        """
        t = X.shape[0]
        # Calculate the time indices for the current window
        time_indices = np.arange(t)
        # Calculate the exponential decay factors
        decay_factors = np.exp(-self.tau * (t - time_indices))
        decay_factors /= decay_factors.sum()  # Normalize decay factors
        # Calculate the weighted sum for each feature (link) using the decay factors
        weighted_sum = np.dot(decay_factors, X[:t])
        return weighted_sum
