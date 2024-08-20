import numpy as np


class SCDModel:
    def __init__(self, tau, L, alpha, beta, gamma, G_global):
        self.tau = tau  # Decay factor
        self.L = L  # Number of past time steps to consider
        self.alpha = alpha  # Weight for self-driven component
        self.beta = beta  # Weight for neighbor-driven component
        self.gamma = gamma  # Weight for common neighbor-driven component
        self.G_global = G_global  # The global graph structure

    def fit(self, X, y=None):
        # No fitting process needed for SCDModel
        pass

    def predict(self, X):
        """
        Predict the future states of links based on the past states using the self- and cross-driven model.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (T, M) where T is the number of time steps and M is the number of links.

        Returns:
        numpy.ndarray: A 2D array of shape (T, M) containing the predicted states for each time step.
        """
        T, M = X.shape
        predictions = np.zeros((T, M))

        for t in range(T):
            # Determine the start of the window
            start_index = max(0, t - self.L + 1)
            # Calculate the time indices for the current window
            time_indices = np.arange(start_index, t + 1)
            # Calculate the exponential decay factors
            decay_factors = np.exp(-self.tau * (t - time_indices))
            decay_factors /= decay_factors.sum()  # Normalize decay factors

            for j in range(M):
                # Identify the edge corresponding to feature j
                edge = list(self.G_global.edges())[j]

                # Self-driven component (influence of the past states of the same link)
                weighted_sum_self = np.dot(decay_factors, X[start_index:t + 1, j])
                self_driven = self.alpha * weighted_sum_self

                # Neighbor-driven component (influence of neighboring links that share a node with the target link)
                neighbor_sum = 0
                for neighbor in self.G_global.neighbors(edge[0]):
                    if neighbor != edge[1] and (edge[0], neighbor) in self.G_global.edges:
                        neighbor_edge_id = self.G_global.edges[edge[0], neighbor]["id"]
                        neighbor_sum += np.dot(decay_factors, X[start_index:t + 1, neighbor_edge_id])
                for neighbor in self.G_global.neighbors(edge[1]):
                    if neighbor != edge[0] and (edge[1], neighbor) in self.G_global.edges:
                        neighbor_edge_id = self.G_global.edges[edge[1], neighbor]["id"]
                        neighbor_sum += np.dot(decay_factors, X[start_index:t + 1, neighbor_edge_id])
                neighbor_driven = self.beta * neighbor_sum

                # Common neighbor-driven component (influence of links that form a triangle with the target link)
                common_neighbor_sum = 0
                common_neighbors = set(self.G_global.neighbors(edge[0])) & set(self.G_global.neighbors(edge[1]))
                for common_neighbor in common_neighbors:
                    if (edge[0], common_neighbor) in self.G_global.edges:
                        common_neighbor_edge_id = self.G_global.edges[edge[0], common_neighbor]["id"]
                        common_neighbor_sum += np.dot(decay_factors, X[start_index:t + 1, common_neighbor_edge_id])
                    if (edge[1], common_neighbor) in self.G_global.edges:
                        common_neighbor_edge_id = self.G_global.edges[edge[1], common_neighbor]["id"]
                        common_neighbor_sum += np.dot(decay_factors, X[start_index:t + 1, common_neighbor_edge_id])
                common_neighbor_driven = self.gamma * common_neighbor_sum

                # Total prediction for the current link at time t
                predictions[t, j] = self_driven + neighbor_driven + common_neighbor_driven

        return predictions
