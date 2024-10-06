import numpy as np


class BaseModel:
    def __init__(self):
        self.L = 1  # Number of past time steps to consider
        pass


    def fit(self, X, y=None):
        pass  # No fitting process needed for SDModel

    def predict(self, X, indices):
        return X[indices-1]