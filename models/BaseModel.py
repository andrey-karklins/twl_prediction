import numpy as np


class BaseModel:
    def __init__(self):
        pass


    def fit(self, X, y=None):
        pass  # No fitting process needed for SDModel

    def predict(self, X, indices):
        return X[indices]