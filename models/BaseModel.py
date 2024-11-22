class BaseModel:
    def predict(self, X, indices):
        return X[indices - 1]
