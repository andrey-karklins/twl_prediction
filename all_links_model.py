from statistics import mean

import numpy as np
from sklearn import linear_model, __all__, tree

import get_data
from utils import *
from get_data import *


# Assuming W is your weight matrix with shape [M, T]
# Generate sample data

W, G = aggregate_to_matrix(get_data.get_hypertext(), 20 * M)
m_len = W.shape[0]
t_len = W.shape[1]

# Train a linear regression model for each edge
models = []
t_train = round(t_len * 0.8)
t_test = t_len - t_train
X_train = W.T[:t_train]
X_test = W.T[t_train:-1]
Y_train = W[:, 1:t_train+1]
Y_test = W[:, 1+t_train:]
for i in range(m_len):
    if (i + 1) % 100 == 0:
        print(f"Training model {i + 1}/{m_len}")
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, Y_train[i])
    models.append(model)

scores = [model.score(X_test, Y_test[i]) for i, model in enumerate(models)]
print(scores)
