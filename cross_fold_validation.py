import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def temporal_cross_validation(data, model, n_splits=5, metric=mean_squared_error):
    # Transpose the data to (T, M) format
    data = data.T

    # TimeSeriesSplit for temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Placeholder for performance metrics
    scores = []

    # Temporal cross-validation loop
    for train_index, test_index in tscv.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = X_train[1:], X_test[1:]
        X_train, X_test = X_train[:-1], X_test[:-1]

        # Fit the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate the performance metric
        score = mean_squared_error(y_test, predictions)
        scores.append(score)

    # Calculate the average score
    average_score = np.mean(scores)

    return scores, average_score
