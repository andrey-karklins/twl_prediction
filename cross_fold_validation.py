
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, auc


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


def model_no_fit(data, model, threshold=0, activity_threshold=1):
    # Transpose the data to (T, M) format (time steps first)
    data = data.T
    T = data.shape[0]
    L = model.L  # Assuming the model has an attribute 'L' that defines the sliding window length

    # Number of predictions to make
    num_predictions = T - L - 1
    np.random.seed(42)
    # Select data indices for prediction based on threshold
    if threshold != 0 and num_predictions > threshold:
        indices = np.random.randint(L, T, threshold)
    else:
        indices = np.arange(L, T, dtype=int)

    # Make predictions using the model
    predictions = model.predict(data[:-1], indices)
    y_test = data[indices]

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Convert predictions and true values to binary (active/inactive) based on activity_threshold
    predicted_active = (predictions.ravel() >= activity_threshold).astype(int)
    true_active = (y_test.ravel() >= activity_threshold).astype(int)

    # Precision-Recall Curve and Area Under Precision-Recall Curve (AUPRC)
    precision, recall, _ = precision_recall_curve(true_active, predicted_active)
    auprc = auc(recall, precision)

    # Return all necessary metrics
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'AUPRC': auprc
    }
