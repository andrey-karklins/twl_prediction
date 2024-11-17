
import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

def model_no_fit(data, model, threshold=100, activity_threshold=1):
    # Transpose the data to (T, M) format (time steps first)
    data = data.T
    T = data.shape[0]
    L = model.L  # Assuming the model has an attribute 'L' that defines the sliding window length

    # Number of predictions to make
    num_predictions = T - L - 1
    np.random.seed(42)

    # If too many predictions are requested, sample equally spaced indices
    if threshold != 0 and num_predictions > 100:
        indices = np.linspace(L, T - 1, threshold).astype(int)
    else:
        indices = np.arange(L, T, dtype=int)

    avg_coefs = np.zeros(4)
    predictions = np.zeros(indices, data.shape[1])
    for i in indices:
        # Get the input data for the current prediction
        X = data[i - L:i]

        model.fit(X)
        # Make a prediction using the model
        prediction = model.predict(X)

        # Update the data with the prediction
        data[i] = prediction
        avg_coefs[0] += model.beta0
        avg_coefs[1:4] += model.coefficients_

    y_test = data[indices]

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)

    # Convert predictions and true values to binary (active/inactive) based on activity_threshold
    predicted_active = (predictions.ravel() >= activity_threshold).astype(int)
    true_active = (y_test.ravel() >= activity_threshold).astype(int)

    # Precision-Recall Curve and Area Under Precision-Recall Curve (AUPRC)
    precision, recall, _ = precision_recall_curve(true_active, predicted_active)
    auprc = auc(recall, precision)

    # Return all necessary metrics
    return {
        "coefs": avg_coefs / len(indices),
        'MSE': mse,
        'AUPRC': auprc
    }
