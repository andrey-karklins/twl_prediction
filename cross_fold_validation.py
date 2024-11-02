
import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

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

    # Convert predictions and true values to binary (active/inactive) based on activity_threshold
    predicted_active = (predictions.ravel() >= activity_threshold).astype(int)
    true_active = (y_test.ravel() >= activity_threshold).astype(int)

    # Precision-Recall Curve and Area Under Precision-Recall Curve (AUPRC)
    precision, recall, _ = precision_recall_curve(true_active, predicted_active)
    auprc = auc(recall, precision)

    # Return all necessary metrics
    return {
        'MSE': mse,
        'AUPRC': auprc
    }
