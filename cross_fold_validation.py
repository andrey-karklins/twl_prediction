# Import necessary libraries
import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc

# Import base and specialized models from their respective modules
from models.BaseModel import BaseModel
from models.SCDModel import SCDModel
from models.SCDOModel import SCDOModel
from models.SDModel import SDModel


# Function to calculate performance metrics
def calculate_metrics(predictions, y_test, activity_threshold=1):
    """
    Calculate performance metrics for model evaluation.

    Args:
        predictions: Predicted values (e.g., edge weights over time).
        activity_threshold: Threshold for classifying edges as active/inactive.
        y_test: True values (actual edge weights).

    Returns:
        Dictionary containing:
            - Mean Squared Error (MSE)
            - Area Under Precision-Recall Curve (AUPRC)
    """
    # Compute Mean Squared Error (MSE) between predictions and true values
    mse = mean_squared_error(y_test, predictions)

    # Binarize predictions and true values based on the activity_threshold
    predicted_active = (predictions.ravel() >= activity_threshold).astype(int)
    true_active = (y_test.ravel() >= activity_threshold).astype(int)

    # Compute precision-recall curve and Area Under Precision-Recall Curve (AUPRC)
    precision, recall, _ = precision_recall_curve(true_active, predicted_active)
    auprc = auc(recall, precision)

    # Return computed metrics
    return {
        'MSE': mse,
        'AUPRC': auprc
    }


# Function to determine indices for prediction sampling
def find_indices(T, L, threshold):
    """
    Determine indices for prediction time steps based on observation window and sampling threshold.

    Args:
        T: Total number of time steps.
        L: Observation window (number of previous timestamps used for prediction).
        threshold: Maximum number of time steps to evaluate (0 for all).

    Returns:
        Array of indices representing prediction time steps.
    """
    num_predictions = T - L - 1

    # If number of predictions exceeds threshold, sample evenly spaced indices
    if threshold != 0 and num_predictions > threshold:
        indices = np.linspace(L, T - 1, threshold).astype(int)
    else:
        # Otherwise, use all indices from L to T
        indices = np.arange(L, T, dtype=int)

    return indices


# Function to evaluate the base model (reference solution)
def base_model_no_fit(data, L, threshold=100):
    """
    Evaluate the BaseModel without fitting as a reference solution.

    Args:
        data: Input data matrix (MxT), representing temporal weighted network.
        L: Observation window (number of previous timestamps used for prediction).
        threshold: Maximum number of time steps to evaluate.

    Returns:
        Dictionary of performance metrics for the BaseModel.
    """
    # Transpose data to (T, M) format for processing (time steps first)
    data = data.T
    T = data.shape[0]
    L = int(T * L)

    # Determine prediction indices
    indices = find_indices(T, L, threshold)

    # Instantiate the BaseModel and make predictions
    model = BaseModel()
    predictions = model.predict(data, indices)

    # Return performance metrics for the BaseModel
    return calculate_metrics(predictions, data[indices])


# Function to evaluate SDModel and SCDModel
def model_fit(data, tau, L, G_global, train_window_rate=0.1, threshold=300):
    """
    Fit and evaluate the SDModel, SCDModel, and SCDOModel on the temporal weighted network.

    Args:
        train_window_rate: Percentage of the data to use as the training window within L.
        data: Input data matrix (MxT), representing temporal weighted network.
        tau: Decay factor for SDModel.
        L: Observation window (number of previous timestamps used for prediction).
        G_global: Global parameter for SCDModel and SCDOModel.
        threshold: Maximum number of time steps to evaluate.

    Returns:
        Tuple containing:
            - Performance metrics for SDModel
            - Performance metrics for SCDModel
            - Performance metrics for SCDOModel
    """
    # Transpose data to (T, M) format (time steps first)
    data = data.T
    T = data.shape[0]
    L = int(T * L)
    M = data.shape[1]
    train_window = min(10, max(1, int(L * train_window_rate)))

    # Determine prediction indices
    indices = find_indices(T, L, threshold)

    # Initialize SDModel, SCDModel, and SCDOModel
    sd_model = SDModel(tau)  # SDModel uses decay factor tau
    scd_model = SCDModel(G_global)  # SCDModel uses global parameter G_global
    scdo_model = SCDOModel(G_global)  # SCDOModel uses global parameter G_global

    # Initialize arrays to store predictions and coefficients
    sd_predictions = np.zeros((len(indices), M))  # Predictions from SDModel
    scd_predictions = np.zeros((len(indices), M))  # Predictions from SCDModel
    scdo_predictions = np.zeros((len(indices), M))  # Predictions from SCDOModel
    scd_beta_sum = np.zeros(4)  # Sum of coefficients (betas) from SCDModel
    scdo_beta_sum = np.zeros(4)  # Sum of coefficients (betas) from SCDOModel

    sd_predictions_cache = {}
    # Loop through prediction indices
    for j, i in enumerate(indices):
        # Predict current state using SDModel
        sd_predictions[j] = sd_model.predict(data[i - L:i])
        sd_fit_predictions = np.zeros((train_window, M))
        sd_fit_y = np.zeros((train_window, M))
        for k in range(train_window):
            if (i - k - 1 in sd_predictions_cache):
                sd_fit_predictions[k] = sd_predictions_cache[i - k - 1]
            else:
                sd_fit_predictions[k] = sd_model.predict(data[i - L - 1:i - k - 1])
                sd_predictions_cache[i - k - 1] = sd_fit_predictions[k]
            sd_fit_y[k] = data[i - k - 1]

        # Fit and predict using SCDModel
        scd_model.fit(sd_fit_predictions, sd_fit_y)
        scd_predictions[j] = scd_model.predict(sd_predictions[j])
        scd_beta_sum += scd_model.betas
        if not scd_model.has_converged:
            print(f"SCDModel did not converge")

        # Fit and predict using SCDOModel
        scdo_model.fit(sd_fit_predictions, sd_fit_y)
        scdo_predictions[j] = scdo_model.predict(sd_predictions[j])
        scdo_beta_sum += scdo_model.betas
        if not scdo_model.has_converged:
            print(f"SCDOModel did not converge")

    # Calculate metrics for SDModel predictions
    sd_score = calculate_metrics(sd_predictions, data[indices])

    # Calculate metrics for SCDModel predictions
    scd_score = calculate_metrics(scd_predictions, data[indices])
    scd_score['coefs'] = scd_beta_sum / len(indices)  # Average coefficients for SCDModel

    # Calculate metrics for SCDOModel predictions
    scdo_score = calculate_metrics(scdo_predictions, data[indices])
    scdo_score['coefs'] = scdo_beta_sum / len(indices)  # Average coefficients for SCDOModel

    # Return performance metrics for all models
    return sd_score, scd_score, scdo_score
