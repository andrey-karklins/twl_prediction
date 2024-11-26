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
def model_fit(data, tau, L, G_global, threshold=300):
    """
    Fit and evaluate the SDModel, SCDModel, and SCDOModel on the temporal weighted network.

    Args:
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
    # Determine prediction indices
    indices = find_indices(T, L, threshold)

    # Initialize SDModel, SCDModel, and SCDOModel
    sd_model = SDModel(tau, L)  # SDModel uses decay factor tau
    scd_model = SCDModel(G_global)  # SCDModel uses global parameter G_global
    scdo_model = SCDOModel(G_global)  # SCDOModel uses global parameter G_global

    # Initialize arrays to store predictions and coefficients
    sd_predictions = sd_model.predict(data, indices)
    sd_score = calculate_metrics(sd_predictions, data[indices])

    # SCDModel
    scd_model.fit(sd_predictions, data[indices])
    scd_predictions = scd_model.predict(sd_predictions)
    scd_score = calculate_metrics(scd_predictions, data[indices])
    scd_score['coefs'] = scd_model.betas

    # SCDOModel
    scdo_model.fit(sd_predictions, data[indices])
    scdo_predictions = scdo_model.predict(sd_predictions)
    scdo_score = calculate_metrics(scdo_predictions, data[indices])
    scdo_score['coefs'] = scdo_model.betas

    # Return performance metrics for all models
    return sd_score, scd_score, scdo_score
