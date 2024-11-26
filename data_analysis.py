import csv

# import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from get_data import aggregate_to_matrix
from utils import seconds_to_human_readable


def weighted_jaccard_similarity(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("The input arrays must have the same length.")

    min_sum = sum(min(a, b) for a, b in zip(array1, array2))
    max_sum = sum(max(a, b) for a, b in zip(array1, array2))

    if max_sum == 0:
        return 1.0 if min_sum == 0 else 0.0

    return min_sum / max_sum


def plot_lag_autocorrelation(datasets, delta_ts, max_lag=6):
    """
    Plots the lag vs. autocorrelation for each dataset with different aggregation times as lines.

    Args:
        datasets (list of tuples): List of tuples where each tuple contains (data, G),
                                   and G is a graph structure or dataset metadata.
        delta_ts (list of int): List of different aggregation times (Δt).
        max_lag (int): Maximum lag value to calculate autocorrelation for.
    """
    fig, axs = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)))

    if len(datasets) == 1:
        axs = [axs]  # Ensure axs is a list even for one dataset

    for idx, (dataset, G) in enumerate(datasets):
        ax = axs[idx]
        for delta_t in delta_ts:
            data, G = aggregate_to_matrix(dataset, delta_t)
            N, T = data.shape
            avg_autocorrelations = []

            for lag in range(max_lag + 1):
                autocorrelations = []
                for feature in range(N):
                    if T - lag > 0:
                        autocorr = pearsonr(data[feature, :-lag], data[feature, lag:])[0] if lag != 0 else 1.0
                        autocorrelations.append(autocorr)
                avg_autocorrelations.append(np.nanmean(autocorrelations))

            ax.plot(range(max_lag + 1), avg_autocorrelations, label=f'Δt = {seconds_to_human_readable(delta_t)}')

        ax.set_title(f'Autocorrelation for Dataset: {G.name}')
        ax.set_xlabel('Lag Δ')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_lag_weighted_jaccard_similarity(datasets, delta_ts, max_lag=6):
    """
    Plots the lag vs. weighted Jaccard similarity for each dataset with different aggregation times as lines.

    Args:
        datasets (list of tuples): List of tuples where each tuple contains (data, G),
                                   and G is a graph structure or dataset metadata.
        delta_ts (list of int): List of different aggregation times (Δt).
        max_lag (int): Maximum lag value to calculate weighted Jaccard similarity for.
    """
    fig, axs = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)))

    if len(datasets) == 1:
        axs = [axs]  # Ensure axs is a list even for one dataset

    for idx, (dataset, G) in enumerate(datasets):
        ax = axs[idx]
        for delta_t in delta_ts:
            data, G = aggregate_to_matrix(dataset, delta_t)
            N, T = data.shape
            avg_jaccard_similarities = []

            for lag in range(max_lag + 1):
                jaccard_similarities = []
                for feature in range(N):
                    if T - lag > 0:
                        original_data = data[feature, :-lag] if lag != 0 else data[feature, :]
                        lagged_data = data[feature, lag:]
                        jaccard_similarity = weighted_jaccard_similarity(original_data, lagged_data)
                        jaccard_similarities.append(jaccard_similarity)
                avg_jaccard_similarities.append(np.nanmean(jaccard_similarities))

            ax.plot(range(max_lag + 1), avg_jaccard_similarities, label=f'Δt = {seconds_to_human_readable(delta_t)}')

        ax.set_title(f'Weighted Jaccard Similarity for Dataset: {G.name}')
        ax.set_xlabel('Lag Δ')
        ax.set_ylabel('Weighted Jaccard Similarity')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_autocorrelation_jaccard(datasets, delta_ts, max_lag=6, filename='combined_plots.png'):
    """
    Combines both autocorrelation and weighted Jaccard similarity plots for each dataset,
    with different aggregation times as colored lines, arranged side by side (2 columns).

    Args:
        datasets (list of tuples): List of datasets with their corresponding metadata.
        delta_ts (list of int): Different aggregation times to plot.
        max_lag (int): Maximum lag to calculate.
        filename (str): File name to save the plot image.
    """
    fig, axs = plt.subplots(len(datasets), 2, figsize=(14, 6 * len(datasets)))

    if len(datasets) == 1:
        axs = [axs]  # Ensure axs is a list even for one dataset

    for idx, dataset in enumerate(datasets):
        # Autocorrelation Plot
        ax_autocorr = axs[idx][0]
        for delta_t in delta_ts:
            data, G = aggregate_to_matrix(dataset, delta_t)
            N, T = data.shape
            avg_autocorrelations = []

            for lag in range(max_lag + 1):
                autocorrelations = []
                for feature in range(N):
                    if T - lag > 0:
                        autocorr = pearsonr(data[feature, :-lag], data[feature, lag:])[0] if lag != 0 else 1.0
                        autocorrelations.append(autocorr)
                avg_autocorrelations.append(np.nanmean(autocorrelations))

            ax_autocorr.plot(range(max_lag + 1), avg_autocorrelations,
                             label=f'Δt = {seconds_to_human_readable(delta_t)}')

        ax_autocorr.set_title(f'Autocorrelation for Dataset: {G.name}')
        ax_autocorr.set_xlabel('Lag Δ')
        ax_autocorr.set_ylabel('Autocorrelation')
        ax_autocorr.legend()
        ax_autocorr.grid(True)

        # Weighted Jaccard Similarity Plot
        ax_jaccard = axs[idx][1]
        for delta_t in delta_ts:
            data, G = aggregate_to_matrix(dataset, delta_t)
            N, T = data.shape
            avg_jaccard_similarities = []

            for lag in range(max_lag + 1):
                jaccard_similarities = []
                for feature in range(N):
                    if T - lag > 0:
                        original_data = data[feature, :-lag] if lag != 0 else data[feature, :]
                        lagged_data = data[feature, lag:]
                        jaccard_similarity = weighted_jaccard_similarity(original_data, lagged_data)
                        jaccard_similarities.append(jaccard_similarity)
                avg_jaccard_similarities.append(np.nanmean(jaccard_similarities))

            ax_jaccard.plot(range(max_lag + 1), avg_jaccard_similarities,
                            label=f'Δt = {seconds_to_human_readable(delta_t)}')

        ax_jaccard.set_title(f'Weighted Jaccard Similarity for Dataset: {G.name}')
        ax_jaccard.set_xlabel('Lag Δ')
        ax_jaccard.set_ylabel('Weighted Jaccard Similarity')
        ax_jaccard.legend()
        ax_jaccard.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def save_autocorrelation_jaccard_to_csv(datasets, delta_ts, max_lag=6, filename='autocorrelation_jaccard_results.csv'):
    """
    Saves both autocorrelation and weighted Jaccard similarity for each dataset into a CSV file,
    with different aggregation times as rows.

    Args:
        datasets (list of tuples): List of datasets with their corresponding metadata.
        delta_ts (list of int): Different aggregation times to calculate.
        max_lag (int): Maximum lag to calculate.
        filename (str): File name to save the CSV.
    """

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Aggregation Time', 'Metric', 'Lag', 'Value'])

        for idx, dataset in enumerate(datasets):
            dataset_name = dataset.name

            # Autocorrelation values
            for delta_t in delta_ts:
                data, G = aggregate_to_matrix(dataset, delta_t)
                N, T = data.shape

                # Calculate and save autocorrelation values
                for lag in range(max_lag + 1):
                    autocorrelations = []
                    for feature in range(N):
                        if T - lag > 0:
                            autocorr = pearsonr(data[feature, :-lag], data[feature, lag:])[0] if lag != 0 else 1.0
                            autocorrelations.append(autocorr)
                    avg_autocorrelation = np.nanmean(autocorrelations)
                    writer.writerow(
                        [dataset_name, seconds_to_human_readable(delta_t), 'Autocorrelation', lag, avg_autocorrelation])

            # Weighted Jaccard similarity values
            for delta_t in delta_ts:
                data, G = aggregate_to_matrix(dataset, delta_t)
                N, T = data.shape

                # Calculate and save Jaccard similarity values
                for lag in range(max_lag + 1):
                    jaccard_similarities = []
                    for feature in range(N):
                        if T - lag > 0:
                            original_data = data[feature, :-lag] if lag != 0 else data[feature, :]
                            lagged_data = data[feature, lag:]
                            jaccard_similarity = weighted_jaccard_similarity(original_data, lagged_data)
                            jaccard_similarities.append(jaccard_similarity)
                    avg_jaccard_similarity = np.nanmean(jaccard_similarities)
                    writer.writerow(
                        [dataset_name, seconds_to_human_readable(delta_t), 'Weighted Jaccard Similarity', lag,
                         avg_jaccard_similarity])


def apply_fourier_transform(matrices, delta_ts, dataset_name, filename="fourier_transform.png"):
    """
    Apply Fourier Transform to each row (time series of each edge) in the MxT matrices.

    Parameters:
    matrices (list of numpy.ndarray): List of MxT matrices, where M is the number of edges and T is the number of timestamps.
    delta_ts (list of int): List of aggregation times corresponding to each matrix.
    dataset_name (str): Name of the dataset for plot title.
    filename (str): The filename for saving the plot.

    Returns:
    list of numpy.ndarray: List of averaged magnitudes for each input matrix.
    """
    num_matrices = len(matrices)
    fig, axs = plt.subplots(num_matrices, 1, figsize=(12, 6 * num_matrices))  # Create a column of subplots

    for i, (matrix, delta_t) in enumerate(zip(matrices, delta_ts)):
        # Apply the Fourier Transform to each row (axis=1 applies it to the time dimension)
        fourier_matrix = np.fft.fft(matrix, axis=1)
        magnitude = np.abs(fourier_matrix)

        # Averaging over all edges (rows)
        avg_magnitude = np.mean(magnitude, axis=0)

        # Get the number of timestamps (T)
        T = fourier_matrix.shape[1]

        # Frequency axis
        frequencies = np.fft.fftfreq(T)

        # Plot the averaged Magnitude Spectrum in the respective subplot
        axs[i].stem(frequencies, avg_magnitude, markerfmt='.', basefmt=" ")
        axs[i].set_title(f'Averaged Magnitude Spectrum - Aggregation time: {seconds_to_human_readable(delta_t)}',
                         fontsize=10)
        axs[i].set_xlabel('Frequency')
        axs[i].set_ylabel('Magnitude')
        axs[i].grid(True)

    # Overall figure title and layout adjustments
    fig.suptitle(f'Magnitude Spectrum Analysis for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the suptitle
    plt.savefig(filename)
    plt.show()

    return [np.mean(np.abs(np.fft.fft(matrix, axis=1)), axis=0) for matrix in matrices]
