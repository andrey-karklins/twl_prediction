import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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


def plot_lag_autocorrelation(datasets, max_lag=6):
    """
    Plots the lag vs. autocorrelation for each dataset with features.

    Args:
        datasets (list of np.array): List of 2D numpy arrays, each representing a dataset (NxT).
        max_lag (int): Maximum lag value to calculate autocorrelation for.
    """
    plt.figure(figsize=(10, 6))

    for i, (data, G) in enumerate(datasets):
        N, T = data.shape
        avg_autocorrelations = []

        for lag in range(max_lag + 1):
            autocorrelations = []
            for feature in range(N):
                if T - lag > 0:
                    autocorr = pearsonr(data[feature, :-lag], data[feature, lag:])[0] if lag != 0 else 1.0
                    autocorrelations.append(autocorr)
            avg_autocorrelations.append(np.mean(autocorrelations))

        plt.plot(range(max_lag + 1), avg_autocorrelations, label=G.name)

    plt.xlabel('Lag Δ')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lag_weighted_jaccard_similarity(datasets, max_lag=6):
    """
    Plots the lag vs. weighted Jaccard similarity for each dataset with features.

    Args:
        datasets (list of np.array): List of 2D numpy arrays, each representing a dataset (NxT).
        max_lag (int): Maximum lag value to calculate weighted Jaccard similarity for.
    """
    plt.figure(figsize=(10, 6))

    for i, (data, G) in enumerate(datasets):
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
            avg_jaccard_similarities.append(np.mean(jaccard_similarities))

        plt.plot(range(max_lag + 1), avg_jaccard_similarities, label=G.name)

    plt.xlabel('Lag Δ')
    plt.ylabel('Weighted Jaccard Similarity')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_autocorrelation_jaccard(datasets, delta_ts, max_lag=6, filename='combined_plots.png'):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()
    for i, delta_t in enumerate(delta_ts):
        for j, dataset in enumerate(datasets):
            data, G = aggregate_to_matrix(dataset, delta_t)
            # Autocorrelation Plot
            N, T = data.shape
            avg_autocorrelations = []

            for lag in range(max_lag + 1):
                autocorrelations = []
                for feature in range(N):
                    if T - lag > 0:
                        autocorr = pearsonr(data[feature, :-lag], data[feature, lag:])[0] if lag != 0 else 1.0
                        autocorrelations.append(autocorr)
                avg_autocorrelations.append(np.nanmean(autocorrelations))

            axs[i * 2].plot(range(max_lag + 1), avg_autocorrelations, label=G.name)
            axs[i * 2].set_title(f'Autocorrelation, Δt = {seconds_to_human_readable(delta_t)}')
            axs[i * 2].set_xlabel('Lag Δ')
            axs[i * 2].set_ylabel('Autocorrelation')
            axs[i * 2].legend()
            axs[i * 2].grid(True)

            # Weighted Jaccard Similarity Plot
            avg_jaccard_similarities = []

            for lag in range(max_lag + 1):
                jaccard_similarities = []
                for feature in range(N):
                    if T - lag > 0:
                        original_data = data[feature, :-lag] if lag != 0 else data[feature, :]
                        lagged_data = data[feature, lag:]
                        jaccard_similarity = weighted_jaccard_similarity(original_data, lagged_data)
                        jaccard_similarities.append(jaccard_similarity)
                avg_jaccard_similarities.append(np.mean(jaccard_similarities))

            axs[i * 2 + 1].plot(range(max_lag + 1), avg_jaccard_similarities, label=G.name)
            axs[i * 2 + 1].set_title(f'Weighted Jaccard Similarity, Δt = {seconds_to_human_readable(delta_t)}')
            axs[i * 2 + 1].set_xlabel('Lag Δ')
            axs[i * 2 + 1].set_ylabel('Weighted Jaccard Similarity')
            axs[i * 2 + 1].legend()
            axs[i * 2 + 1].grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
