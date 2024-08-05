import numpy as np

import get_data
from SDModel import SDModel
from cross_fold_validation import temporal_cross_validation
from get_data import *


def grid_search_sdmodel(data, taus, Ls, n_splits=5):
    best_score = float('inf')
    best_params = None
    results = []

    for tau in taus:
        for L in Ls:
            model = SDModel(tau=tau, L=L)
            _, average_score = temporal_cross_validation(data, model, n_splits=n_splits)
            results.append((tau, L, average_score))
            print(f"tau: {tau}, L: {L}, MSE: {average_score}")

            if average_score < best_score:
                best_score = average_score
                best_params = (tau, L)

    return best_params, best_score, results


data, _ = aggregate_to_matrix(get_socio_sms(), delta_t=1 * D)

# Parameter grid
taus = [0.1, 0.5, 1.0]
Ls = [3, 5, 10]

# Perform grid search
best_params, best_score, results = grid_search_sdmodel(data, taus, Ls)

# Output results
print("Best params:", best_params)
print("Best score:", best_score)
