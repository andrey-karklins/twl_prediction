import numpy as np

import get_data
from BaseModel import BaseModel
from SCDModel import SCDModel
from SDModel import SDModel
from cross_fold_validation import temporal_cross_validation, model_no_fit
from get_data import *


def grid_search_sdmodel(data, taus, Ls):
    best_score = float('inf')
    best_params = None
    results = []

    for tau in taus:
        for L in Ls:
            model = SDModel(tau=tau, L=L)
            score = model_no_fit(data, model)
            results.append((tau, L, score))
            print(f"tau: {tau}, L: {L}, MSE: {score}")

            if score < best_score:
                best_score = score
                best_params = (tau, L)

    return best_params, best_score, results


def grid_search_scdmodel(data, taus, Ls, coefs, G_global):
    best_score = float('inf')
    best_params = None
    results = []

    for tau in taus:
        for L in Ls:
            for coef in coefs:
                model = SCDModel(tau=tau, L=L, alpha=coef[0], beta=coef[1], gamma=coef[2], G_global=G_global)
                score = model_no_fit(data, model)
                results.append((tau, L, coef, score))
                print(f"tau: {tau}, L: {L}, coef: {coef}, MSE: {score}")

                if score < best_score:
                    best_score = score
                    best_params = (tau, L, coef)

    return best_params, best_score, results


def print_top_sd_results(results, top_n=5):
    # Sort results by score
    sorted_results = sorted(results, key=lambda x: x[-1])
    print(f"Top {top_n} SDModel results:")
    for i in range(min(top_n, len(sorted_results))):
        print(f"{i + 1} - tau: {sorted_results[i][0]}, L: {sorted_results[i][1]}, MSE: {sorted_results[i][2]}")


def print_top_scd_results(results, top_n=5):
    # Sort results by score
    sorted_results = sorted(results, key=lambda x: x[-1])
    print(f"Top {top_n} SCDModel results:")
    for i in range(min(top_n, len(sorted_results))):
        print(f"{i + 1} - tau: {sorted_results[i][0]}, L: {sorted_results[i][1]}, coef: {sorted_results[i][2]}, MSE: {sorted_results[i][3]}")


data, G_global = aggregate_to_matrix(get_socio_sms(), delta_t=1 * D)

# Parameter grid
taus = [0.1, 0.5, 1, 3, 5]
Ls = [1, 3, 5, 10]
coefs = [(1, 0, 0), (0.8, 0.2, 0), (0.8, 0, 0.2), (0.8, 0.1, 0.1)]

# Baseline

# Grid search
_, _, sd_results = grid_search_sdmodel(data, taus, Ls)
_, _, scd_results = grid_search_scdmodel(data, taus, Ls, coefs, G_global)
print("---------------------------------------------------")
print("Baseline score: ", model_no_fit(data, BaseModel()))
print("---------------------------------------------------")
print_top_sd_results(sd_results)
print("---------------------------------------------------")
print_top_scd_results(scd_results)
