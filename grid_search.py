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
                score = model_no_fit(data, model, threshold=300)
                results.append((tau, L, coef, score))
                print(f"tau: {tau}, L: {L}, coef: {coef}, MSE: {score}")

                if score < best_score:
                    best_score = score
                    best_params = (tau, L, coef)

    return best_params, best_score, results


def write_combined_results_to_file(sd_results, scd_results, baseline_score, filename='combined_results.txt', top_n=3):
    # Sort SD and SCD results by score
    sorted_sd_results = sorted(sd_results, key=lambda x: x[-1])
    sorted_scd_results = sorted(scd_results, key=lambda x: x[-1])

    with open(filename, 'w') as file:
        file.write(f"Baseline score: {baseline_score}\n")

        # Writing Top SDModel results
        file.write(f"\nTop {top_n} SDModel results:\n")
        for i in range(min(top_n, len(sorted_sd_results))):
            file.write(
                f"{i + 1} - tau: {sorted_sd_results[i][0]}, L: {sorted_sd_results[i][1]}, MSE: {sorted_sd_results[i][2]}\n")

        # Writing Top SCDModel results
        file.write(f"\nTop {top_n} SCDModel results:\n")
        for i in range(min(top_n, len(sorted_scd_results))):
            file.write(
                f"{i + 1} - tau: {sorted_scd_results[i][0]}, L: {sorted_scd_results[i][1]}, coef: {sorted_scd_results[i][2]}, MSE: {sorted_scd_results[i][3]}\n")



data, G_global = aggregate_to_matrix(get_socio_sms(), delta_t=1 * D)
