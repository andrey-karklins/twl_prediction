from models.SCDModel import SCDModel
from models.SDModel import SDModel
from cross_fold_validation import model_no_fit
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
                coef_sum = sum(coef)
                model = SCDModel(tau=tau, L=L, alpha=coef[0]/coef_sum, beta=coef[1]/coef_sum, gamma=coef[2]/coef_sum, G_global=G_global)
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

def write_top1_results_to_file(sd_results, scd_results, baseline_scores, delta_ts, dataset_name, filename='combined_results.txt'):
    with open(filename, 'w') as file:
        for (sd_res, scd_res, base_score, delta_t) in zip(sd_results, scd_results, baseline_scores, delta_ts):
            file.write(f"Dataset: {dataset_name}, Delta_t: {seconds_to_human_readable(delta_t)}\n")
            file.write(f"Baseline model - {base_score}\n")
            file.write(f"SDModel - {sd_res[2]} - tau: {sd_res[0]}, L: {sd_res[1]}\n")
            file.write(f"SCDModel - {scd_res[3]} - tau: {scd_res[0]}, L: {scd_res[1]}, coef: {scd_res[2]}\n")
            file.write(f"-----------------------------------------------------------------------------------\n")

def write_results_to_csv(sd_results, scd_results, baseline_scores, delta_ts, dataset_name, filename='results/results.csv'):
    with open(filename, 'a') as file:
        for (sd_res, scd_res, base_score, delta_t) in zip(sd_results, scd_results, baseline_scores, delta_ts):
            file.write(f"{dataset_name},{delta_t},{base_score},{sd_res[2]},{sd_res[0]},{sd_res[1]},{scd_res[3]},{scd_res[0]},{scd_res[1]},{scd_res[2]}\n")