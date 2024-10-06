import csv

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
            print(f"tau: {tau}, L: {L}, MSE: {score['MSE']}, MAE: {score['MAE']}, RMSE: {score['RMSE']}, AUPRC: {score['AUPRC']}")

            if score['MSE'] < best_score:
                best_score = score['MSE']
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
                print(f"tau: {tau}, L: {L}, coef: {coef}, MSE: {score['MSE']}, MAE: {score['MAE']}, RMSE: {score['RMSE']}, AUPRC: {score['AUPRC']}")
                if score['MSE'] < best_score:
                    best_score = score['MSE']
                    best_params = (tau, L, coef)
    return best_params, best_score, results

def write_top1_results_to_file(sd_results, scd_results, baseline_scores, delta_ts, dataset_name, filename='combined_results.txt'):
    with open(filename, 'w') as file:
        for (sd_res, scd_res, base_score, delta_t) in zip(sd_results, scd_results, baseline_scores, delta_ts):
            file.write(f"Dataset: {dataset_name}, Delta_t: {seconds_to_human_readable(delta_t)}\n")
            file.write(f"Baseline model - MSE: {base_score['MSE']}, MAE: {base_score['MAE']}, RMSE: {base_score['RMSE']}, AUPRC: {base_score['AUPRC']}\n")
            file.write(f"SDModel - MSE: {sd_res[2]['MSE']}, MAE: {sd_res[2]['MAE']}, RMSE: {sd_res[2]['RMSE']}, AUPRC: {sd_res[2]['AUPRC']} | tau: {sd_res[0]}, L: {sd_res[1]} |\n")
            file.write(f"SCDModel - MSE: {scd_res[3]['MSE']}, MAE: {scd_res[3]['MAE']}, RMSE: {scd_res[3]['RMSE']}, AUPRC: {scd_res[3]['AUPRC']} | tau: {scd_res[0]}, L: {scd_res[1]}, coef: {scd_res[2]} |\n")
            file.write(f"-----------------------------------------------------------------------------------\n")



def write_results_to_csv(sd_results, scd_results, baseline_scores, delta_ts, dataset_name, filename='results/results.csv'):
    # Define the header for the CSV file
    header = [
        'Dataset name', 'delta_t', 'Baseline MSE', 'Baseline MAE', 'Baseline RMSE', 'Baseline AUPRC',
        'SDModel MSE', 'SDModel MAE', 'SDModel RMSE', 'SDModel AUPRC', 'SDModel tau', 'SDModel L',
        'SCDModel MSE', 'SCDModel MAE', 'SCDModel RMSE', 'SCDModel AUPRC', 'SCDModel tau', 'SCDModel L', 'SCDModel coefs'
    ]

    # First check if file exists and is empty
    try:
        file_is_empty = False
        try:
            with open(filename, 'r') as file:
                if file.read(1) == '':
                    file_is_empty = True
        except FileNotFoundError:
            file_is_empty = True

        # Open the file for appending or writing
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)

            # If the file is empty or new, write the header
            if file_is_empty:
                writer.writerow(header)
                print(f"Header written to {filename}")

            # Write each result set to the CSV file
            for (sd_res, scd_res, base_score, delta_t) in zip(sd_results, scd_results, baseline_scores, delta_ts):
                print(f"Writing results for dataset {dataset_name} with delta_t {delta_t}")
                writer.writerow([
                    dataset_name,
                    delta_t,
                    base_score['MSE'], base_score['MAE'], base_score['RMSE'], base_score['AUPRC'],  # Baseline metrics
                    sd_res[2]['MSE'], sd_res[2]['MAE'], sd_res[2]['RMSE'], sd_res[2]['AUPRC'],  # SDModel metrics
                    sd_res[0],  # SDModel tau
                    sd_res[1],  # SDModel L
                    scd_res[3]['MSE'], scd_res[3]['MAE'], scd_res[3]['RMSE'], scd_res[3]['AUPRC'],  # SCDModel metrics
                    scd_res[0],  # SCDModel tau
                    scd_res[1],  # SCDModel L
                    scd_res[2]  # SCDModel coefficients (alpha, beta, gamma)
                ])
            print(f"Results written to {filename}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")