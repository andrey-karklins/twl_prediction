import csv
from models.SCDModel import SCDModel
from models.SCDOModel import SCDOModel
from models.SDModel import SDModel
from cross_fold_validation import model_no_fit
from get_data import *

from models.SCDModel import SCDModel
from models.SCDOModel import SCDOModel
from models.SDModel import SDModel
from cross_fold_validation import model_no_fit


def grid_search_sd_model(data, taus, Ls):
    best_score = float('inf')
    best_params = None
    results = []

    # Instantiate the SDModel only once
    model = SDModel(tau=taus[0], L=Ls[0])

    # Sequentially execute each parameter combination
    for tau in taus:
        for L in Ls:
            # Update model parameters instead of creating a new model each time
            model.tau = tau
            model.L = L

            # Run the model and evaluate its performance
            score = model_no_fit(data, model)
            results.append((tau, L, score))

            print(
                f"tau: {tau}, L: {L}, MSE: {score['MSE']}, MAE: {score['MAE']}, RMSE: {score['RMSE']}, AUPRC: {score['AUPRC']}")

            # Track the best score and parameters
            if score['MSE'] < best_score:
                best_score = score['MSE']
                best_params = (tau, L)

    return best_params, best_score, results


def grid_search_scd_model(data, taus, Ls, coefs, G_global):
    best_score = float('inf')
    best_params = None
    results = []

    # Instantiate the SCDModel only once
    model = SCDModel(tau=taus[0], L=Ls[0], alpha=coefs[0][0], beta=coefs[0][1], gamma=coefs[0][2], G_global=G_global)

    for tau in taus:
        for L in Ls:
            for coef in coefs:
                # Update model parameters instead of creating a new model each time
                model.tau = tau
                model.L = L
                model.alpha, model.beta, model.gamma = coef

                # Run the model and evaluate its performance
                score = model_no_fit(data, model, threshold=300)
                results.append((tau, L, coef, score))

                print(
                    f"tau: {tau}, L: {L}, coef: {coef}, MSE: {score['MSE']}, MAE: {score['MAE']}, RMSE: {score['RMSE']}, AUPRC: {score['AUPRC']}")

                if score['MSE'] < best_score:
                    best_score = score['MSE']
                    best_params = (tau, L, coef)

    return best_params, best_score, results


def grid_search_scdo_model(data, taus, Ls, coefs, G_global):
    best_score = float('inf')
    best_params = None
    results = []

    # Instantiate the SCDOModel only once
    model = SCDOModel(tau=taus[0], L=Ls[0], alpha=coefs[0][0], beta=coefs[0][1], gamma=coefs[0][2], G_global=G_global)

    for tau in taus:
        for L in Ls:
            for coef in coefs:
                # Update model parameters instead of creating a new model each time
                model.tau = tau
                model.L = L
                model.alpha, model.beta, model.gamma = coef

                # Run the model and evaluate its performance
                score = model_no_fit(data, model, threshold=300)
                results.append((tau, L, coef, score))

                print(
                    f"tau: {tau}, L: {L}, coef: {coef}, MSE: {score['MSE']}, MAE: {score['MAE']}, RMSE: {score['RMSE']}, AUPRC: {score['AUPRC']}")

                if score['MSE'] < best_score:
                    best_score = score['MSE']
                    best_params = (tau, L, coef)

    return best_params, best_score, results


def write_top1_results_to_file(sd_results, scd_results, scdo_results, baseline_scores, delta_ts, dataset_name,
                               filename='combined_results.txt'):
    with open(filename, 'w') as file:
        for (sd_res, scd_res, scd_orig_res, base_score, delta_t) in zip(sd_results, scd_results, scdo_results,
                                                                        baseline_scores, delta_ts):
            file.write(f"Dataset: {dataset_name}, Delta_t: {seconds_to_human_readable(delta_t)}\n")
            file.write(
                f"Baseline model - MSE: {base_score['MSE']}, MAE: {base_score['MAE']}, RMSE: {base_score['RMSE']}, AUPRC: {base_score['AUPRC']}\n")
            file.write(
                f"SDModel - MSE: {sd_res[2]['MSE']}, MAE: {sd_res[2]['MAE']}, RMSE: {sd_res[2]['RMSE']}, AUPRC: {sd_res[2]['AUPRC']} | tau: {sd_res[0]}, L: {sd_res[1]} |\n")
            file.write(
                f"SCDModel - MSE: {scd_res[3]['MSE']}, MAE: {scd_res[3]['MAE']}, RMSE: {scd_res[3]['RMSE']}, AUPRC: {scd_res[3]['AUPRC']} | tau: {scd_res[0]}, L: {scd_res[1]}, coef: {scd_res[2]} |\n")
            file.write(
                f"SCDOModel - MSE: {scdo_results[3]['MSE']}, MAE: {scdo_results[3]['MAE']}, RMSE: {scdo_results[3]['RMSE']}, AUPRC: {scdo_results[3]['AUPRC']} | tau: {scdo_results[0]}, L: {scdo_results[1]}, coef: {scdo_results[2]} |\n")
            file.write(f"-----------------------------------------------------------------------------------\n")


def write_results_to_csv(sd_results, scd_results, scd_orig_results, baseline_scores, delta_ts, dataset_name,
                         filename='results/results.csv'):
    header = [
        'Dataset name', 'delta_t', 'Baseline MSE', 'Baseline MAE', 'Baseline RMSE', 'Baseline AUPRC',
        'SDModel MSE', 'SDModel MAE', 'SDModel RMSE', 'SDModel AUPRC', 'SDModel tau', 'SDModel L',
        'SCDModel MSE', 'SCDModel MAE', 'SCDModel RMSE', 'SCDModel AUPRC', 'SCDModel tau', 'SCDModel L',
        'SCDModel coefs',
        'SCDOModel MSE', 'SCDOModel MAE', 'SCDOModel RMSE', 'SCDOModel AUPRC', 'SCDOModel tau', 'SCDOModel L',
        'SCDOModel coefs'
    ]

    try:
        file_is_empty = not os.path.exists(filename) or os.path.getsize(filename) == 0
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if file_is_empty:
                writer.writerow(header)
            for (sd_res, scd_res, scdo_results, base_score, delta_t) in zip(sd_results, scd_results, scd_orig_results,
                                                                            baseline_scores, delta_ts):
                writer.writerow([
                    dataset_name, delta_t,
                    base_score['MSE'], base_score['MAE'], base_score['RMSE'], base_score['AUPRC'],
                    sd_res[2]['MSE'], sd_res[2]['MAE'], sd_res[2]['RMSE'], sd_res[2]['AUPRC'],
                    sd_res[0], sd_res[1],
                    scd_res[3]['MSE'], scd_res[3]['MAE'], scd_res[3]['RMSE'], scd_res[3]['AUPRC'],
                    scd_res[0], scd_res[1], scd_res[2],
                    scdo_results[3]['MSE'], scdo_results[3]['MAE'], scdo_results[3]['RMSE'], scdo_results[3]['AUPRC'],
                    scdo_results[0], scdo_results[1], scdo_results[2]
                ])
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
