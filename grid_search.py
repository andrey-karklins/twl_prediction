import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from cross_fold_validation import model_fit, base_model_no_fit


def grid_search(data, taus, Ls, G_global, delta_t):
    def search_task(tau, L):
        base_score = base_model_no_fit(data, L)
        sd_score, scd_score, scdo_score = model_fit(data, tau, L, G_global)
        print(
            f"{G_global.name} | tau: {tau}, L: {L} | sd: {sd_score["MSE"]}, scd: {scd_score["MSE"]}, scdo: {scdo_score["MSE"]}")
        return tau, L, sd_score, scd_score, scdo_score, base_score

    # Run tasks sequentially
    # for tau in taus:
    #     for L in Ls:
    #         tau, L, sd_score, scd_score, scdo_score, base_score = search_task(tau, L)
    #         write_results_to_csv(G_global.name, delta_t, tau, L, sd_score, scd_score, scdo_score, base_score)

    # Run tasks concurrently
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(search_task, tau, L) for tau in taus for L in Ls]
        for future in as_completed(futures):
            tau, L, sd_score, scd_score, scdo_score, base_score = future.result()
            write_results_to_csv(G_global.name, delta_t, tau, L, sd_score, scd_score, scdo_score, base_score)


def write_results_to_csv(dataset_name, delta_t, tau, L, sd_res, scd_res, scdo_res, base_score,
                         filename='results/results.csv'):
    header = [
        'Dataset name', 'delta_t', 'tau', 'L', 'Baseline MSE', 'Baseline AUPRC', 'SDModel MSE', 'SDModel AUPRC',
        'SCDModel MSE', 'SCDModel AUPRC', 'SCDModel coefs', 'SCDOModel MSE', 'SCDOModel AUPRC', 'SCDOModel coefs'
    ]

    try:
        file_is_empty = not os.path.exists(filename) or os.path.getsize(filename) == 0
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if file_is_empty:
                writer.writerow(header)
            writer.writerow([
                dataset_name, delta_t, tau, L,
                base_score['MSE'], base_score['AUPRC'],
                sd_res['MSE'], sd_res['AUPRC'],
                scd_res['MSE'], scd_res['AUPRC'], scd_res['coefs'],
                scdo_res['MSE'], scdo_res['AUPRC'], scdo_res['coefs']
            ])
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
