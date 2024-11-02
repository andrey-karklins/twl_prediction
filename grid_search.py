import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.SCDModel import SCDModel
from models.SCDOModel import SCDOModel
from models.SDModel import SDModel
from cross_fold_validation import model_no_fit
from get_data import *


def grid_search_sd_model(data, taus, Ls, G_global):
    results = []
    model = SDModel(tau=taus[0], L=Ls[0])

    for tau in taus:
        for L in Ls:
            model.tau = tau
            model.L = L
            score = model_no_fit(data, model)
            print(f"SDModel | {G_global.name} | tau={tau}, L={L}")
            results.append((tau, L, score))

    return results


def grid_search_scd_model(data, taus, Ls, coefs, G_global):
    results = []

    def search_task(tau, L, coef):
        model = SCDModel(tau=tau, L=L, alpha=coef[0], beta=coef[1], gamma=coef[2], G_global=G_global)
        score = model_no_fit(data, model, threshold=300)
        return tau, L, coef, score

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(search_task, tau, L, coef) for tau in taus for L in Ls for coef in coefs]
        for future in as_completed(futures):
            tau, L, coef, score = future.result()
            print(f"SCDModel | {G_global.name} | tau={tau}, L={L}, coef={coef}")
            results.append((tau, L, coef, score))

    return results


def grid_search_scdo_model(data, taus, Ls, coefs, G_global):
    results = []

    def search_task(tau, L, coef):
        model = SCDOModel(tau=tau, L=L, alpha=coef[0], beta=coef[1], gamma=coef[2], G_global=G_global)
        score = model_no_fit(data, model, threshold=300)
        return tau, L, coef, score

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(search_task, tau, L, coef) for tau in taus for L in Ls for coef in coefs]
        for future in as_completed(futures):
            tau, L, coef, score = future.result()
            print(f"SCDOModel | {G_global.name} | tau={tau}, L={L}, coef={coef}")
            results.append((tau, L, coef, score))

    return results


def write_results_to_csv(sd_res, scd_res, scdo_res, base_score, delta_t, dataset_name, filename='results/results.csv'):
    header = [
        'Dataset name', 'delta_t', 'Baseline MSE', 'Baseline AUPRC',
        'SDModel MSE', 'SDModel AUPRC', 'SDModel tau', 'SDModel L',
        'SCDModel MSE', 'SCDModel AUPRC', 'SCDModel tau', 'SCDModel L', 'SCDModel coefs',
        'SCDOModel MSE', 'SCDOModel AUPRC', 'SCDOModel tau', 'SCDOModel L', 'SCDOModel coefs'
    ]

    try:
        file_is_empty = not os.path.exists(filename) or os.path.getsize(filename) == 0
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if file_is_empty:
                writer.writerow(header)
            writer.writerow([
                dataset_name, delta_t,
                base_score['MSE'], base_score['AUPRC'],
                sd_res[2]['MSE'], sd_res[2]['AUPRC'], sd_res[0], sd_res[1],
                scd_res[3]['MSE'], scd_res[3]['AUPRC'], scd_res[0], scd_res[1], scd_res[2],
                scdo_res[3]['MSE'], scdo_res[3]['AUPRC'], scdo_res[0], scdo_res[1], scdo_res[2]
            ])
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
