import csv
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from cross_fold_validation import base_model_no_fit, model_fit
from get_data import aggregate_to_matrix, get_hypertext, get_SFHH, get_college_1, get_college_2, get_socio_calls, \
    get_socio_sms
from utils import load_or_fetch_dataset, H, M, D

# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.linear_model._coordinate_descent")

# Parameter definitions
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]
Ls = [10]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_completed_tasks():
    """
    Load completed tasks from a CSV file.
    Returns a set of tuples (dataset_name, delta_t, tau, L) for completed tasks.
    """
    if not os.path.exists('results/results.csv'):
        return set()

    completed_tasks = set()
    with open('results/results.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            completed_tasks.add((row["Dataset name"], int(row["delta_t"]), float(row["tau"]), float(row["L"])))
            logging.info(f"Loaded completed task: {row['Dataset name']} - delta_t: {row['delta_t']} - tau: {row['tau']} - L: {row['L']}")
    return completed_tasks


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
            logging.info(f"Writing to CSV: {dataset_name} - delta_t: {delta_t} - tau: {tau} - L: {L}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")


def search_task(dataset_name, data, delta_t, tau, L, G):
    """
    Perform a single search task and save the results.
    """
    try:
        base_score = base_model_no_fit(data, L)
        sd_score, scd_score, scdo_score = model_fit(data, tau, L, G)
        write_results_to_csv(
            dataset_name, delta_t, tau, L,
            sd_score, scd_score, scdo_score, base_score
        )
    except Exception as e:
        logging.error(f"Error processing {dataset_name} with delta_t: {delta_t}, tau: {tau}, L: {L} - {e}")


def main():
    # Load datasets
    datasets_physical = [
        load_or_fetch_dataset(get_hypertext, 'pickles/hypertext.pkl'),
        load_or_fetch_dataset(get_SFHH, 'pickles/SFHH.pkl')
    ]
    datasets_virtual = [
        load_or_fetch_dataset(get_college_1, 'pickles/college_1.pkl'),
        load_or_fetch_dataset(get_college_2, 'pickles/college_2.pkl'),
        load_or_fetch_dataset(get_socio_calls, 'pickles/socio_calls.pkl'),
        load_or_fetch_dataset(get_socio_sms, 'pickles/socio_sms.pkl')
    ]

    # Combine all datasets and delta_t values
    all_datasets = [(dataset, delta_t) for dataset in datasets_physical for delta_t in delta_ts_physical] + \
                   [(dataset, delta_t) for dataset in datasets_virtual for delta_t in delta_ts_virtual]

    completed_tasks = load_completed_tasks()

    tasks = []
    for dataset, delta_t in all_datasets:
        dataset_name = dataset.name
        data, G = aggregate_to_matrix(dataset, delta_t)

        for tau in taus:
            for L in Ls:
                if (dataset_name, delta_t, tau, L) not in completed_tasks:
                    tasks.append((dataset_name, data, delta_t, tau, L, G))

    # Use ProcessPoolExecutor to parallelize the tasks
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(search_task, *task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()  # Block until the future is completed
            except Exception as e:
                logging.error(f"Error occurred during task {task}: {e}")


if __name__ == '__main__':
    main()
