import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from cross_fold_validation import model_no_fit
from get_data import aggregate_to_matrix, get_hypertext, get_SFHH, get_college_1, get_college_2, get_socio_calls, \
    get_socio_sms
from grid_search import write_results_to_csv, grid_search_scdo_model, grid_search_scd_model, grid_search_sd_model
from models.BaseModel import BaseModel
from utils import load_or_fetch_dataset, H, M, D

# Parameter definitions
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.1, 0.5, 1, 3]
Ls = [1, 3, 5, 10]
coefs = [(0.8, 0.2, 0), (0.8, 0.2, 0), (0.8, 0.1, 0.1),
         (0.6, 0.4, 0), (0.6, 0.3, 0.1), (0.6, 0.2, 0.2), (0.6, 0.1, 0.3), (0.6, 0.4, 0),
         (0.4, 0.5, 0.1), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.4, 0.2, 0.4), (0.4, 0.1, 0.5),
         (0.2, 0.6, 0.2), (0.2, 0.4, 0.4), (0.2, 0.2, 0.6)]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_results(dataset, delta_t):
    data, G = aggregate_to_matrix(dataset, delta_t)

    # Run grid searches concurrently
    scdo_results = grid_search_scdo_model(data, taus, Ls, coefs, G)
    scd_results = grid_search_scd_model(data, taus, Ls, coefs, G)
    sd_results = grid_search_sd_model(data, taus, Ls, G)

    top_scdo = sorted(scdo_results, key=lambda x: x[-1]['MSE'])[0]
    top_scd = sorted(scd_results, key=lambda x: x[-1]['MSE'])[0]
    top_sd = sorted(sd_results, key=lambda x: x[-1]['MSE'])[0]
    baseline_score = model_no_fit(data, BaseModel())

    write_results_to_csv(top_sd, top_scd, top_scdo, baseline_score, delta_t, dataset.name,
                         filename='results/results.csv')


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

    # Combine all datasets and delta_t values into a single list of tasks
    tasks = [(dataset, delta_t) for dataset in datasets_physical for delta_t in delta_ts_physical] + \
            [(dataset, delta_t) for dataset in datasets_virtual for delta_t in delta_ts_virtual]

    # Run tasks concurrently
    with ProcessPoolExecutor(max_workers=9) as executor:  # Adjust max_workers as per CPU capacity
        futures = [executor.submit(generate_results, dataset, delta_t) for dataset, delta_t in tasks]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                future.result()  # Will raise an exception if the task failed
            except Exception as e:
                logging.error(f"An error occurred during grid search: {e}")


if __name__ == "__main__":
    main()
