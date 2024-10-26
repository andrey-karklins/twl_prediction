import pickle

from grid_search import *
import concurrent.futures
import logging

from models.BaseModel import BaseModel

delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.1, 0.5, 1, 3, 5]
Ls = [1, 3, 5, 10]
coefs = [(0.8, 0.1, 0.1), (0.8, 0.2, 0), (0.8, 0, 0.2), (0.6, 0.2, 0.2), (0.6, 0.3, 0.1), (0.6, 0.1, 0.3)]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming get_hypertext, get_SFHH, etc., are already defined functions
def generate_results(dataset, delta_ts, delta_ts_label):
    top_sd_results = []
    top_scd_results = []
    top_scdo_results = []
    base_scores = []
    for delta_t in delta_ts:
        data, G = aggregate_to_matrix(dataset, delta_t)
        baseline_score = model_no_fit(data, BaseModel())
        # Grid search
        _, _, sd_results = grid_search_sd_model(data, taus, Ls)
        _, _, scd_results = grid_search_scd_model(data, taus, Ls, coefs, G)
        _, _, scdo_results = grid_search_scdo_model(data, taus, Ls, coefs, G)

        base_scores.append(baseline_score)
        top_sd_results.append(sorted(sd_results, key=lambda x: x[-1]['MSE'])[0])
        top_scd_results.append(sorted(scd_results, key=lambda x: x[-1]['MSE'])[0])
        top_scdo_results.append(sorted(scdo_results, key=lambda x: x[-1]['MSE'])[0])

    # apply_fourier_transform(tmp_datasets, tmp_delta_ts, dataset.name, filename=f'results/plots/fourier_transform_{dataset.name}.png')
    write_top1_results_to_file(top_sd_results, top_scd_results, top_scdo_results, base_scores, delta_ts, dataset.name,
                               filename=f'results/{dataset.name}_results.txt')
    write_results_to_csv(top_sd_results, top_scd_results, top_scdo_results, base_scores, delta_ts, dataset.name, filename='results/results.csv')


def load_or_fetch_dataset(fetch_func, pickle_filename):
    """Loads dataset from a pickle file if it exists; otherwise, fetches, pickles, and returns it."""
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as file:
            dataset = pickle.load(file)
    else:
        dataset = fetch_func()
        with open(pickle_filename, 'wb') as file:
            pickle.dump(dataset, file)
    return dataset

# Function to handle parallel execution
def process_dataset(dataset, delta_ts, delta_ts_label):
    try:
        generate_results(dataset, delta_ts, delta_ts_label)
    except Exception as e:
        logging.error(f"Error processing dataset {dataset}: {e}")
        raise  # Re-raise to propagate to the main process

if __name__ == "__main__":
    # Define dataset loading with pickling
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
    # save_autocorrelation_jaccard_to_csv(datasets_virtual, delta_ts_virtual, filename='results/correlation_jaccard_virtual.csv')
    # save_autocorrelation_jaccard_to_csv(datasets_physical, delta_ts_physical, filename='results/correlation_jaccard_physical.csv')
    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks for physical datasets
        futures_physical = [executor.submit(process_dataset, dataset, delta_ts_physical, 'physical') for dataset in
                            datasets_physical]
        # Submit tasks for virtual datasets
        futures_virtual = [executor.submit(process_dataset, dataset, delta_ts_virtual, 'virtual') for dataset in
                           datasets_virtual]

        # Wait for all tasks to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures_physical + futures_virtual):
                future.result()  # This will raise any exceptions caught in process_dataset

