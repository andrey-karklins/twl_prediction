import logging
import concurrent.futures
from models.BaseModel import BaseModel
from grid_search import *

delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.1, 0.5, 1, 3, 5]
Ls = [1, 3, 5, 10]
coefs = [
    (0.8, 0.1, 0.1), (0.8, 0.2, 0), (0.8, 0, 0.2),
    (0.6, 0.2, 0.2), (0.6, 0.3, 0.1), (0.6, 0.1, 0.3),
    (0.5, 0.3, 0.2), (0.5, 0.2, 0.3), (0.5, 0.4, 0.1), (0.5, 0.1, 0.4)
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_results_for_combination(dataset, delta_t, delta_ts_label):
    data, G = aggregate_to_matrix(dataset, delta_t)
    baseline_score = model_no_fit(data, BaseModel())

    # Perform grid search
    _, _, sd_results = grid_search_sd_model(data, taus, Ls)
    _, _, scd_results = grid_search_scd_model(data, taus, Ls, coefs, G)
    _, _, scdo_results = grid_search_scdo_model(data, taus, Ls, coefs, G)


    # Collect top results based on MSE
    top_sd_result = sorted(sd_results, key=lambda x: x[-1]['MSE'])[0]
    top_scd_result = sorted(scd_results, key=lambda x: x[-1]['MSE'])[0]
    top_scdo_result = sorted(scdo_results, key=lambda x: x[-1]['MSE'])[0]

    return (dataset.name, delta_t, baseline_score, top_sd_result, top_scd_result, top_scdo_result)


def save_results(results, dataset_name, delta_ts_label):
    base_scores, top_sd_results, top_scd_results, top_scdo_results, delta_ts = zip(*results)

    # Write the top results to files
    write_top1_results_to_file(
        top_sd_results, top_scd_results, top_scdo_results, base_scores, delta_ts, dataset_name,
        filename=f'results/{dataset_name}_results.txt'
    )
    write_results_to_csv(
        top_sd_results, top_scd_results, top_scdo_results, base_scores, delta_ts, dataset_name,
        filename='results/results.csv'
    )


def process_datasets_concurrently(datasets, delta_ts, delta_ts_label):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_combination = {
            executor.submit(generate_results_for_combination, dataset, delta_t, delta_ts_label): (dataset.name, delta_t)
            for dataset in datasets
            for delta_t in delta_ts
        }

        for future in concurrent.futures.as_completed(future_to_combination):
            dataset_name, delta_t = future_to_combination[future]
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Error processing {dataset_name} with delta_t {delta_t}: {e}")

    if results:
        dataset_name = results[0][0]
        save_results(results, dataset_name, delta_ts_label)


if __name__ == "__main__":
    # Define datasets
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

    # Process physical datasets concurrently by dataset, delta_t combination
    process_datasets_concurrently(datasets_physical, delta_ts_physical, 'physical')

    # Process virtual datasets concurrently by dataset, delta_t combination
    process_datasets_concurrently(datasets_virtual, delta_ts_virtual, 'virtual')
