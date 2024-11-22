import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from get_data import aggregate_to_matrix, get_hypertext, get_SFHH, get_college_1, get_college_2, get_socio_calls, \
    get_socio_sms
from grid_search import grid_search
from utils import load_or_fetch_dataset, H, M, D, load_completed_tasks
# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.linear_model._coordinate_descent")

# Parameter definitions
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.1, 0.5, 1, 3]
Ls = [1 / 2, 1 / 4, 1 / 8]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_results(dataset, delta_t):
    data, G = aggregate_to_matrix(dataset, delta_t)
    grid_search(data, taus, Ls, G, delta_t)


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

    # # Load completed tasks
    # completed_tasks = load_completed_tasks()

    # Combine all datasets and delta_t values into a single list of tasks
    all_tasks = [(dataset, delta_t) for dataset in datasets_physical for delta_t in delta_ts_physical] + \
                [(dataset, delta_t) for dataset in datasets_virtual for delta_t in delta_ts_virtual]

    # # Filter out tasks that are already completed
    # tasks = [(dataset, delta_t) for dataset, delta_t in all_tasks
    #          if (dataset.name, delta_t) not in completed_tasks]

    # if not tasks:
    #     logging.info("All tasks have already been completed.")
    #     return

    # Run tasks sequentially
    # for dataset, delta_t in all_tasks:
    #     generate_results(dataset, delta_t)

    # Run tasks concurrently
    with ProcessPoolExecutor(max_workers=3) as executor:  # Adjust max_workers as per CPU capacity
        futures = [executor.submit(generate_results, dataset, delta_t) for dataset, delta_t in all_tasks]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                future.result()  # Will raise an exception if the task failed
            except Exception as e:
                logging.error(f"An error occurred during grid search: {e}")


if __name__ == '__main__':
    main()

# def calculate_mean_neighbour_count():
#     datasets_physical = [
#         load_or_fetch_dataset(get_hypertext, 'pickles/hypertext.pkl'),
#         load_or_fetch_dataset(get_SFHH, 'pickles/SFHH.pkl')
#     ]
#     datasets_virtual = [
#         load_or_fetch_dataset(get_college_1, 'pickles/college_1.pkl'),
#         load_or_fetch_dataset(get_college_2, 'pickles/college_2.pkl'),
#         load_or_fetch_dataset(get_socio_calls, 'pickles/socio_calls.pkl'),
#         load_or_fetch_dataset(get_socio_sms, 'pickles/socio_sms.pkl')
#     ]
#     for G in datasets_physical + datasets_virtual:
#         sum_common_neighbours = 0
#         for e in G.common_neighbor_geometric_cache:
#             sum_common_neighbours += len(G.common_neighbor_geometric_cache[e])
#         mean_common_neighbours = sum_common_neighbours / len(G.common_neighbor_geometric_cache)
#         sum_distinct_neighbours = 0
#         for e in G.neighbor_edges_cache_1:
#             sum_distinct_neighbours += len(G.neighbor_edges_cache_1[e])
#             sum_distinct_neighbours += len(G.neighbor_edges_cache_2[e])
#         mean_distinct_neighbours = sum_distinct_neighbours / len(G.neighbor_edges_cache_1)
#         # write to csv
#         with open('results/mean_neighbour_count.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([G.name, mean_common_neighbours, mean_distinct_neighbours])
