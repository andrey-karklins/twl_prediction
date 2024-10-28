import logging

from cross_fold_validation import model_no_fit
from get_data import aggregate_to_matrix, get_hypertext, get_SFHH, get_college_1, get_college_2, get_socio_calls, \
    get_socio_sms
from grid_search import write_results_to_csv, grid_search_scdo_model, grid_search_scd_model, grid_search_sd_model
from models.BaseModel import BaseModel
from utils import load_or_fetch_dataset, H, M, D

# Parameter definitions
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 1 * D, 3 * D]
taus = [0.1, 0.5, 1, 3, 5]
Ls = [1, 3, 5, 10]
coefs = [(0.8, 0.1, 0.1), (0.8, 0.2, 0), (0.8, 0, 0.2), (0.6, 0.2, 0.2), (0.6, 0.3, 0.1), (0.6, 0.1, 0.3)]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_results(dataset, delta_t):
    data, G = aggregate_to_matrix(dataset, delta_t)
    baseline_score = model_no_fit(data, BaseModel())

    # Grid search
    _, _, sd_results = grid_search_sd_model(data, taus, Ls)
    _, _, scd_results = grid_search_scd_model(data, taus, Ls, coefs, G)
    _, _, scdo_results = grid_search_scdo_model(data, taus, Ls, coefs, G)

    top_sd = sorted(sd_results, key=lambda x: x[-1]['MSE'])[0]
    top_scd = sorted(scd_results, key=lambda x: x[-1]['MSE'])[0]
    top_scdo = sorted(scdo_results, key=lambda x: x[-1]['MSE'])[0]

    write_results_to_csv(top_sd, top_scd, top_scdo, baseline_score, delta_t, dataset.name,
                         filename='results/results.csv')


# Main function for sequential execution
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

    for dataset in datasets_physical:
        for delta_t in delta_ts_physical:
            generate_results(dataset, delta_t)

    for dataset in datasets_virtual:
        for delta_t in delta_ts_virtual:
            generate_results(dataset, delta_t)

if __name__ == "__main__":
    main()
