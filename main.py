from data_analysis import apply_fourier_transform, plot_autocorrelation_jaccard
from grid_search import *
import concurrent.futures

# Assuming get_hypertext, get_SFHH, etc., are already defined functions
datasets_physical = [get_hypertext(), get_SFHH()]
datasets_virtual = [
    get_college_1(),
    get_college_2(),
    get_socio_calls(),
    get_socio_sms()
]
delta_ts_physical = [10 * M, 30 * M, 1 * H]
delta_ts_virtual = [1 * H, 12 * H, 1 * D]
taus = [0.1, 0.5, 1, 3, 5]
Ls = [1, 3, 5, 10]
coefs = [(1, 0, 0), (0.75, 0.25, 0), (0.75, 0, 0.25), (0.6, 0.2, 0.2)]

def generate_results(dataset, delta_ts, delta_ts_label):
    top_sd_results = []
    top_scd_results = []
    base_scores = []
    for delta_t in delta_ts:
        data, G = aggregate_to_matrix(dataset, delta_t)
        baseline_score = model_no_fit(data, BaseModel())
        # Grid search
        _, _, sd_results = grid_search_sdmodel(data, taus, Ls)
        _, _, scd_results = grid_search_scdmodel(data, taus, Ls, coefs, G)

        base_scores.append(baseline_score)
        top_sd_results.append(sorted(sd_results, key=lambda x: x[-1])[0])
        top_scd_results.append(sorted(scd_results, key=lambda x: x[-1])[0])

    # apply_fourier_transform(tmp_datasets, tmp_delta_ts, dataset.name, filename=f'results/plots/fourier_transform_{dataset.name}.png')
    write_top1_results_to_file(top_sd_results, top_scd_results, base_scores, delta_ts, dataset.name,
                               filename=f'results/{dataset.name}_results.txt')

# Function to handle parallel execution
def process_dataset(dataset, delta_ts, delta_ts_label):
    generate_results(dataset, delta_ts, delta_ts_label)

if __name__ == "__main__":
    plot_autocorrelation_jaccard(datasets_virtual, delta_ts_virtual)
    # Use ProcessPoolExecutor for parallel execution
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # Submit tasks for physical datasets
    #     futures_physical = [executor.submit(process_dataset, dataset, delta_ts_physical, 'physical') for dataset in datasets_physical]
    #     # Submit tasks for virtual datasets
    #     futures_virtual = [executor.submit(process_dataset, dataset, delta_ts_virtual, 'virtual') for dataset in datasets_virtual]
    #
    #     # Wait for all tasks to complete
    #     concurrent.futures.wait(futures_physical + futures_virtual)

    # Optionally, handle post-execution tasks or results collection
    # e.g., plotting correlation after all processing is done
    # plot_autocorrelation_jaccard(datasets_physical, delta_t+s_physical, filename='results/plots/correlation_jaccard_physical.png')
    # plot_autocorrelation_jaccard(datasets_virtual, delta_ts_virtual, filename='results/plots/correlation_jaccard_virtual.png')
