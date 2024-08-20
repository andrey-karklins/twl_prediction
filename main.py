from grid_search import *

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
coefs = [(1, 0, 0), (0.8, 0.2, 0), (0.8, 0, 0.2), (0.8, 0.1, 0.1)]

datasets = []

for dataset in datasets_physical:
    for delta_t in delta_ts_physical:
        datasets.append(aggregate_to_matrix(dataset, delta_t))
for dataset in datasets_virtual:
    for delta_t in delta_ts_virtual:
        datasets.append(aggregate_to_matrix(dataset, delta_t))

# plot_autocorrelation_jaccard(datasets_physical, delta_ts_physical, filename='results/plots/correlation_jaccard_physical.png')
# plot_autocorrelation_jaccard(datasets_virtual, delta_ts_virtual, filename='results/plots/correlation_jaccard_virtual.png')

for (data, G) in datasets:
    # apply_fourier_transform(data, filename=f'results/plots/{G.name}_{seconds_to_human_readable(G.graph["delta_t"])}_fft.png')
    # Baseline
    baseline_score = model_no_fit(data, BaseModel())

    # Grid search
    _, _, sd_results = grid_search_sdmodel(data, taus, Ls)
    _, _, scd_results = grid_search_scdmodel(data, taus, Ls, coefs, G)

    print("---------------------------------------------------")
    print("Baseline score: ", baseline_score)
    print("---------------------------------------------------")

    # Write top results to files
    write_combined_results_to_file(sd_results, scd_results, baseline_score, filename=f'results/{G.name}_{seconds_to_human_readable(G.graph["delta_t"])}.txt', top_n = 5)