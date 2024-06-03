from collections import Counter, defaultdict
from get_data import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import *


def plot_graph_density(G, title, bin_size='1h'):
    # Extract timestamp and connections count
    data = [(data['timestamp'], 1) for _, _, data in G.edges(data=True)]
    df = pd.DataFrame(data, columns=['timestamp', 'connections'])

    # Convert UNIX timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Group data into bins based on bin_size
    df.set_index('timestamp', inplace=True)
    df = df.resample(bin_size).sum().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['connections'], marker='o', linestyle='-')

    # Dynamically adjust the number of labels based on the date range and desired density
    date_range = df['timestamp'].max() - df['timestamp'].min()
    max_labels = 10  # Maximum number of labels you want to show on the x-axis

    if bin_size.endswith('min'):  # Minutes
        interval = max(1, int(date_range.total_seconds() / 60 / max_labels))
        locator = mdates.MinuteLocator(interval=interval)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif bin_size.endswith('h'):  # Hours
        interval = max(1, int(date_range.total_seconds() / 3600 / max_labels))
        locator = mdates.HourLocator(interval=interval)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif bin_size.endswith('D'):  # Days
        interval = max(1, int(date_range.days / max_labels))
        locator = mdates.DayLocator(interval=interval)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        locator = mdates.AutoDateLocator(minticks=3,
                                         maxticks=max_labels)  # Fallback for unspecified or complex intervals

    plt.gca().xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()  # Rotate date labels

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of Connections')
    plt.tight_layout()
    plt.show()

def plot_graph_density_aggregated(snapshots, title):
    snapshots = sorted(snapshots, key=lambda g: g.graph['t'])
    data = defaultdict(int)
    for g in snapshots:
        for _, _, d in g.edges(data=True):
            data[g.graph['t']] += d['weight']

    df = pd.DataFrame(data.items(), columns=['timestamp', 'connections'])
    df['timestamp'] = list(map(lambda g: g.graph['t'], snapshots))

    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['connections'], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Number of Connections')
    plt.tight_layout()
    plt.show()


def print_graph_properties(G, plot_bin_size='1h'):
    print(f"---------------------------{G.name}--------------------------------")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    timestamps = [data['timestamp'] for _, _, data in G.edges(data=True)]
    print(f"Number of timestamps: {int((max(timestamps) - min(timestamps)))}")
    interactions = Counter((min(node1, node2), max(node1, node2)) for node1, node2 in G.edges())
    unique_interactions = len(interactions)
    avg_interactions_per_pair = sum(interactions.values()) / unique_interactions
    print(f"Total number of unique interactions: {unique_interactions}")
    print(f"Average number of recurrent interactions: {avg_interactions_per_pair:.2f}")
    plot_graph_density(G, G.name, bin_size=plot_bin_size)


def print_graph_properties_aggregated(snapshots):
    print(f"-----------------------{snapshots[0].name} aggregated -------------------------")
    # Print aggregated graph properties
    delta_t = snapshots[0].graph['delta_t']
    print("delta_t (seconds): ", delta_t)
    print(f"Number of aggregated snapshots: {snapshots[-1].graph['t'] + 1}")

    # Extracting aggregated timestamp info and weights
    weights = sum([[data['weight'] for _, _, data in g.edges(data=True)] for g in snapshots], [])
    # Since interactions are now aggregated, we count edges for unique interactions directly
    total_weight = sum(weights)
    print(f"Total number of interactions: {len(weights)}")
    print(f"Total weight (number of interactions): {total_weight}")
    print(f"Average weight (number of interactions per aggregated pair): {total_weight / len(weights):.2f}")

    # New metric: Average number of aggregated interactions per snapshot
    avg_interactions_per_snapshot = total_weight / len(snapshots)
    print(f"Average number of aggregated interactions per snapshot: {avg_interactions_per_snapshot:.2f}")
    plot_graph_density_aggregated(snapshots, f"{snapshots[0].name} aggregated")


# Example usage
plt.close('all')
# print_graph_properties(get_hypertext(), plot_bin_size='1h')
print_graph_properties_aggregated(aggregate_into_snapshots(get_hypertext(), delta_t=H))
print_graph_properties_aggregated(aggregate_into_snapshots_step(get_hypertext(), delta_t=H, step_t=H))
# print('########################################################################################')
# print_graph_properties(get_college(), plot_bin_size='1D')
# print('########################################################################################')
# print_graph_properties(get_college_1(), plot_bin_size='1D')
# print('########################################################################################')
# print_graph_properties(get_college_2(), plot_bin_size='1D')
# print_graph_properties_aggregated(aggregate_into_snapshots(get_college_2(), delta_t=D))
# print('########################################################################################')
# print_graph_properties(get_SFHH(), plot_bin_size='30min')
# print_graph_properties_aggregated(aggregate_into_snapshots(get_SFHH(), delta_t=30 * M))
# print('########################################################################################')
# print_graph_properties(get_socio_calls(), plot_bin_size='1D')
# print_graph_properties_aggregated(aggregate_into_snapshots(get_socio_calls(), delta_t=7 * D))
# print('########################################################################################')
# print_graph_properties(get_socio_sms_original(), plot_bin_size='1D')
# print('########################################################################################')
# print_graph_properties(get_socio_sms(), plot_bin_size='1D')
# print_graph_properties_aggregated(aggregate_into_snapshots(get_socio_sms(), delta_t=7 * D))
# print('########################################################################################')
# print_graph_properties_aggregated(get_infectious())
