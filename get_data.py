import glob
import os
import datetime
from collections import defaultdict

import networkx as nx

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds


# Format: (100, 106, {'timestamp': 1246360360})
# Non aggregated, 2009-06-29 - 2009-07-01 (3 days)
def get_hypertext():
    G = nx.MultiGraph(name="Hypertext graph")  # Initialize an empty MultiGraph
    with open("data/hypertext.dat", "r") as file:
        for line in file:
            node1, node2, _, timestamp = line.strip().split()  # Adjust split method based on your file's delimiter
            G.add_edge(int(node1), int(node2), timestamp=int(timestamp))
    return G


# Format: (1, 2, {'timestamp': 1082040961})
# Non aggregated, 193 days in total
def get_college():
    G = nx.MultiGraph(name="College graph")  # Initialize an empty MultiGraph
    with open("data/CollegeMsg.txt", "r") as file:
        for line in file:
            node1, node2, timestamp = line.strip().split()  # Adjust split method based on your file's delimiter
            G.add_edge(int(node1), int(node2), timestamp=int(timestamp))
    return G


# Format: (0, 1, {'timestamp': '2009-07-16', 'weight': 23})
# Aggregated by 24h, 69 days in total
def get_infectious():
    def extract_timestamp_from_filename(filename):
        # Extracts and returns the timestamp from the filename
        return datetime.datetime.strptime(filename.split('_')[1].split('.')[0], "%Y-%m-%d").timestamp()

    graph_files = sorted(glob.glob('data/infectious/*.gml'))
    # Ensure D is defined; it represents the duration of each interval in seconds
    delta_t = D
    snapshots = {}  # Use a dictionary instead of a list

    for graph_file in graph_files:
        timestamp = extract_timestamp_from_filename(os.path.basename(graph_file))
        G = nx.read_gml(graph_file, label='id')

        # Calculate the interval index for the current file
        interval_index = int((timestamp - extract_timestamp_from_filename(graph_files[0])) // delta_t)

        # Create a new graph for the interval if it doesn't exist
        if interval_index not in snapshots:
            snapshots[interval_index] = nx.Graph(name="Infectious graph", t=interval_index, delta_t=delta_t)

        graph = snapshots[interval_index]

        for u, v, data in G.edges(data=True):
                graph.add_edge(u, v, weight=data.get('weight', 1))

    # Convert the dictionary to a sorted list of graphs by interval index
    return [snapshots[key] for key in sorted(snapshots)]


# Format: ('1467', '1591', {'timestamp': 1244108840})
# Non aggregated, 2009-06-04 - 2009-06-05 (2 days),
# added 1244066400 seconds to timestamps (2009-06-04 00:00:00 France time)
def get_SFHH():
    G = nx.MultiGraph(name="SFHH graph")
    filepath = "data/SFHH.dat"

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Split each line into parts
            timestamp, node1, node2 = int(parts[0]), parts[1], parts[2]
            G.add_edge(node1, node2, timestamp=timestamp + 1244066400)
    return G


# Format: (19, 18, {'timestamp': 1225677828, 'weight': 32767})
# Non aggregated, 2008-09-05 - 2009-06-29 (298 days), weight is number of seconds per call
# removed records with unknown caller or callee
def get_calls():
    G = nx.MultiGraph(name="Calls graph")
    filepath = "data/Calls.csv"

    def unix_from_str(s):
        return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(sep=',')  # Split each line into parts
            node1, timestamp, weight, node2 = parts[0], parts[1], parts[2], parts[3]
            if node1 == "" or node2 == "":
                continue
            G.add_edge(int(node1), int(node2[1:-1]), timestamp=unix_from_str(timestamp), weight=int(weight))
    return G


def aggregate_into_snapshots(G, delta_t):
    # Calculate the global minimum timestamp to establish the starting point for intervals
    timestamps = [data['timestamp'] for _, _, data in G.edges(data=True)]
    if not timestamps:
        return []  # Return an empty list if there are no edges

    min_timestamp = min(timestamps)

    # Use a dictionary to dynamically create graphs for intervals when needed
    snapshots = {}

    for u, v, data in G.edges(data=True):
        timestamp = data['timestamp']
        # Calculate the interval index based on the edge's timestamp
        interval_index = (timestamp - min_timestamp) // delta_t

        # Create a new graph for this interval if it doesn't exist
        if interval_index not in snapshots:
            snapshots[interval_index] = nx.Graph(name=G.name, t=interval_index, delta_t=delta_t)

        graph = snapshots[interval_index]

        # Aggregate edges as weights in the interval graph
        if graph.has_edge(u, v):
            graph[u][v]['weight'] += 1
        else:
            graph.add_edge(u, v, weight=1)

    # Convert the snapshots dictionary back into a sorted list of graphs
    return [snapshots[key] for key in sorted(snapshots)]
