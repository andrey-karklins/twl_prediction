import glob
import os
import datetime

import networkx as nx
import numpy as np

from utils import *


# Format: (100, 106, {'timestamp': 1246360360})
# Non aggregated, 2009-06-29 - 2009-07-01 (3 days)
def get_hypertext():
    G = nx.MultiGraph(name="Hypertext graph")  # Initialize an empty MultiGraph
    with open("data/Hypertext.dat", "r") as file:
        for line in file:
            node1, node2, _, timestamp = line.strip().split()  # Adjust split method based on your file's delimiter
            G.add_edge(int(node1), int(node2), timestamp=int(timestamp))
    return G


def get_college():
    G = nx.MultiGraph(name="College graph original")  # Initialize an empty MultiGraph
    with open("data/CollegeMsg.dat", "r") as file:
        for line in file:
            node1, node2, timestamp = line.strip().split()  # Adjust split method based on your file's delimiter
            G.add_edge(int(node1), int(node2), timestamp=int(timestamp))
    return G


def get_college_1():
    # 2004-06-20 00:00:00
    college_split = 1087660800
    G = nx.MultiGraph(name="College graph 1")  # Initialize an empty MultiGraph
    with open("data/CollegeMsg.dat", "r") as file:
        for line in file:
            node1, node2, timestamp = line.strip().split()  # A djust split method based on your file's delimiter
            if int(timestamp) > college_split:
                break
            G.add_edge(int(node1), int(node2), timestamp=int(timestamp))
    return G


def get_college_2():
    # 2004-06-20 00:00:00
    college_split = 1087660800
    G = nx.MultiGraph(name="College graph 2")  # Initialize an empty MultiGraph
    with open("data/CollegeMsg.dat", "r") as file:
        for line in file:
            node1, node2, timestamp = line.strip().split()  # Adjust split method based on your file's delimiter
            if int(timestamp) < college_split:
                continue
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
            graph.add_edge(int(u), int(v), weight=data["weight"])

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
            G.add_edge(int(node1), int(node2), timestamp=timestamp + 1244066400)
    return G


# Format: (19, 18, {'timestamp': 1225677828, 'weight': 32767})
# Non aggregated, 2008-09-05 - 2009-06-29 (298 days), weight is number of seconds per call
# removed records with unknown caller or callee
def get_socio_calls():
    G = nx.MultiGraph(name="Socio-calls graph")
    filepath = "data/Calls.dat"

    def unix_from_str(s):
        return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(sep=',')  # Split each line into parts
            node1, timestamp, weight, node2 = parts[0], parts[1], parts[2], parts[3].replace("\"", "")
            if node1 == "" or node2 == "":
                continue
            G.add_edge(int(node1), int(node2), timestamp=unix_from_str(timestamp), weight=int(weight))
    return G


def get_socio_sms_original():
    G = nx.MultiGraph(name="Socio-sms graph original")
    filepath = "data/SMS.dat"

    def unix_from_str(s):
        return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(sep=',')  # Split each line into parts
            node1, timestamp, _, node2 = parts[0], parts[1], parts[2], parts[3]
            if node1 == "" or node2 == "":
                continue
            G.add_edge(int(node1), int(node2), timestamp=unix_from_str(timestamp))
    return G


# Format: (19, 18, {'timestamp': 1225677828})
# Non aggregated, 2008-01-01 - 2009-06-27 (543 days)
# removed records with unknown recipient
def get_socio_sms():
    G = nx.MultiGraph(name="Socio-sms graph")
    filepath = "data/SMS.dat"

    # 2008-09-01 00:00:00
    socio_sms_start = 1220227200
    # 2009-04-01 00:00:00
    socio_sms_end = 1238534400

    def unix_from_str(s):
        return int(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(sep=',')  # Split each line into parts
            node1, timestamp, _, node2 = parts[0], parts[1], parts[2], parts[3]
            if node1 == "" or node2 == "":
                continue
            if unix_from_str(timestamp) < socio_sms_start or unix_from_str(timestamp) > socio_sms_end:
                continue
            G.add_edge(int(node1), int(node2), timestamp=unix_from_str(timestamp))
    return G


def aggregate_into_snapshots(G, delta_t, step_t=None):
    if step_t is None:
        step_t = delta_t

    min_timestamp = min([data['timestamp'] for _, _, data in G.edges(data=True)])
    snapshots = {}

    for u, v, data in G.edges(data=True):
        timestamp = data['timestamp']
        interval_index = (timestamp - min_timestamp) // step_t

        while interval_index * step_t + delta_t > timestamp - min_timestamp and interval_index >= 0:
            if interval_index not in snapshots:
                snapshots[interval_index] = nx.Graph(name=G.name, t=interval_index, delta_t=delta_t, step_t=step_t)
                snapshots[interval_index].add_nodes_from(G)
            graph = snapshots[interval_index]
            if graph.has_edge(u, v):
                graph[u][v]['weight'] += 1
            else:
                graph.add_edge(u, v, weight=1)
            interval_index -= 1

    return [snapshots[key] for key in sorted(snapshots)]


def aggregate_to_matrix(G, delta_t):
    edges = set([(u, v) for (u, v) in G.edges()])
    global_G = nx.Graph(name=G.name, delta_t=delta_t)
    for i, (u, v) in enumerate(edges):
        global_G.add_edge(u, v, id=i)
    timestamps = [data['timestamp'] for _, _, data in G.edges(data=True)]
    min_timestamp = min(timestamps)
    time_dict = {}
    for i, (u, v) in enumerate(edges):
        for _, _, data in G.edges((u, v), data=True):
            timestamp = data['timestamp']
            interval_index = int((timestamp - min_timestamp) // delta_t)
            if time_dict.get(interval_index) is None:
                time_dict[interval_index] = np.zeros(len(edges))
            time_dict[interval_index][i] += 1
    matrix = np.zeros((len(edges), len(time_dict)))
    for i, k in enumerate(sorted(time_dict.keys())):
        matrix[:, i] = time_dict[k]
    return matrix, global_G