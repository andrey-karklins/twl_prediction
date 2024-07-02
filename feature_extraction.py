import networkx as nx
import pandas as pd
from tsfresh import extract_features
from utils import *
from get_data import *

datasets = [(get_hypertext(), 20 * M),
            # (get_SFHH(), 10 * M),
            # (get_college_1(), D),
            # (get_college_2(), 2 * D),
            # (get_socio_calls(), 2 * D),
            # (get_socio_sms(), 6 * H),
            ]

datasets = [aggregate_to_matrix(G, delta_t) for G, delta_t in datasets]

