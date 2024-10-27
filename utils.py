import os
import pickle

import numpy as np
from numba import float64

M = 60  # 1 minute in seconds
H = M * 60  # 1 hour in seconds
D = H * 24  # 1 day in seconds


def seconds_to_human_readable(seconds):
    """
    Convert a given number of seconds into a human-readable format (days, hours, minutes, seconds).

    Args:
    seconds (int): Number of seconds.

    Returns:
    str: Human-readable string representing the time in days, hours, minutes, and seconds.
    """
    # Define time units
    day = 86400  # seconds in a day
    hour = 3600  # seconds in an hour
    minute = 60  # seconds in a minute

    # Calculate the number of days, hours, minutes, and seconds
    days = seconds // day
    seconds %= day
    hours = seconds // hour
    seconds %= hour
    minutes = seconds // minute
    seconds %= minute

    # Create a human-readable string
    human_readable = []
    if days > 0:
        human_readable.append(f"{days}d")
    if hours > 0:
        human_readable.append(f"{hours}h")
    if minutes > 0:
        human_readable.append(f"{minutes}m")
    if seconds > 0 or not human_readable:  # Include seconds even if zero or if the list is empty
        human_readable.append(f"{seconds}s")
    human_readable = " ".join(human_readable)
    return human_readable

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


def geo_mean(iterable):
    a = np.array(iterable, dtype=np.float64)

    # Filter out zeros and negative values to prevent issues with log
    a = a[a > 0]
    if len(a) == 0:
        return 0  # or np.nan, depending on your use case

    # Use log transformation to avoid overflow, then take the mean
    return np.exp(np.mean(np.log(a)))