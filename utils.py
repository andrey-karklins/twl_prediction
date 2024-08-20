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

    return human_readable