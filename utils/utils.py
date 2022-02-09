def min_max_norm(x):
    """Returns the min-max normalization of x."""
    return (x - x.min()) / (x.max() - x.min())