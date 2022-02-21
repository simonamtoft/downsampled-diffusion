import torch


def min_max_norm(x:torch.Tensor) -> torch.Tensor:
    """Returns the min-max normalization of x."""
    return (x - x.min()) / (x.max() - x.min())
