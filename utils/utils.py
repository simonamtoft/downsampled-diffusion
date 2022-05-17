import torch
import numpy as np


def modify_config(config, model_config):
    for key, value in model_config.items():
        config[key] = value
    return config


def min_max_norm_batch(x:torch.tensor):
    """Returns the min-max normalization of x across batch."""
    return (x - x.min()) / (x.max() - x.min())


def min_max_norm_image(x:torch.tensor):
    """
    Returns the min-max normalization per image in the
    batch instead of over the entire batch of images.
    """
    b = x.shape[0]
    x_min = x.view(b, -1).min(dim=1).values[:, None, None, None]
    x_max = x.view(b, -1).max(dim=1).values[:, None, None, None]
    return (x - x_min) / (x_max - x_min)


def reduce_mean(x:torch.tensor) -> torch.tensor:
    """
    Reduce input x to a single dimension tensor 
    by taking the mean over all non-batch dimensions.
    """
    return x.mean(dim=list(range(1, len(x.shape))))


def reduce_sum(x:torch.tensor) -> torch.tensor:
    """
    Reduce input x to a single dimension tensor 
    by taking the sum over all non-batch dimensions.
    """
    return x.sum(dim=list(range(1, len(x.shape))))


def flat_bits(x:torch.tensor):
    """
    Take the mean over all non-batch dimensions, and scale by log(2).
    Returns the bits per dim of the input tensor.
    """
    return reduce_mean(x) / np.log(2.)


def get_model_state_dict(save_data):
    if 'ema_model' in save_data:
        return save_data['ema_model']
    return save_data['model']
