import torch


def min_max_norm(x:torch.Tensor) -> torch.Tensor:
    """Returns the min-max normalization of x."""
    return (x - x.min()) / (x.max() - x.min())


def modify_config(config, model_config):
    for key, value in model_config.items():
        config[key] = value
    return config
