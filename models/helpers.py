import torch
import numpy as np
from inspect import isfunction

def exists(x):
    return x is not None


def get_identity_like(x) -> torch.tensor:
    """Return Identity matrix matching x of shape (N x C x H x W)"""
    shape = x.shape
    return (
        torch.eye(shape[-1], device=x.device, dtype=x.dtype)
        .repeat(shape[1], 1, 1)
        .repeat(shape[0], 1, 1, 1)
    )


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Different beta schedule implementations 
    Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py
    """
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def flat_bits(tensor:torch.tensor):
    """
    Take the mean over all non-batch dimensions, and scale by log(2).
    Returns the bits per dim of the input tensor.
    """
    tensor = tensor.mean(dim=list(range(1, len(tensor.shape))))
    return tensor / np.log(2.)


def flat_nats(tensor:torch.tensor):
    """
    Returns the nats value of the input tensor. 
    Done by summing over all non-batch dimensions.
    """
    # tensor = tensor.sum(1)
    # return tensor.mean(dim=list(range(1, len(tensor.shape))))
    return tensor.sum(dim=list(range(1, len(tensor.shape))))
