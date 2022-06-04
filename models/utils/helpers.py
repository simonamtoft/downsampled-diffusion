import torch
from torch import tensor
from inspect import isfunction


def exists(x):
    return x is not None


def get_identity_like(x) -> tensor:
    """Return Identity matrix matching x of shape (N x C x H x W)"""
    shape = x.shape
    return (
        torch.eye(shape[-1], device=x.device, dtype=x.dtype)
        .repeat(shape[1], 1, 1)
        .repeat(shape[0], 1, 1, 1)
    )


def get_ones_like(x:tensor) -> tensor:
    """Return tensor filled with ones of same shape, device and type as x"""
    return torch.ones(x.shape, dtype=x.dtype, device=x.device)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a:tensor, t:tensor, x_shape) -> tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
