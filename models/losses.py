import torch
import torch.nn.functional as F
import numpy as np
from .helpers import get_ones_like


def l1_loss(target:torch.tensor, output:torch.tensor) -> torch.tensor:
    """Computes the l1 loss between noise and reconstruction."""
    return (target - output).abs().mean()


def l2_loss(target:torch.tensor, output:torch.tensor, reduction:str='mean') -> torch.tensor:
    """Computes the l2 loss between output and target."""
    return F.mse_loss(target, output, reduction=reduction)


def normal_kl(mean1:torch.tensor, logvar1:torch.tensor, mean2:torch.tensor, logvar2:torch.tensor) -> torch.tensor:
    """
    Compute the KL divergence between two gaussians:
        D_KL( p1 || p2 ) = log(std2 / std1) + (var1 + (mean1 - mean2)^2) / (2*var2) - 1/2
    Rearranged according to log variances:
        D_KL( p1 || p2 ) = 0.5 * (logvar2 - logvar1 - 1 + exp(logvar1 - logvar2) + (mean1 - mean2)^2 * exp(-logvar2))
    Shapes are automatically broadcasted, so batches can be compared to scalars, among other use cases.
    
    Args:
        mean1 (torch.tensor):   The mean of the first Gaussian distribution.
        logvar1 (torch.tensor): The log variance of the first Gaussian distribution.
        mean2 (torch.tensor):   The mean of the second Gaussian distribution.
        logvar2 (torch.tensor): The log variance of the second Gaussian distribution.
        
    Returns:
        The KL divergence between the two gaussians given by their means and log variances.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # compute KL-divergence
    return 0.5 * (
        + logvar2 - logvar1 - 1.0
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the standard normal.
    
    Reference:
    Original TensorFlow implementation is done by Jonathan Ho.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py#L112
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x:torch.tensor, *, means:torch.tensor, log_scales:torch.tensor) -> torch.tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    
    Args:
        x (torch.tensor):           The target images of shape (N x C x H x W). 
                                    It is assumed that this was uint8 values,
                                    rescaled to the range [-1, 1].
        means (torch.tensor):       The Gaussian mean Tensor of shape (N x C x H x W).
        log_scales (torch.tensor):  The Gaussian log stddev Tensor of shape (N x C x H x W)
                                    or a single value for each batch (N x 1 x 1 x 1).
        
    Returns
        A tensor of same shape as x of log probabilities (in nats).
    
    Reference:
    Original TensorFlow implementation is done by Jonathan Ho.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py#L116
    """
    # match shape of log_scales if on shape (N x 1 x 1 x 1)
    if list(log_scales.shape) == [x.shape[0], 1, 1, 1]:
        log_scales = log_scales * get_ones_like(x)
    assert x.shape == means.shape == log_scales.shape
    
    # compute discretized log-likelihood of Gaussian distribution
    # given by means and log_scales
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1. - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
