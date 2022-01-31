import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def log_gaussian(x, mu, log_var):
    """
        Returns the log pdf of a normal distribution parametrised
        by mu and log_var evaluated at x.
    
    Inputs:
        x       : the point to evaluate
        mu      : the mean of the distribution
        log_var : the log variance of the distribution
    Returns:
        log(N(x | mu, sigma))
    """
    log_pdf = (
        - 0.5 * math.log(2 * math.pi) 
        - log_var / 2 
        - (x - mu)**2 / (2 * torch.exp(log_var))
    )
    return torch.sum(log_pdf, dim=-1)


def log_standard_gaussian(x):
    """
        Returns the log pdf of a standard normal distribution N(0, 1)
    
    Inputs:
        x   : the point to evaluate
    Returns:
        log(N(x | 0, I))
    """
    log_pdf = (
        -0.5 * math.log(2 * math.pi) 
        - x ** 2 / 2
    )

    return torch.sum(log_pdf, dim=-1)


# Define reparameterization trick
def reparametrize(mu, log_var):
    # draw epsilon from N(0, 1)
    eps = Variable(torch.randn(mu.size()), requires_grad=False)

    # Ensure it is on correct device
    if mu.is_cuda:
        eps = eps.cuda()

    # std = exp(log_std) <= log_std = 0.5 * log_var
    std = log_var.mul(0.5).exp_()

    # trick: z = std*eps + mu
    z = mu.addcmul(std, eps)
    return z


class GaussianSample(nn.Module):
    """ 
    Layer that enables sampling from a Gaussian distribution
    By calling GaussianSample(x), it returns [z, mu, log_var]
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create linear layers for the mean and log variance
        self.mu = nn.Linear(self.in_features, self.out_features)
        self.log_var = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return reparametrize(mu, log_var), mu, log_var