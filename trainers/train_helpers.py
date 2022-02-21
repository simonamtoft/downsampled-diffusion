import torch
import os
import wandb
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision import utils


def compute_bits_dim(nats:torch.Tensor, x_dim:int):
    """Compute the bits/dim from nats"""
    nats_dim = nats / x_dim
    bits_dim = nats_dim / np.log(2)
    return bits_dim


def mean_and_bits_dim(x_dim:int, loss:list):
    """Takes mean of input loss and returns the bits dim instead of nats."""
    nats = nats_mean(loss)
    return compute_bits_dim(nats, x_dim)


def nats_mean(loss:list):
    """Takes mean of input loss, thus returning nats"""
    return np.array(loss).mean()


def cycle(dl):
    """Makes the input DataLoader cyclic. Used for DDPM training."""
    while True:
        for data in dl:
            yield data


def num_to_groups(num:int, divisor:int):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def lambda_lr(n_epochs:int, offset:int=0, delay:int=0):
    """
    Creates learning rate step function for LambdaLR scheduler.
    Stepping starts after "delay" epochs and will reduce LR to 0 when "n_epochs" has been reached
    Offset is used continuing training models.
    """
    if (n_epochs - delay) == 0:
        raise Exception("Error: delay and n_epochs cannot be equal!")
    return lambda epoch: 1 - max(0, epoch + offset - delay)/(n_epochs - delay)


class DeterministicWarmup(object):
    """Linear deterministic warm-up over n epochs. Ends at t_max."""
    def __init__(self, n:int=100, t_max:int=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t


def bce_loss(r:torch.Tensor, x:torch.Tensor):
    """Binary Cross Entropy Loss"""
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def delete_if_exists(path:str):
    if os.path.exists(path):
        os.remove(path)


def log_images(x_recon:torch.Tensor=None, x_sample:torch.Tensor=None, folder:str='.', name:str='tmp', nrow:int=None):
    """Log reconstruction and sample images to wandb."""

    # instantiate
    log_dict = {}
    name_recon = './tmp/recon_tmp.png'
    name_sample = './tmp/sample_tmp.png'
    
    # add reconstruction to log
    if x_recon is not None:
        name_recon = f'{folder}/recon_{name}.png'
        utils.save_image(x_recon, name_recon, nrow=nrow)
        log_dict['recon'] = wandb.Image(name_recon)
     
    # add samples to log
    if x_sample is not None:
        name_sample = f'{folder}/sample_{name}.png'
        utils.save_image(x_sample, name_sample, nrow=nrow)
        log_dict['sample'] = wandb.Image(name_sample)
    
    # Log the images to wandb
    wandb.log(log_dict, commit=True)

    # Delete the logged images
    delete_if_exists(name_recon)
    delete_if_exists(name_sample)


def min_max_norm(x):
    """Returns the min-max normalization of x."""
    return (x - x.min()) / (x.max() - x.min())
