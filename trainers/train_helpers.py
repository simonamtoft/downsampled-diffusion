import torch
import os
import wandb
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision import utils


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
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


def bce_loss(r, x):
    """ Binary Cross Entropy Loss """
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def log_images(x_recon, x_sample, folder:str, name:str, nrow:int):
    """Log reconstruction and sample images to wandb, for a VAE / DRAW model."""
    
    name_recon = f'{folder}/recon_{name}.png'
    name_sample = f'{folder}/sample_{name}.png'
    
    # save batch of images
    utils.save_image(x_recon, name_recon, nrow=nrow)
    utils.save_image(x_sample, name_sample, nrow=nrow)
    
    # Log the images to wandb
    wandb.log({
        "recon": wandb.Image(name_recon),
        "sample": wandb.Image(name_sample)
    }, commit=True)

    # Delete the logged images
    os.remove(name_recon)
    os.remove(name_sample)
