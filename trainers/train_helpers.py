import os
import wandb
import torch
from torch import tensor
from torchvision import utils


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


def bce_loss(r:tensor, x:tensor) -> tensor:
    """Binary Cross Entropy Loss"""
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def delete_if_exists(path:str):
    if os.path.exists(path):
        os.remove(path)


def log_images(x_recon:tensor=None, x_sample:tensor=None, folder:str='.', name:str='tmp', nrow:int=None, rname:str=None, sname:str=None, commit:bool=True):
    """Log reconstruction and sample images to wandb."""

    # instantiate
    log_dict = {}
    name_recon = './tmp/recon_tmp.png'
    name_sample = './tmp/sample_tmp.png'
    
    # set wandb log names
    rname = 'recon' if rname is None else rname
    sname = 'sample' if sname is None else sname
    
    # add reconstruction to log
    if x_recon is not None:
        name_recon = os.path.join(folder, f'recon_{name}.png')
        utils.save_image(x_recon, name_recon, nrow=nrow)
        log_dict[rname] = wandb.Image(name_recon)
     
    # add samples to log
    if x_sample is not None:
        name_sample = os.path.join(folder, f'sample_{name}.png')
        utils.save_image(x_sample, name_sample, nrow=nrow)
        log_dict[sname] = wandb.Image(name_sample)

    # Log the images to wandb
    wandb.log(log_dict, commit=commit)
    return name_recon, name_sample