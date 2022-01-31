import torch
import os
import wandb
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from torchvision.utils import save_image


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


def log_images(x_recon, x_sample, append_str:str, folder:str, n_channels:int=1):
    """Log reconstruction and sample images to wandb, for a VAE / DRAW model."""
    
    # Convert recon and sample input to images
    name_recon = convert_img(x_recon, f"{folder}/recon_{append_str}", n_channels)
    name_sample = convert_img(x_sample, f"{folder}/sample_{append_str}", n_channels)

    # Log the images to wandb
    wandb.log({
        "recon": wandb.Image(name_recon),
        "sample": wandb.Image(name_sample)
    }, commit=True)

    # Delete the logged images
    os.remove(name_recon)
    os.remove(name_sample)


def convert_img(img, path_name:str, n_channels:int=1):
    name_jpg = f'{path_name}.jpg'
    name_png = f'{path_name}.png'

    # Save batch as single image
    save_image(img, name_jpg)

    # Load image
    imag = image.imread(name_jpg)[:, :, 0]

    # Delete image
    os.remove(name_jpg)

    # Save image as proper plots
    save_name = plt_img_save(imag, name_png, n_channels)
    
    # return name of final image file
    return save_name


def reshape_color(img, img_shape, n_channels):
    img_ = np.reshape(img, img_shape)
    if n_channels != 1:
        img_ = np.moveaxis(img_, 0, -1)
    return img_

def plt_img_save(img, name:str='log_image.png', n_channels:int=1):
    # Fix dimensionality for saving
    N = img.shape[0]
    K = img.shape[1]
    if n_channels == 1:
        k = int(np.sqrt(K))
        img_shape = (k, k)
    else:
        K = int(K/n_channels)
        k = int(np.sqrt(K))
        img_shape = (n_channels, k, k)
    print(img_shape)
    
    # define plot
    if N >= 16:
        f, ax = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            idx = i*2
            for j in range(2):
                img_ = img[idx+j, :]
                img_ = reshape_color(img_, img_shape, n_channels)
                ax[j, i].imshow(img_, cmap='gray')
                ax[j, i].axis('off')
    else:
        f, ax = plt.subplots(1, N, figsize=(16, 4))
        for i in range(N):
            img_ = img[i, :]
            img_ = reshape_color(img_, img_shape, n_channels)
            ax[i].imshow(img_, cmap='gray')
            ax[i].axis('off')
    
    f.savefig(name, transparent=True, bbox_inches='tight')
    plt.close()
    return name
