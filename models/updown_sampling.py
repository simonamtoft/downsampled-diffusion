import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce, partial

DEFAULT_INTERPOLATE_MODE = 'bicubic'


def get_interpolate(size:tuple, mode:str, align:bool=True):
    """
    Deterministic interpolation, can work both as downsampling and upsampling.
    
    Args:
        size (tuple):   Resulting size after interpolation.
        mode (str):     The interpolation mode. For batched color image data
                        the mode can be one of: 'nearest', 'bicubic', 'bilinear' 
        align (bool):   Whether to align corners or not, check F.interpolate 
                        documentation for an explanation.
    """
    align = None if mode == 'nearest' else align
    return partial(
        F.interpolate,
        size=size,
        mode=mode,
        align_corners=align,
    )


class SimpleDownConv(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        unet_vars = [2, 2, 0]   # kernel_size, stride, padding for standard U-Net
        ddpm_vars = [3, 2, 1]   # kernel_size, stride, padding for U-Net in DDPM
        self.conv = nn.Conv2d(channels, channels*2, *ddpm_vars)

    def forward(self, x):
        return self.conv(x)


class SimpleUpConv(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels*2, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class SimpleDownConvPlus(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        unet_vars = [2, 2, 0]   # kernel_size, stride, padding for standard U-Net
        ddpm_vars = [3, 2, 1]   # kernel_size, stride, padding for U-Net in DDPM
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, *ddpm_vars),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUpConvPlus(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, 4, 2, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)


class UnetDownConv(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, 2, 2, 0)
        )

    def forward(self, x):
        return self.conv(x)


class UnetUpConv(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, 4, 2, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)


class AEDown(nn.Module):
    def __init__(self, shape_in:tuple):
        """
        Autoencoder Downsampling Module
        Reference: https://kharshit.github.io/blog/2019/02/15/autoencoder-downsampling-and-upsampling
        
        Args:
            shape_in (tuple):   Shape of input image before downsampling.
                                Should consist of (C x H x W), where H and W will 
                                be halved after downsampling.
        """
        super().__init__()
        assert len(shape_in) == 3
        
        # define output shape
        self.shape_out = (shape_in[0], int(shape_in[1]/2), int(shape_in[2]/2))
        
        # compute input and output dimensionality
        self.dim_in = reduce(mul, shape_in, 1)
        dim_out = reduce(mul, self.shape_out, 1)
        
        # instantiate encoder
        self.encoder = nn.Linear(self.dim_in, dim_out)
    
    def forward(self, x):
        # flatten original image input
        x = x.reshape(-1, self.dim_in)
        
        # convert to latent space
        z = self.encoder(x)
        z = F.relu(z)
        
        # return latent with image shape
        return z.reshape(-1, *self.shape_out)


class AEUp(nn.Module):
    def __init__(self, shape_out:tuple):
        """
        Autoencoder Upsampling Module
        Reference: https://kharshit.github.io/blog/2019/02/15/autoencoder-downsampling-and-upsampling
        
        Args:
            shape_out (tuple):  The resulting output image shape after upsampling.
                                Should consist of (C x H x W), where H and W is half
                                before upsampling.
        """
        super().__init__()
        assert len(shape_out) == 3
        
        # define output shape
        self.shape_out = shape_out
        shape_in = (shape_out[0], int(shape_out[1]/2), int(shape_out[2]/2))
        
        # compute input and output dimensionality
        self.dim_in = reduce(mul, shape_in, 1)
        dim_out = reduce(mul, self.shape_out, 1)
        
        # instantiate decoder
        self.decoder = nn.Linear(self.dim_in, dim_out)
    
    def forward(self, z):
        # flatten latent space input
        z = z.reshape(-1, self.dim_in)
        
        # convert to original image space
        x_hat = self.decoder(z)
        x_hat = torch.sigmoid(x_hat)
        
        # return image with image shape
        return x_hat.reshape(-1, *self.shape_out)


def get_upsampling(mode:str, shape:tuple, interpolate_mode:str=None):
    """
    Returns upsampling function.
        mode (str):             The way to perform upsampling. 
                                'deterministic' uses F.interpolate
                                'convolutional' uses a trainable convolutional network.
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
        interpolate_mode (str): The mode to use for deterministic upsampling.
                                Only valid when mode == 'deterministic'.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    
    if mode == 'deterministic':
        mode_ = DEFAULT_INTERPOLATE_MODE if interpolate_mode is None else interpolate_mode
        size = (shape[1], shape[2])
        return get_interpolate(size, mode_)
    elif mode == 'convolutional':
        return SimpleUpConv(in_channels)
    elif mode == 'convolutional_plus':
        return SimpleUpConvPlus(in_channels)
    elif mode == 'convolutional_unet':
        return UnetUpConv(in_channels)
    elif mode == 'autoencoder':
        return AEUp(shape)
    else:
        raise NotImplementedError(f'Upsampling method for "{mode}" not implemented!')


def get_downsampling(mode:str, shape:tuple, scale:int=None, interpolate_mode:str=None):
    """
    Returns downsampling function.
        mode (str):             The way to perform upsampling. 
                                'deterministic' uses F.interpolate
                                'convolutional' uses a trainable convolutional network
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
        scale (int):            How many times the result should be downsampled.
                                E.g. scale=2 on a (3x32x32) image results in a (3x16x16) image.
        interpolate_mode (str): The mode to use for deterministic downsampling.
                                Only valid when mode == 'deterministic'.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    
    if mode == 'deterministic':
        mode_ = DEFAULT_INTERPOLATE_MODE if interpolate_mode is None else interpolate_mode
        scale = 2 if scale is None else scale
        size = (int(shape[1] / scale), int(shape[2] / scale))
        assert size[0] % 2 == 0, 'result from downsampling should have even dimensions.'
        return get_interpolate(size, mode_)
    elif mode == 'convolutional':
        return SimpleDownConv(in_channels)
    elif mode == 'convolutional_plus':
        return SimpleDownConvPlus(in_channels)
    elif mode == 'convolutional_unet':
        return UnetDownConv(in_channels)
    elif mode == 'autoencoder':
        return AEDown(shape)
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
