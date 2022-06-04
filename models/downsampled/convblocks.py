import numpy as np
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from functools import partial


def get_interpolate(size:tuple, mode:str=None, align:bool=True):
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
    mode = 'bicubic' if mode is None else mode
    return partial(
        F.interpolate,
        size=size,
        mode=mode,
        align_corners=align,
    )


def get_3x3(in_dim:int, out_dim:int, stride:int=1, padding:int=1, padding_mode:str='zeros') -> nn.Conv2d:
    """Wrapper function to get a 3-by-3 convolution."""
    return nn.Conv2d(
        in_dim, out_dim, 
        kernel_size=3, 
        stride=stride, 
        padding=padding, 
        padding_mode=padding_mode
    )


def get_1x1(in_dim:int, out_dim:int) -> nn.Conv2d:
    """Wrapper function to get a 1-by-1 convolution."""
    return nn.Conv2d(
        in_dim, out_dim, 
        kernel_size=1, 
        stride=1, 
        padding=0,
    )


def get_4x4_transpose(in_dim:int, out_dim:int, stride:int=2, padding:int=1) -> nn.ConvTranspose2d:
    """Wrapper function to get a 4-by-4 transpose convolution."""
    return nn.ConvTranspose2d(
        in_dim, out_dim, 
        kernel_size=4, 
        stride=stride, 
        padding=padding
    )


class BaseConv(nn.Module):
    def __init__(self, dim:int=8, in_channels:int=3, n_downsamples:int=1):
        super().__init__()
        dims = [in_channels, *(np.ones(n_downsamples).astype(int) * dim)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
    
    def forward(self, x:tensor) -> tensor:
        return self.conv(x)


class SimpleDownConv(BaseConv):
    def __init__(self, dim:int=8, in_channels:int=3, n_downsamples:int=1):
        super().__init__(dim, in_channels, n_downsamples)
        conv_list = []
        for in_, out_ in self.in_out:
            conv_list.append(
                get_3x3(in_, out_, stride=2)
            )
        self.conv = nn.Sequential(*conv_list)


class SimpleUpConv(BaseConv):
    def __init__(self, dim:int=8, in_channels:int=3, n_downsamples:int=1):
        super().__init__(dim, in_channels, n_downsamples)
        conv_list = []
        for in_, out_ in self.in_out[::-1]:
            conv_list.append(
                get_4x4_transpose(out_, in_)
            )
        self.conv = nn.Sequential(*conv_list)


class ConvResBlock(nn.Module):
    def __init__(self, dim:int, in_channels:int, out_channels:int=None, upsample:bool=False, downsample:bool=False, dropout:float=0, residual:bool=False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        assert not (self.upsample and self.downsample), 'Does not make sense to both down- and upsample.'
        self.residual = residual # set to false for start/end

        # convolutional layers
        self.c1 = get_1x1(in_channels, dim)
        self.c2 = get_3x3(dim, dim)
        self.c3 = get_3x3(dim, dim)
        self.c4 = get_1x1(dim, out_channels)
        
        # dropout layer
        self.drop = nn.Dropout2d(p=dropout)
        
        # define activation function
        self.activation = F.mish # gelu elu silu

    def forward(self, x:tensor) -> tensor:
        # perform convolutions
        x_hat = self.c1(self.activation(x))
        x_hat = self.c2(self.activation(x_hat))
        x_hat = self.c3(self.activation(x_hat))
        x_hat = self.c4(self.activation(x_hat))

        # add dropout
        x_hat = self.drop(x_hat)

        # residual connection
        out = x + x_hat if self.residual else x_hat

        # perform up or downsampling
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        elif self.downsample:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class ConvResNet(nn.Module):
    def __init__(self, dim:int, in_channels:int, out_channels:int, n_downsamples:int=1, upsample:bool=False, dropout:float=0, n_blocks:float=1):
        super().__init__()
        downsample = not upsample
        conv_list = []

        # explode channels from in_channels to dim
        conv_list.append(get_1x1(in_channels, dim))

        # add convolutional Resnet blocks
        for _ in range(n_downsamples):
            conv_list.append(
                ConvResBlock(int(dim/2), dim, dim, upsample, downsample, dropout, residual=True),
            )
            if n_blocks > 1:
                for _ in range(n_blocks-1):
                    conv_list.append(
                        ConvResBlock(int(dim/2), dim, dim, False, False, dropout, residual=True)
                    )

        # condense channels from dim to out_channels
        conv_list.append(get_1x1(dim, out_channels))
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x:tensor) -> tensor:
        x = self.conv(x)
        return x
