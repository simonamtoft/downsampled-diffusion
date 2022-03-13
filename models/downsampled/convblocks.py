import numpy as np
import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from functools import partial
from models.unet.blocks import ResnetBlock, Residual, LinearAttention, \
    PreNorm, Downsample, Upsample, Block


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
    def __init__(self, dim:int, in_channels:int, out_channels:int=None, upsample:bool=False, dropout:float=0, residual:bool=False):
        super().__init__()
        self.upsample = upsample
        self.residual = residual # set to false for start/end

        # convolutional layers
        self.c1 = get_1x1(in_channels, dim)
        self.c2 = get_3x3(dim, dim)
        self.c3 = get_3x3(dim, dim)
        self.c4 = get_1x1(dim, out_channels)
        
        # dropout layer
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x:tensor) -> tensor:
        # perform convolutions
        x_hat = self.c1(F.gelu(x))
        x_hat = self.c2(F.gelu(x_hat))
        x_hat = self.c3(F.gelu(x_hat))
        x_hat = self.c4(F.gelu(x_hat))
        
        # add dropout
        x_hat = self.drop(x_hat)
        
        # residual connection
        out = x + x_hat if self.residual else x_hat
        
        # perform up or downsampling
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        else:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class ConvResNet(nn.Module):
    def __init__(self, dim:int, in_channels:int, out_channels:int, n_downsamples:int=1, upsample:bool=False, dropout:float=0):
        super().__init__()
        dims = [in_channels, *(np.ones(n_downsamples).astype(int) * dim), out_channels]
        dim_list = list(zip(dims[:-1], dims[1:], dims[2:]))
        conv_list = []
        for i, (in_, mid_, out_) in enumerate(dim_list):
            # residual = i != n_downsamples
            conv_list.append(
                ConvResBlock(mid_, in_, out_, upsample, dropout, False)
            )
        self.conv = nn.Sequential(*conv_list)

    def forward(self, x:tensor) -> tensor:
        return self.conv(x)


class UnetBase(nn.Module):
    def __init__(self, dim:int=8, in_channels:int=3, out_channels:int=3, n_downsamples:int=1, n_groups:int=1, dropout:float=0):
        super().__init__()
        dims = [in_channels, *(np.ones(n_downsamples).astype(int) * dim), out_channels]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.dropout = dropout
        self.n_groups = n_groups
        self.num_resolutions = len(self.in_out)


class UnetDown(UnetBase):
    def __init__(self, dim:int=8, in_channels:int=3, out_channels:int=3, n_downsamples:int=1, n_groups:int=1, dropout:float=0):
        super().__init__(dim, in_channels, out_channels, n_downsamples, n_groups, dropout)
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (self.num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=None, groups=self.n_groups, dropout=self.dropout),
                ResnetBlock(dim_out, dim_out, time_emb_dim=None, groups=self.n_groups, dropout=self.dropout),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

    def forward(self, x:tensor) -> tensor:
        # h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, None)
            x = resnet2(x, None)
            x = attn(x)
            # h.append(x)
            x = downsample(x)
        # return x, h
        return x


class UnetUp(UnetBase):
    def __init__(self, dim:int=8, in_channels:int=3, out_channels:int=3, n_downsamples:int=1, n_groups:int=1, dropout:float=0):
        super().__init__(dim, in_channels, out_channels, n_downsamples, n_groups, dropout)
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out[1:])):
            is_last = ind >= (self.num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out, dim_in, time_emb_dim=None, groups=self.n_groups, dropout=self.dropout), #dim_out * 2
                ResnetBlock(dim_in, dim_in, time_emb_dim=None, groups=self.n_groups, dropout=self.dropout), 
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=self.n_groups),
            nn.Conv2d(dim, in_channels, 1)
        )

    def forward(self, x:tensor) -> tensor: #, h=None
        for resnet, resnet2, attn, upsample in self.ups:
            # x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, None)
            x = resnet2(x, None)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)
