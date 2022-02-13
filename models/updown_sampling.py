import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

DEFAULT_INTERPOLATE_MODE = 'bicubic'

# default padding mode for 3-by-3 convolutions with padding
# options: zeros, reflect, replicate, circular
DEFAULT_PADDING_MODE = 'zeros'


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


def get_3x3(in_dim:int, out_dim:int, stride:int=1, padding:int=1, padding_mode:str=None):
    """Wrapper function to get_conv a 3-by-3 convolution."""
    pad_mode = DEFAULT_PADDING_MODE if padding_mode is None else padding_mode
    return nn.Conv2d(
        in_dim, out_dim, 
        kernel_size=3, 
        stride=stride, 
        padding=padding, 
        padding_mode=pad_mode
    )


def get_1x1(in_dim:int, out_dim:int):
    """Wrapper function to get_conv a 1-by-1 convolution."""
    return nn.Conv2d(
        in_dim, out_dim, 
        kernel_size=1, 
        stride=1, 
        padding=0,
    )


def get_4x4_transpose(in_dim:int, out_dim:int, stride:int=2, padding:int=1):
    """Wrapper function to get_conv a 4-by-4 transpose convolution."""
    return nn.ConvTranspose2d(
        in_dim, out_dim, 
        kernel_size=4, 
        stride=stride, 
        padding=padding
    )


class SimpleDownConv(nn.Module):
    def __init__(self, channels:int, padding_mode:str=None):
        super().__init__()
        self.conv = get_3x3(channels, channels*2, stride=2, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)


class SimpleUpConv(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = get_4x4_transpose(channels*2, channels)

    def forward(self, x):
        return self.conv(x)


class SimpleDownConvPlus(nn.Module):
    def __init__(self, channels:int, padding_mode:str=None):
        super().__init__()
        self.conv = nn.Sequential(
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_3x3(channels, channels*2, stride=2, padding_mode=padding_mode)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUpConvPlus(nn.Module):
    def __init__(self, channels:int, padding_mode:str=None):
        super().__init__()
        self.conv = nn.Sequential(
            get_4x4_transpose(channels*2, channels),
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_1x1(channels, channels)
        )

    def forward(self, x):
        return self.conv(x)


class UnetDownConv(nn.Module):
    def __init__(self, channels:int, padding_mode:str=None):
        super().__init__()
        self.conv = nn.Sequential(
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_3x3(channels, channels*2, stride=2, padding_mode=padding_mode)
        )

    def forward(self, x):
        return self.conv(x)


class UnetUpConv(nn.Module):
    def __init__(self, channels:int, padding_mode:str=None):
        super().__init__()
        self.conv = nn.Sequential(
            get_4x4_transpose(channels*2, channels),
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_3x3(channels, channels, padding_mode=padding_mode),
            nn.ReLU(),
            get_1x1(channels, channels)
        )

    def forward(self, x):
        return self.conv(x)


class ConvResBlock(nn.Module):
    def __init__(self, in_channels:int, middle_channels:int, out_channels:int, down_rate:bool=False, up_rate:bool=False, dropout:bool=False):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.dropout = dropout
        
        self.c1 = get_1x1(in_channels, middle_channels)
        self.c2 = get_3x3(middle_channels, middle_channels)
        self.c3 = get_3x3(middle_channels, middle_channels)
        self.c4 = get_1x1(middle_channels, out_channels)
        
        if dropout:
            self.drop = nn.Dropout2d(p=0.1)

    def forward(self, x):
        x_hat = self.c1(F.gelu(x))
        x_hat = self.c2(F.gelu(x_hat))
        x_hat = self.c3(F.gelu(x_hat))
        x_hat = self.c4(F.gelu(x_hat))
        if self.dropout:
            x_hat = self.drop(x_hat)

        if self.down_rate:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        if self.up_rate:
            out = F.interpolate(out, scale_factor=2)
        return out


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
    elif mode == 'convolutional_res':
        chans = [in_channels, in_channels, int(in_channels/2)]
        return ConvResBlock(*chans, up_rate=True)
    # elif mode == 'autoencoder':
    #     return AEUp(shape)
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
    elif mode == 'convolutional_res':
        chans = [in_channels, in_channels*2, in_channels*2]
        return ConvResBlock(*chans, down_rate=True)
    # elif mode == 'autoencoder':
    #     return AEDown(shape)
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
