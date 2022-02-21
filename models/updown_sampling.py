import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


def get_channels_list(channels:int, n_downsamples:int, upsample:bool=False) -> list:
    """Returns a list of tuples (in_channels, out_channels) for each stage in the convolutional network."""
    range_ = np.arange(n_downsamples) + 1
    if upsample:
        return [
            (channels * np.power(2, i), channels * np.power(2, i-1)) 
            for i in range_
        ][::-1]
    return [
        (channels * np.power(2, i-1), channels * np.power(2, i)) 
        for i in range_
    ]


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
    def __init__(self, channels:int, n_downsamples:int=1, upsample:bool=False):
        super().__init__()
        self.channel_list = get_channels_list(channels, n_downsamples, upsample=upsample)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleDownConv(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, False)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.append(
                get_3x3(in_, out_, stride=2, padding_mode=padding_mode)
            )
        self.conv = nn.Sequential(*conv_list)


class SimpleUpConv(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, True)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.append(
                get_4x4_transpose(in_, out_)
            )
        self.conv = nn.Sequential(*conv_list)


class SimpleDownConvPlus(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, False)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.extend([
                get_3x3(in_, in_, padding_mode=padding_mode),
                nn.ReLU(),
                get_3x3(in_, out_, stride=2, padding_mode=padding_mode)
            ])
        self.conv = nn.Sequential(*conv_list)


class SimpleUpConvPlus(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, True)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.extend([
                get_4x4_transpose(in_, out_),
                get_3x3(out_, out_, padding_mode=padding_mode),
                nn.ReLU(),
            ])
        conv_list.append(get_1x1(out_, out_))
        self.conv = nn.Sequential(*conv_list)


class UnetDownConv(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, False)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.extend([
                get_3x3(in_, in_, padding_mode=padding_mode),
                nn.ReLU(),
                get_3x3(in_, in_, padding_mode=padding_mode),
                nn.ReLU(),
                get_3x3(in_, out_, stride=2, padding_mode=padding_mode)
            ])


class UnetUpConv(BaseConv):
    def __init__(self, channels:int, n_downsamples:int=1, padding_mode:str='zeros'):
        super().__init__(channels, n_downsamples, True)
        conv_list = []
        for in_, out_ in self.channel_list:
            conv_list.extend([
                get_4x4_transpose(in_, out_),
                get_3x3(out_, out_, padding_mode=padding_mode),
                nn.ReLU(),
                get_3x3(out_, out_, padding_mode=padding_mode),
                nn.ReLU(),
            ])
        conv_list.append(get_1x1(out_, out_))
        self.conv = nn.Sequential(*conv_list)


# class ConvResBlock(nn.Module):
#     def __init__(self, in_channels:int, n_downsamples:int=1, down_rate:bool=False, up_rate:bool=False, dropout:bool=False, residual:bool=True):
#         super().__init__()
#         self.down_rate = down_rate
#         self.up_rate = up_rate
#         self.dropout = dropout
#         self.residual = residual # set to false for start/end

#         self.c1 = get_1x1(in_channels, middle_channels)
#         self.c2 = get_3x3(middle_channels, middle_channels)
#         self.c3 = get_3x3(middle_channels, middle_channels)
#         self.c4 = get_1x1(middle_channels, out_channels)
        
#         if dropout:
#             self.drop = nn.Dropout2d(p=0.1)

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         x_hat = self.c1(F.gelu(x))
#         x_hat = self.c2(F.gelu(x_hat))
#         x_hat = self.c3(F.gelu(x_hat))
#         x_hat = self.c4(F.gelu(x_hat))
#         if self.dropout:
#             x_hat = self.drop(x_hat)
        
#         out = x + x_hat if self.residual else x_hat
#         if self.down_rate:
#             out = F.avg_pool2d(out, kernel_size=2, stride=2)
#         if self.up_rate:
#             out = F.interpolate(out, scale_factor=2)
#         return out


def get_upsampling(mode:str, shape:tuple, n_downsamples:int=1, interpolate_mode:str=None):
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
        size = (shape[1], shape[2])
        return get_interpolate(size, interpolate_mode)
    elif mode == 'convolutional':
        return SimpleUpConv(in_channels, n_downsamples)
    elif mode == 'convolutional_plus':
        return SimpleUpConvPlus(in_channels, n_downsamples)
    elif mode == 'convolutional_unet':
        return UnetUpConv(in_channels, n_downsamples)
    # elif mode == 'convolutional_res':
    #     return ConvResBlock(*chans, up_rate=True)
    else:
        raise NotImplementedError(f'Upsampling method for "{mode}" not implemented!')


def get_downsampling(mode:str, shape:tuple, n_downsamples:int=1, interpolate_mode:str=None):
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
        scale = np.power(2, n_downsamples).astype(int)
        size = (int(shape[1] / scale), int(shape[2] / scale))
        assert size[0] % 2 == 0, 'result from downsampling should have even dimensions.'
        return get_interpolate(size, interpolate_mode)
    elif mode == 'convolutional':
        return SimpleDownConv(in_channels, n_downsamples)
    elif mode == 'convolutional_plus':
        return SimpleDownConvPlus(in_channels, n_downsamples)
    elif mode == 'convolutional_unet':
        return UnetDownConv(in_channels, n_downsamples)
    # elif mode == 'convolutional_res':
    #     return ConvResBlock(*chans, down_rate=True)
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
