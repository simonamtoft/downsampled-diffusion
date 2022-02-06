from functools import partial
import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bicubic'


def get_interpolate(size:tuple, mode:str, align:bool=True):
    """Returns the function F.interpolate but with size, mode and align_corners set."""
    align = None if mode == 'nearest' else align
    return partial(
        F.interpolate,
        size=size,
        mode=mode,
        align_corners=align,
    )


def get_upsampling(mode:str, shape:tuple, interpolate_mode:str=None):
    """
    Returns upsampling function.
        mode (str):             The way to perform upsampling. 
                                'deterministic' uses F.interpolate
                                'convolutional' uses a trainable convolutional network
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
        interpolate_mode (str): Determines the mode used for interpolation.    
                                Only valid when mode = 'deterministic'.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    
    if mode == 'deterministic':
        mode = INTERPOLATE_MODE if interpolate_mode is None else interpolate_mode
        size = (shape[1], shape[2])
        return get_interpolate(size, mode)
    elif mode == 'convolutional':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
        )
    else:
        raise NotImplementedError(f'Upsampling method for "{mode}" not implemented!')


def get_downsampling(mode:str, shape:tuple, scale:int, interpolate_mode:str=None):
    """
    Returns downsampling function.
        mode (str):             The way to perform upsampling. 
                                'deterministic' uses F.interpolate
                                'convolutional' uses a trainable convolutional network
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
        scale (int):            How many times the result should be downsampled.
                                E.g. scale=2 on a (3x32x32) image results in a (3x16x16) image.
        interpolate_mode (str): Determines the mode used for interpolation.    
                                Only valid when mode = 'deterministic'.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    
    if mode == 'deterministic':
        mode = INTERPOLATE_MODE if interpolate_mode is None else interpolate_mode
        size = (int(shape[1] / scale), int(shape[2] / scale))
        return get_interpolate(size, mode)
    elif mode == 'convolutional':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels*2, kernel_size=2, padding=0, stride=2)
        )
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
