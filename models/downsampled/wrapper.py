import numpy as np
from models.downsampled.convblocks import get_interpolate, \
    SimpleUpConv, SimpleDownConv, ConvResBlock, UnetUp, UnetDown


def get_upsampling(config:dict, shape:tuple):
    """
    Returns upsampling function.
        config (dict):          ....
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    mode = config['mode']
    n_down = config['n_downsamples']
    dim = config['unet_in']
    
    if mode == 'deterministic':
        size = (shape[1], shape[2])
        return get_interpolate(size)
    elif mode == 'convolutional':
        return SimpleUpConv(dim, in_channels, n_down)
    elif mode == 'convolutional_unet':
        return UnetUp(dim, in_channels, n_down)
    elif mode == 'convolutional_res':
        return ConvResBlock(dim, in_channels, upsample=True)
    else:
        raise NotImplementedError(f'Upsampling method for "{mode}" not implemented!')


def get_downsampling(config:dict, shape:tuple):
    """
    Returns downsampling function.
        config (dict):          ....
        shape (tuple):          The shape of the data without batch size.
                                Such that (C x H x W), where H == W.
    """
    assert shape[1] == shape[2]
    assert shape[0] == 1 or shape[0] == 3
    in_channels = shape[0]
    mode = config['mode']
    n_down = config['n_downsamples']
    dim = config['unet_in']
    
    if mode == 'deterministic':
        scale = np.power(2, n_down).astype(int)
        size = (int(shape[1] / scale), int(shape[2] / scale))
        assert size[0] % 2 == 0, 'result from downsampling should have even dimensions.'
        return get_interpolate(size)
    elif mode == 'convolutional':
        return SimpleDownConv(dim, in_channels, n_down)
    elif mode == 'convolutional_unet':
        return UnetDown(dim, in_channels, n_down)
    elif mode == 'convolutional_res':
        return ConvResBlock(in_channels, dim, upsample=False)
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
