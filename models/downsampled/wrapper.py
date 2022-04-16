import numpy as np
from .convblocks import get_interpolate, \
    SimpleUpConv, SimpleDownConv, ConvResNet


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
    mode = config['u_mode']
    dim = config['d_chans']
    out_channels = config['unet_in']
    dropout = config['d_dropout']
    n_down = config['n_downsamples']

    if mode == 'deterministic':
        size = (shape[1], shape[2])
        return get_interpolate(size)
    elif mode == 'convolutional':
        return SimpleUpConv(out_channels, in_channels, n_down)
    elif mode == 'convolutional_res':
        return ConvResNet(dim, out_channels, in_channels, n_down, upsample=True, dropout=dropout, n_blocks=config['u_n_blocks'])
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
    mode = config['d_mode']
    dim = config['d_chans']
    out_channels = config['unet_in']
    dropout = config['d_dropout']
    n_down = config['n_downsamples']
    
    if mode == 'deterministic':
        scale = np.power(2, n_down).astype(int)
        size = (int(shape[1] / scale), int(shape[2] / scale))
        assert size[0] % 2 == 0, 'result from downsampling should have even dimensions.'
        return get_interpolate(size)
    elif mode == 'convolutional':
        return SimpleDownConv(out_channels, in_channels, n_down)
    elif mode == 'convolutional_res':
        return ConvResNet(dim, in_channels, out_channels, n_down, upsample=False, dropout=dropout, n_blocks=config['d_n_blocks'])
    else:
        raise NotImplementedError(f'Downsampling method for "{mode}" not implemented!')
