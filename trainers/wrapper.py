import torch
from operator import mul
from functools import reduce
from utils import get_dataloader, \
    get_color_channels, seed_everything
from models import Unet, DDPM, \
    DownsampleDDPM, DownsampleDDPMAutoencoder
from .trainer_ddpm import TrainerDDPM, TrainerDownsampleDDPM

def setup_trainer(config:dict, mute:bool, data_root:str, wandb_project:str='tmp', seed:int=None):
    """Instantiate a trainer for a model specified by the config dict"""
    # fix seed
    seed_everything(seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get DataLoaders
    train_loader, val_loader = get_dataloader(config, device, True, data_root, config['val_split'])

    # set channels
    color_channels = get_color_channels(config['dataset'])

    # Get shape of input datapoints
    x_shape = [color_channels, config['image_size'], config['image_size']]

    # get flatten shape of input data
    x_dim = reduce(mul, x_shape, 1)

    # instantiate model and trainer for specified model and dataset
    train_args = [train_loader, val_loader, device, wandb_project, mute, color_channels]
    if config['model'] == 'ddpm':
        print('Instantiating DDPM')
        config['unet_in'] = color_channels
        latent_model = Unet(config)
        model = DDPM(config, latent_model, device, color_channels)
        trainer = TrainerDDPM(config, model, *train_args)
    elif config['model'] == 'dddpm':
        print('Instantiating DownsampledDDPM')
        latent_model = Unet(config)
        if config['ae_loss']:
            model = DownsampleDDPMAutoencoder(config, latent_model, device, color_channels)
        else:
            model = DownsampleDDPM(config, latent_model, device, color_channels)
        trainer = TrainerDownsampleDDPM(config, model, *train_args)
    else: 
        raise NotImplementedError('Specified model not implemented.')
    config['model_size'] = sum(p.numel() for p in model.parameters())
    return trainer, config
