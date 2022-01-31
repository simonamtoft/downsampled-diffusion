import os
import torch
import json
from functools import reduce
from operator import mul

from utils import get_dataloader, get_color_channels, \
    get_args, DATASETS
from trainers import TrainerDDPM, TrainerDownsampleDDPM, \
    TrainerDRAW, TrainerVAE
from models import MODEL_NAMES, Unet, DDPM, DownsampleDDPM, \
    DRAW, VariationalAutoencoder

# setup path to data root
DATA_ROOT = '../data/'

# define WANDB project name
WANDB_PROJECT = 'ddpm-test'

# standard config for every model
CONFIG = {
    'image_size': 32,
}

# specific model architecture config
CONFIG_MODEL = {
    'ddpm': {
        'unet_chan': 64,
        'unet_dims': (1, 2, 4, 8),
        'timesteps': 1000,
        'loss_type': 'l2',
        'lr': 2e-5,
        'batch_size': 32,
    },
    'draw': {
        'h_dim': 256,
        'z_dim': 32,    
        'T': 10,
        'batch_size': 128,
        'lr': 1e-3,
    },
    'vae': {
        'h_dim': [512, 256, 128, 64],
        'z_dim': 64,
        'batch_size': 128,
        'lr': 1e-3,
        'as_beta': True,
    }
}


def modify_config(config, model_config):
    for key, value in model_config.items():
        config[key] = value
    return config


def get_trainer(config:dict, mute:bool):
    """Instantiate a trainer for a model specified by the config dict"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get DataLoaders
    train_loader, val_loader = get_dataloader(config, device, True, DATA_ROOT, 0.15)
    
    # set channels
    color_channels = get_color_channels(config['dataset'])
    
    # Get shape of input datapoints
    x_shape = [color_channels, config['image_size'], config['image_size']]
    
    # get flatten shape of input data
    x_dim = reduce(mul, x_shape, 1)    
    
    # instantiate model and trainer for specified model and dataset
    if config['model'] == 'ddpm':
        # instantiate latent model
        unet_in = config['unet_chan'] if config['downsample'] else color_channels
        latent_model = Unet(
            dim=config['unet_chan'],
            in_channels=unet_in,
            dim_mults=config['unet_dims'],
        )

        # instantiate diffusion model and trainer
        if config['downsample']:
            model = DownsampleDDPM(config, latent_model, device, color_channels)
            trainer = TrainerDownsampleDDPM(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute)
        else:
            model = DDPM(config, latent_model, device, color_channels)
            trainer = TrainerDDPM(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute)
    elif config['model'] == 'draw':
        model = DRAW(config, x_dim)
        trainer = TrainerDRAW(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute)
    elif config['model'] == 'vae':
        model = VariationalAutoencoder(config, x_dim)
        trainer = TrainerVAE(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute)
    else: 
        raise NotImplementedError('Specified model not implemented.')
    return trainer


if __name__ == '__main__':
    # Get CLI arguments
    config, args = get_args(CONFIG, DATASETS, MODEL_NAMES)
    
    # add specific model architecture stuff to config
    config = modify_config(config, CONFIG_MODEL[config['model']])
    
    # setup model and trainer
    trainer = get_trainer(config, args['mute'])
    
    # print out train configuration
    print('\nTraining configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    
    # train model
    losses = trainer.train()
    
    # Store losses
    trainer.save_losses(losses)
    
    print("train.py script finished!")
