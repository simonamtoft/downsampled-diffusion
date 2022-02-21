import os
import torch
import json
import random
import numpy as np
from functools import reduce
from operator import mul

from utils import get_dataloader, get_color_channels, \
    get_args, DATASETS
from trainers import TrainerDDPM, TrainerDownsampleDDPM, \
    TrainerDRAW, TrainerVAE
from models import MODEL_NAMES, Unet, DDPM, DownsampleDDPM, \
    DRAW, VariationalAutoencoder, LadderVariationalAutoencoder, \
    DownsampleDDPMAutoencoder

# setup path to data root
DATA_ROOT = '../data/'

# define WANDB project name
WANDB_PROJECT = 'ddpm-test'

# standard config for every model
CONFIG = {
    'lr': 1e-3, # standard for VAE and DRAW models
}

# specific model architecture config
CONFIG_MODEL = {
    'ddpm': {
        'lr': 2e-5,
        'unet_chan': 64,
        'unet_dims': (1, 2, 4), #, 8
        'T': 1000,
        # simple, vlb, hybrid
        'loss_type': 'simple',
        # linear, cosine, sqrt_linear, sqrt
        'beta_schedule': 'cosine',
    },
    # for mnist
    # 'draw': {
    #     'h_dim': 256,
    #     'z_dim': 32,    
    #     'T': 10,
    # },
    # for Cifar
    'draw': {
        'h_dim': 400,
        'z_dim': 200,
        'T': 16,
    },
    'vae': {
        'h_dim': [512, 256, 128, 64],
        'z_dim': 64,
        'as_beta': True,
    },
    'lvae': {
        # Bottom to top
        'h_dim': [512, 256, 256],
        'z_dim': [64, 32, 32],
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
        if config['n_downsamples'] == 0:          
            latent_model = Unet(
                dim=config['unet_chan'],
                in_channels=color_channels,
                dim_mults=config['unet_dims'],
            )
            model = DDPM(config, latent_model, device, color_channels)
            trainer = TrainerDDPM(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute, color_channels)
        else:
            # set mode of down-up sampling architecture.
            # options: 
            #   deterministic
            #   convolutional
            #   convolutional_plus
            #   convolutional_unet
            config['mode'] = 'convolutional'
            
            # set padding mode for convolutional down-up sampling
            # options: zeros, reflect, replicate, circular
            pad_mode = 'zeros'
            
            # define loss mode
            # if true, recon loss is computed directly by
            # z = downsample(x), x_hat = upsample(z), l_recon = L2(x, x_hat)
            config['ae_loss'] = False
            
            # instantiate latent model
            unet_in = color_channels
            if 'convolutional' in config['mode']:
                unet_in *= np.power(2, config['n_downsamples']).astype(int)
                config['padding_mode'] = pad_mode
            latent_model = Unet(
                dim=config['unet_chan'],
                in_channels=unet_in,
                dim_mults=config['unet_dims'],
            )
            
            # instantiate DDPM
            if config['ae_loss']:
                model = DownsampleDDPMAutoencoder(config, latent_model, device, color_channels)
            else:
                model = DownsampleDDPM(config, latent_model, device, color_channels)
            trainer = TrainerDownsampleDDPM(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute, color_channels)
    elif config['model'] == 'draw':
        model = DRAW(config, x_dim)
        trainer = TrainerDRAW(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute, color_channels)
    elif config['model'] == 'vae':
        model = VariationalAutoencoder(config, x_dim)
        trainer = TrainerVAE(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute, color_channels)
    elif config['model'] == 'lvae':
        model = LadderVariationalAutoencoder(config, x_dim)
        trainer = TrainerVAE(config, model, train_loader, val_loader, device, WANDB_PROJECT, mute, color_channels)
    else: 
        raise NotImplementedError('Specified model not implemented.')
    return trainer


def seed_everything(seed:int) -> None:
    """Sets the random seed for all the different random calculations."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # fix seed
    seed_everything(0)
    
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
    # trainer.save_losses(losses)
    
    print("train.py script finished!")
