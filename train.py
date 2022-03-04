import json
import torch
from functools import reduce
from operator import mul

from utils import DATASETS, get_dataloader, \
    get_color_channels, get_args, modify_config, \
    seed_everything
from trainers import TrainerDDPM, TrainerDownsampleDDPM, \
    TrainerDRAW, TrainerVAE
from models import MODEL_NAMES, Unet, DDPM, \
    DownsampleDDPM, DownsampleDDPMAutoencoder, \
    DRAW, VariationalAutoencoder, LadderVariationalAutoencoder
# from models.unet.unet import Unet
# from models.diffusion.ddpm import DDPM
# from models.diffusion.dddpm import DownsampleDDPM, DownsampleDDPMAutoencoder
# from models.variational.draw import DRAW
# from models.variational.vae import VariationalAutoencoder
# from models.variational.lvae import LadderVariationalAutoencoder

# setup path to data root
DATA_ROOT = '../data/'
RES_FOLDER = './results'

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
        'unet_dims': (1, 2, 4, 8),
        'T': 1000,
        'loss_type': 'simple',      # simple, vlb, hybrid
        'beta_schedule': 'cosine',  # linear, cosine, sqrt_linear, sqrt
    },
    'dddpm': {
        # set mode of down-up sampling architecture.
        # options: 
        #   deterministic
        #   convolutional
        #   convolutional_unet
        #   convolutional_res
        'mode': 'convolutional_unet',
        # define loss mode for reconstruction
        # if true, recon loss is computed directly by
        # z = downsample(x), x_hat = upsample(z), l_recon = L2(x, x_hat)
        'ae_loss': False,
        't_rec_max': 500,
        'unet_in': 2,
    },
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
        'h_dim': [512, 256, 256],
        'z_dim': [64, 32, 32],
        'as_beta': True,
    }
}


def setup_trainer(config:dict, mute:bool, data_root:str, wandb_project:str='tmp', res_folder='./tmp', seed:int=None, val_split:float=0.15):
    """Instantiate a trainer for a model specified by the config dict"""
    # fix seed
    seed_everything(seed)
    
    # add specific model architecture stuff to config
    config = modify_config(config, CONFIG_MODEL[config['model']])
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get DataLoaders
    train_loader, val_loader = get_dataloader(config, device, True, data_root, val_split)
    
    # set channels
    color_channels = get_color_channels(config['dataset'])
    
    # Get shape of input datapoints
    x_shape = [color_channels, config['image_size'], config['image_size']]
    
    # get flatten shape of input data
    x_dim = reduce(mul, x_shape, 1)
    
    # instantiate model and trainer for specified model and dataset
    train_args = [train_loader, val_loader, device, wandb_project, mute, res_folder, color_channels]
    if config['model'] == 'ddpm':
        if config['n_downsamples'] == 0:
            config['unet_in'] = color_channels
            latent_model = Unet(config)
            model = DDPM(config, latent_model, device, color_channels)
            trainer = TrainerDDPM(config, model, *train_args)
        else:
            config = modify_config(config, CONFIG_MODEL['dddpm'])
            latent_model = Unet(config)
            if config['ae_loss']:
                model = DownsampleDDPMAutoencoder(config, latent_model, device, color_channels)
            else:
                model = DownsampleDDPM(config, latent_model, device, color_channels)
            trainer = TrainerDownsampleDDPM(config, model, *train_args)
    elif config['model'] == 'draw':
        model = DRAW(config, x_dim)
        trainer = TrainerDRAW(config, model, *train_args)
    elif config['model'] == 'vae':
        model = VariationalAutoencoder(config, x_dim)
        trainer = TrainerVAE(config, model, *train_args)
    elif config['model'] == 'lvae':
        model = LadderVariationalAutoencoder(config, x_dim)
        trainer = TrainerVAE(config, model, *train_args)
    else: 
        raise NotImplementedError('Specified model not implemented.')
    return trainer, config


if __name__ == '__main__':    
    # Get CLI arguments
    config, args = get_args(CONFIG, DATASETS, MODEL_NAMES)
    
    # setup model and trainer
    trainer, config = setup_trainer(config, args['mute'], DATA_ROOT, WANDB_PROJECT, RES_FOLDER, 0)
    
    # print out train configuration
    print('\nTraining configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    
    # train model
    losses = trainer.train()
    
    # Store losses
    # trainer.save_losses(losses)
    
    print("train.py script finished!")
