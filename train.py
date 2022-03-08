import json

from utils import DATASETS, get_args, modify_config
from models import MODEL_NAMES
from trainers import setup_trainer

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
        'lr': 2e-4,                 # iddpm paper: 2e-4 for 32x32, 2e-5 for 256x256
        'unet_chan': 128,           # iddpm paper: 128
        'unet_dims': (1, 2, 2, 2),  # iddpm paper: (1, 2, 2, 2)
        'unet_dropout': 0.1,        # iddpm paper: 0.1 for linear, 0.3 for cosine
        'T': 1000,                  # iddpm paper: 4000, ddpm: 1000
        'loss_type': 'simple',      # simple, vlb, hybrid
        'beta_schedule': 'linear',  # linear, cosine, sqrt_linear, sqrt
        # 'ema': 0.9999,              # iddpm + ddpm: 0.9999
    },
    'dddpm': {
        # set mode of down-up sampling architecture.
        # options: 
        #   deterministic
        #   convolutional
        #   convolutional_unet
        #   convolutional_res
        'mode': 'convolutional',
        # define loss mode for reconstruction
        # if true, recon loss is computed directly by
        # z = downsample(x), x_hat = upsample(z), l_recon = L2(x, x_hat)
        'ae_loss': False,
        't_rec_max': 500,
        'unet_in': 4,
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


if __name__ == '__main__':
    # Get CLI arguments
    config, mute = get_args(CONFIG, DATASETS, MODEL_NAMES)

    # add specific model architecture stuff to config
    config = modify_config(config, CONFIG_MODEL[config['model']])
    if config['model'] == 'ddpm':
        if config['n_downsamples'] > 0:
            config['model'] = 'dddpm'
            config = modify_config(config, CONFIG_MODEL['dddpm'])
    
    # setup model and trainer
    trainer, config = setup_trainer(config, mute, DATA_ROOT, WANDB_PROJECT, RES_FOLDER, 0)
    
    # print out train configuration
    print('\nTraining configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    
    # train model
    losses = trainer.train()
    
    # Store losses
    trainer.save_losses(losses)
    
    print("train.py script finished!")
