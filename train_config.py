# setup path to data root
DATA_ROOT = '../data/'

# define WANDB project name
WANDB_PROJECT = 'ddpm-test'

# define config
CONFIG = {
    # dataset
    'dataset': 'mnist',
    'image_size': 28,
    'batch_size': 32,

    # U-Net stuff
    'unet_chan': 64,
    'unet_dims': (1, 2),

    # ddpm architecture
    'model': 'ddpm_small',
    'timesteps': 1000,      # how deep the ddpm is

    # training parameters 
    'n_steps': 700000,
    'loss_type': 'l1',      # l1 or l2
    'lr': 2e-5,
}