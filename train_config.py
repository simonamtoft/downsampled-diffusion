# setup path to data root
DATA_ROOT = '../data/'

# define WANDB project name
WANDB_PROJECT = 'ddpm-test'

# define config
CONFIG = {
    # dataset
    'dataset': 'mnist',
    'image_size': 32,
    'batch_size': 32,

    # U-Net stuff
    'unet_chan': 64,
    'unet_dims': (1, 2, 4, 8),

    # ddpm architecture
    'model': 'ddpm',
    'downsample': True,
    'timesteps': 1000,      # how deep the ddpm is

    # training parameters 
    # 'n_steps': 700000,
    'n_steps': 300000,      # should be fine for mnist
    'loss_type': 'l1',      # l1 or l2
    'lr': 2e-5,
}