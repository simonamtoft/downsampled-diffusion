from models import Unet, GaussianDiffusion, \
    GaussianDiffusionSmall
from trainers import DDPM_Trainer
from utils import get_dataloader
import torch
import os


# mute weight and biases prints
os.environ["WANDB_SILENT"] = "true"

# setup path to data root
DATA_ROOT = '../data/'

# define config
config = {
    'model': 'ddpm_small',
    'dataset': 'mnist',
    'image_size': 28,
    'batch_size': 32,
    'loss_type': 'l1', # l1 or l2
    'timesteps': 1000,
    'train_steps': 700000,
    'lr': 2e-5,
}

# Set device
cuda = torch.cuda.is_available()
config['device'] = 'cuda' if cuda else 'cpu'

# select number of input channels dependent on data
# (number of color channels of the images in the dataset)

if config['dataset'] == 'mnist':
    color_channels = 1
else:
    color_channels = 3

if config['model'] == 'ddpm_small':
    latent_channels = 2*config['image_size']
else:
    latent_channels = color_channels

# Define Latent Architecture
unet_dims = (1, 2) #, 4, 8
print(f"U-net with {unet_dims}")
model = Unet(
    dim=64,
    channels=latent_channels,
    dim_mults=unet_dims,
).to(config['device'])

# Define Diffusion Model
model_args = {'image_size': config['image_size'], 'timesteps': config['timesteps'], 'loss_type': config['loss_type'], 'channels': color_channels}
if config['model'] == 'ddpm_small':
    diffusion = GaussianDiffusionSmall(model, **model_args).to(config['device'])
else:
    diffusion = GaussianDiffusion(model, **model_args).to(config['device'])

# load in data
train_loader, val_loader = get_dataloader(config, data_root=DATA_ROOT)

# train
trainer = DDPM_Trainer(
    diffusion,
    dataloader = train_loader,
    train_batch_size = config['batch_size'],
    train_lr = config['lr'],
    train_num_steps = config['train_steps'],    # total training steps
    gradient_accumulate_every = 2,              # gradient accumulation steps
    ema_decay = 0.995,                          # exponential moving average decay
    fp16 = False,                               # turn on/off mixed precision training with apex
    config = config,
)
trainer.train()
