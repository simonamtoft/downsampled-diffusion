from models import Unet, GaussianDiffusionSmall
from trainers import DDPM_Trainer
from utils import get_dataloader
import torch
import os

from train_config import DATA_ROOT, config

# mute weight and biases prints
os.environ["WANDB_SILENT"] = "true"

# Set device
cuda = torch.cuda.is_available()
config['device'] = 'cuda' if cuda else 'cpu'

# select number of input channels dependent on data
# (number of color channels of the images in the dataset)
if config['dataset'] == 'mnist':
    color_channels = 1
else:
    color_channels = 3

# Instantiate Latent Model
unet_chans = color_channels*2 # input number of channels for unet
print(f"U-net with {config['unet_dims']}")
model = Unet(
    dim=64,
    channels=unet_chans,
    dim_mults=config['unet_dims'],
).to(config['device'])

# Instantiate Diffusion Model
model_args = {
    'image_size': config['image_size'], 
    'timesteps': config['timesteps'], 
    'loss_type': config['loss_type'], 
    'channels': color_channels}
diffusion = GaussianDiffusionSmall(model, unet_chans, **model_args).to(config['device'])

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
