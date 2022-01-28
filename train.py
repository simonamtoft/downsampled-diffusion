import os
import torch

from utils import get_dataloader
from trainers import TrainerDDPM
from train_config import DATA_ROOT, \
    WANDB_PROJECT, CONFIG
from models import Unet, GaussianDiffusion, \
    GaussianDiffusionSmall

# Set device
cuda = torch.cuda.is_available()
CONFIG['device'] = 'cuda' if cuda else 'cpu'

# get DataLoaders
train_loader, _ = get_dataloader(CONFIG, data_root=DATA_ROOT)
color_channels = 1 if CONFIG['dataset'] == 'mnist' else 3

# instantiate latent model
latent_model = Unet(
    dim=CONFIG['unet_chan'],
    channels=color_channels,
    dim_mults=CONFIG['unet_dims'],
)

# instantiate diffusion model
model = GaussianDiffusion(latent_model, channels=color_channels, **CONFIG)

# Configure training for DDPM
trainer = TrainerDDPM(CONFIG, model, train_loader, device=device, wandb_name='')

# Train model
trainer.train()
