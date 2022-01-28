import os
import torch

from utils import get_dataloader, get_color_channels
from trainers import TrainerDDPM
from train_config import DATA_ROOT, \
    WANDB_PROJECT, CONFIG
from models import Unet, GaussianDiffusion, \
    GaussianDiffusionSmall

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get DataLoaders
train_loader, _ = get_dataloader(CONFIG, data_root=DATA_ROOT, device=device)
color_channels = get_color_channels(CONFIG['dataset'])

# instantiate latent model
latent_model = Unet(
    dim=CONFIG['unet_chan'],
    channels=color_channels,
    dim_mults=CONFIG['unet_dims'],
)

# instantiate diffusion model
model = GaussianDiffusion(CONFIG, latent_model, device, color_channels=1)

# Configure training for DDPM
trainer = TrainerDDPM(config=CONFIG, model=model, train_loader=train_loader, device=device, wandb_name='')

# Train model
trainer.train()
