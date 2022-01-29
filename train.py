import os
import torch

from utils import get_dataloader, get_color_channels
from trainers import TrainerDDPM, TrainerDownsampleDDPM
from train_config import DATA_ROOT, \
    WANDB_PROJECT, CONFIG
from models import Unet, DDPM, DownsampleDDPM

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get DataLoaders
train_loader, _ = get_dataloader(CONFIG, data_root=DATA_ROOT, device=device)

# set channels
color_channels = get_color_channels(CONFIG['dataset'])
unet_in = CONFIG['unet_chan'] if CONFIG['downsample'] else color_channels

# instantiate latent model
latent_model = Unet(
    dim=CONFIG['unet_chan'],
    in_channels=unet_in,
    dim_mults=CONFIG['unet_dims'],
)

# instantiate diffusion model and trainer
if CONFIG['downsample']:
    model = DownsampleDDPM(CONFIG, latent_model, device, color_channels)
    trainer = TrainerDownsampleDDPM(CONFIG, model, train_loader, device=device, wandb_name=WANDB_PROJECT)
else:
    model = DDPM(CONFIG, latent_model, device, color_channels)
    trainer = TrainerDDPM(CONFIG, model, train_loader, device=device, wandb_name=WANDB_PROJECT)

# Train model
trainer.train()
