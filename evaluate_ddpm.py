import os
import json
import torch
import wandb
from models import Unet, DDPM, DownsampleDDPM
from utils import get_dataloader, get_color_channels, \
    create_generator_ddpm, create_generator_dddpm, \
    compute_fid, compute_vlb


WANDB_PROJECT = 'ddpm-test'
FID_DIR = './results/fid_stats'
DATA_ROOT = '../data'
device = 'cuda'
# saved_model = 'checkpoint_ddpm_vlb_paper_4.pt'
saved_model = 'ddpm_simple_paper_5_final.pt'
fid_samples = 50000

# load saved state dict of model and its config file
save_data = torch.load(f'./results/checkpoints/{saved_model}')
config = save_data['config']
model_state_dict = save_data['model']
config['loss_flat'] = 'mean'

# get data
train_loader, val_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=True)
test_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=False)
color_channels = get_color_channels(config['dataset'])

# Setup DDPM model
latent_model = Unet(config)
if config['model'] == 'ddpm':
    model = DDPM(config, latent_model, device, color_channels)
    g = create_generator_ddpm(model, config['batch_size'], fid_samples)
elif config['model'] == 'dddpm':
    model = DownsampleDDPM(config, latent_model, device, color_channels)
    g = create_generator_dddpm(model, config['batch_size'], fid_samples)
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()


### COMPUTE METRICS ###
metrics = {}
wandb.init(project=WANDB_PROJECT, config=config, resume='allow', id=config['wandb_id'])
print(f'\nEvaluating the checkpoint {saved_model} with configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')
# metrics['vlb'] = compute_vlb(model, test_loader, device)
metrics['fid'] = compute_fid(config, g, fid_samples, FID_DIR, device)
# metrics['precision'] = 
# metrics['recall'] = 


# Display resulting metrics
print('\nResults:')
print(json.dumps(metrics, sort_keys=False, indent=4) + '\n')
