import json
from tqdm import tqdm
import torch
import numpy as np

from models import Unet, DDPM, DownsampleDDPM
from utils import get_color_channels, fix_samples

device = 'cuda'
saved_model = 'chq_x3_t100_d_3'
fid_samples = 50000

# load saved state dict of model and its config file
save_data = torch.load(f'./results/checkpoints/{saved_model}.pt')
config = save_data['config']
if 'ema_model' in save_data:
    model_state_dict = save_data['ema_model']
else:
    model_state_dict = save_data['model']

# Setup DDPM model
latent_model = Unet(config)
color_channels = get_color_channels(config['dataset'])
if config['model'] == 'ddpm':
    model = DDPM(config, latent_model, device, color_channels)
elif config['model'] == 'dddpm':
    model = DownsampleDDPM(config, latent_model, device, color_channels)
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

# compute and save samples
print(f'\nGenerating samples from checkpoint {saved_model} with configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')
sample_list = []
latent_list = []
for i in tqdm(range(int(np.ceil(fid_samples/config['batch_size']))), desc='sampling from model'):
    samples = model.sample(config['batch_size'])
    if config['model'] == 'dddpm':
        sample_list.append(fix_samples(samples[0]))
        latent_list.append(fix_samples(samples[1]))
        samples, latent_samples = samples[0], samples[1]
    else:
        sample_list.append(fix_samples(samples))
np.save(f'./results/samples/{saved_model}', sample_list)
if config['model'] == 'dddpm':
    np.save(f'./results/samples/{saved_model}_latent', latent_list)
    print(f'Latent samples saved to ./results/samples/{saved_model}_latent')
print(f'Samples saved to ./results/samples/{saved_model}')