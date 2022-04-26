import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm

from models import Unet, DDPM, DownsampleDDPM
from utils import get_color_channels, fix_samples, \
    CHECKPOINT_DIR, SAMPLE_DIR

device = 'cuda'
saved_model = 'chq_x3_AE_latent'
fid_samples = 10000
batch_size = 192
sample_every = 1

# load saved state dict of model and its config file
save_data = torch.load(os.path.join(CHECKPOINT_DIR, f'{saved_model}.pt'))
config = save_data['config']
if 'ema_model' in save_data:
    model_state_dict = save_data['ema_model']
else:
    model_state_dict = save_data['model']
config['batch_size'] = batch_size

if 'force_latent' not in config:
    config['force_latent'] = False
if 'rnd_flip' not in config:
    config['rnd_flip'] = False

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
time_start = time.time()
n_batches = int(np.ceil(fid_samples/config['batch_size']))
for i in tqdm(range(n_batches), desc='sampling from model'):
    samples = model.sample(config['batch_size'], sample_every)
    if config['model'] == 'dddpm':
        samples, latent_samples = samples[0], samples[1]
        sample_list.append(fix_samples(samples))
        latent_list.append(fix_samples(latent_samples))
    else:
        sample_list.append(fix_samples(samples))
sampling_time = time.time() - time_start

# print stats
print(f'Using batch size {config["batch_size"]}')
print(f'Total time: {sampling_time}')
print(f'Sample time: {sampling_time/fid_samples}')
print(f'Batch time: {sampling_time/n_batches}')

# Input space samples
save_path = os.path.join(SAMPLE_DIR, saved_model)
np.save(save_path, sample_list, allow_pickle=False)
print(f'Samples saved to {save_path}')

# Latent samples
if config['model'] == 'dddpm':
    save_path = os.path.join(SAMPLE_DIR, f'{saved_model}_latent')
    np.save(save_path, latent_list, allow_pickle=False)
    print(f'Latent samples saved to {save_path}')
