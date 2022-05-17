import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm

from models import Unet, DDPM, DownsampleDDPM
from utils import get_color_channels, fix_samples, \
    get_model_state_dict, \
    CHECKPOINT_DIR, SAMPLE_DIR

# checkpoint names
checkpoint_dDDPM = 'chq_x3_AE_act'

# other hyper parameters
device = 'cuda'
fid_samples = 192
batch_size = 192
early_stop = 11

# Load models from files
save_data_dDDPM = torch.load(os.path.join(CHECKPOINT_DIR, f'{checkpoint_dDDPM}.pt'))
config = save_data_dDDPM['config']
config['batch_size'] = batch_size
dDDPM_state_dict = get_model_state_dict(save_data_dDDPM)

# Instantiate dDDPM model
latent_model = Unet(config)
color_channels = get_color_channels(config['dataset'])
model_dDDPM = DownsampleDDPM(config, latent_model, device, color_channels)
model_dDDPM.load_state_dict(dDDPM_state_dict)
model_dDDPM = model_dDDPM.to(device)
model_dDDPM.eval()

# compute and save samples
print(f'\nGenerating samples jointly from checkpoints {checkpoint_dDDPM}')
print(f'Using configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')
print(f'\nDenoising timesteps [T, {early_stop}]')
sample_list = []
time_start = time.time()
x_101, _ = model_dDDPM.sample(batch_size, early_stop=early_stop)
x_101 = x_101.cpu().numpy()
sample_list.append(x_101)
sampling_time = time.time() - time_start
print('shape x', x_101.shape)
print('len list', len(sample_list))

# print stats
print(f'Using batch size {config["batch_size"]}')
print(f'Total time: {sampling_time}')
print(f'Sample time: {sampling_time/fid_samples}')

# Input space samples
save_path = os.path.join(SAMPLE_DIR, f'x_{early_stop}')
np.save(save_path, sample_list, allow_pickle=False)
print(f'Samples saved to {save_path}')
