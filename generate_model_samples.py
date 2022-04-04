from tqdm import tqdm
import torch
from models import Unet, DDPM, DownsampleDDPM
from utils import get_dataloader, get_color_channels, \
    create_generator_ddpm, create_generator_dddpm
import numpy as np
from utils import min_max_norm_image


WANDB_PROJECT = 'ddpm-test'
DATA_ROOT = '../data'
device = 'cuda'
saved_model = 'cifar_simple_ema_3'
fid_samples = 10000

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
    g_model = create_generator_ddpm(model, config['batch_size'], fid_samples)
elif config['model'] == 'dddpm':
    model = DownsampleDDPM(config, latent_model, device, color_channels)
    g_model = create_generator_dddpm(model, config['batch_size'], fid_samples)
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

# compute and save samples
sample_list = []
for i in tqdm(range(int(np.ceil(fid_samples/config['batch_size']))), desc='sampling from model'):
    samples = model.sample(config['batch_size'])
    samples = min_max_norm_image(samples) * 2. - 1.
    samples = samples.cpu().numpy()
    samples = np.moveaxis(samples, 1, -1)
    sample_list.append(samples)
np.save(f'./results/samples/{saved_model}', sample_list)
print(f'Samples saved to ./results/samples/{saved_model}')