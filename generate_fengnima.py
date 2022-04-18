import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils import fix_samples, SAMPLE_DIR, CHECKPOINT_DIR
from fengnima_pretrained.util import print_size, sampling
from fengnima_pretrained.UNet import UNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


saved_model = 'fengnima'
fid_samples = 5000
batch_size = 192

# define hyper params
unet_config = {"t_emb_dim": 128}
T = 1000
beta_0 = 0.0001
beta_T = 0.02

# Compute diffusion hyperparameters
Beta = torch.linspace(beta_0, beta_T, T).cuda()
Alpha = 1 - Beta
Alpha_bar = torch.ones(T).cuda()
Beta_tilde = Beta + 0
for t in range(T):
    Alpha_bar[t] *= Alpha[t] * Alpha_bar[t-1] if t else Alpha[t]
    if t > 0:
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
Sigma = torch.sqrt(Beta_tilde)

# load model
net = UNet(**unet_config)
print_size(net)
checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f'{saved_model}.pkl'), map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
net = net.cuda()

# Generation
time_start = time.time()
sample_list = []
for i in tqdm(range(int(np.ceil(fid_samples/batch_size))), desc='sampling from model'):
    samples = sampling(net, (batch_size, 3, 256, 256), T, Alpha, Alpha_bar, Sigma)
    sample_list.append(fix_samples(samples))
time_diff = time.time() - time_start
print(f'Generated {fid_samples} samples in {time_diff} seconds')
print(f'Average sampling time: {time_diff/fid_samples} (seconds per sample)')
output_directory = os.path.join(SAMPLE_DIR, saved_model)
print(f'Saving to {output_directory}')
np.save(output_directory, sample_list)
