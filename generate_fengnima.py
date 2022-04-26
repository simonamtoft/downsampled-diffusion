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
n_batches = int(np.ceil(fid_samples/batch_size))
for i in tqdm(range(n_batches), desc='sampling from model'):
    samples = sampling(net, (batch_size, 3, 256, 256), T, Alpha, Alpha_bar, Sigma)
    sample_list.append(fix_samples(samples))
sampling_time = time.time() - time_start
save_path = os.path.join(SAMPLE_DIR, saved_model + '_1')
print(f'Using batch size {batch_size}')
print(f'Samples saved to {save_path}')
print(f'Total time: {sampling_time}')
print(f'Sample time: {sampling_time/fid_samples}')
print(f'Batch time: {sampling_time/n_batches}')
np.save(save_path, sample_list)
