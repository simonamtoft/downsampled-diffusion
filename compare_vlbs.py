import torch
import numpy as np

from models import Unet, DDPM
from utils import get_dataloader, get_color_channels
from models.utils.losses import discretized_gaussian_log_likelihood
from models.utils.helpers import flat_bits

DATA_ROOT = '../data'
device = 'cuda'
saved_model = 'cifar_linear_var0.pt'

# load saved state dict of model and its config file
save_data = torch.load(f'./results/{saved_model}')
config = save_data['config']
model_state_dict = save_data['model']

# get test data
test_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=False)
color_channels = get_color_channels(config['dataset'])

# instantiate model
latent_model = Unet(
    dim=config['unet_chan'],
    in_channels=color_channels,
    dim_mults=config['unet_dims'],
)
model = DDPM(config, latent_model, device, color_channels)

# load pretrained model, in evaluation mode
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

# get single batch of data
x, _ = next(iter(test_loader))
print('\nData info:')
print('Dataset:', config['dataset'])
print('(min, max):', x.min().numpy(), x.max().numpy())
print('shape:', x.shape)
x = x.to(device)

# compute VLB using KL (without L_0 and L_T)
vlb = []
for t in list(range(config['T']-1))[::-1][:-1]:
    t_batch = torch.full((config['batch_size'],), t, device=device, dtype=torch.long)
    eps = torch.randn_like(x)
    with torch.no_grad():
        x_t = model.q_sample(x, t_batch, eps)
        vlb_ = model.vlb_terms(x, x_t, t_batch)
    vlb.append(vlb_)
vlb = torch.stack(vlb, dim=1)
vlb_kl = vlb.sum(dim=1).mean()

# compute VLB using weight (without L_0 and L_T)
vlb = []
for t in list(range(config['T']-1))[::-1][:-1]:
    t_batch = torch.full((config['batch_size'],), t, device=device, dtype=torch.long)
    with torch.no_grad():
        _, loss_dict = model.losses(x, t_batch)
    vlb.append(loss_dict['L_vlb'].item())
vlb = torch.tensor(vlb)
vlb = vlb.sum() / np.log(2.)

# get nll
t_0 = torch.full((config['batch_size'],), 0, device=device, dtype=torch.long)
eps = torch.randn_like(x)
x_0 = model.q_sample(x, t_0, eps)
pred_mean, _, pred_log_var = model.p_mean_variance(x_0, t_0)
nll = -discretized_gaussian_log_likelihood(
    x, means=pred_mean, log_scales=0.5*pred_log_var
)
nll = flat_bits(nll).mean()

# get prior
prior = model.calc_prior(x)
prior = prior.mean()

# print stuff
print('\n\nlosses:')
print(f'prior:\t{prior:.2f}')
print(f'vlb:\t{vlb:.2f}')
print(f'vlb_kl:\t{vlb_kl:.2f}')
print(f'nll:\t{nll:.2f}')
print(f'total:\t{prior + vlb + nll:.2f}')
print(f'total_kl:\t{prior + vlb_kl + nll:.2f}')