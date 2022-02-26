import torch
from models import Unet, DDPM
from utils import get_dataloader, get_color_channels
from models.losses import discretized_gaussian_log_likelihood
from models.helpers import flat_bits

DATA_ROOT = '../data'
device = 'cuda'
saved_model = './results/'

# load saved state dict of model and its config file
save_data = torch.load(saved_model)
config = save_data['config']
model_state_dict = save_data['model']

# get test data
test_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=False)
color_channels = get_color_channels(config['dataset'])

# Instantiate model
latent_model = Unet(
    dim=config['unet_chan'],
    in_channels=color_channels,
    dim_mults=config['unet_dims'],
)
model = DDPM(config, latent_model, device, color_channels)

# load the state dict into the model
model.load_state_dict(model_state_dict)

# get single batch of data
x, _ = next(iter(test_loader))
x = x.to(device)

# compute VLB using KL for t in [1, T-1]
vlb = []
for t in list(range(config['T']-1))[::-1][:-1]:
    t_batch = torch.full((config['batch_size'],), t, device=device, dtype=torch.long)
    eps = torch.randn_like(x)
    with torch.no_grad():
        x_t = model.q_sample(x, t_batch, eps)
        vlb_ = model.vlb_terms(x, x_t, t_batch)
    vlb.append(vlb_)
vlb = torch.stack(vlb, dim=1).sum(dim=1)

# get nll (L_0)
t_0 = torch.full((config['batch_size'],), 0, device=device, dtype=torch.long)
pred_mean, _, pred_log_var = model.p_mean_variance(x, t_0)
nll = -discretized_gaussian_log_likelihood(
    x, means=pred_mean, log_scales=0.5*pred_log_var[0]
)
nll = flat_bits(nll)

# get prior (L_T)
prior = model.calc_prior(x)

# add everything together
L_vlb = (vlb + nll + prior).mean()

print(f'Test VLB: {L_vlb}')