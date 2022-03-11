import torch
from models import Unet, DDPM
from utils import get_dataloader, get_color_channels

DATA_ROOT = '../data'
device = 'cuda'
saved_model = 'checkpoint_ddpm_vlb_paper_1.pt'

# load saved state dict of model and its config file
save_data = torch.load(f'./results/{saved_model}')
config = save_data['config']
model_state_dict = save_data['model']

# get test data
test_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=False)
color_channels = get_color_channels(config['dataset'])

# Instantiate model
latent_model = Unet(config)
model = DDPM(config, latent_model, device, color_channels)

# load the state dict into the model
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

# iterate through test set
vlb = []
for x, _ in iter(test_loader):
    x = x.to(device)
    losses = model.calc_vlb(x)
    vlb.append(losses['vlb'])
vlb = torch.stack(vlb, dim=1).mean()
print(f'For {saved_model}')
print(f'Test VLB: {vlb}')
