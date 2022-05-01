import os
import numpy as np
from tqdm import tqdm
from utils import get_dataloader, \
    create_generator_loader, \
    DATA_DIR, REFERENCE_DIR

# Create a reference batch for the desired dataset
fid_samples = 50000
config = {
    'dataset': 'celeba',
    'image_size': 64,
    'model': 'dddpm',
    'batch_size': 125,
}
train_loader, _ = get_dataloader(config, data_root=DATA_DIR, device='cuda', train=True, val_split=0, train_transform=False)
data = create_generator_loader(train_loader)
image_list = []
for i in tqdm(range(int(np.ceil(fid_samples/config['batch_size']))), desc='generating reference batch'):
    x = next(data)
    image_list.append(x)
save_path = os.path.join(REFERENCE_DIR, 'celeba_50k')
np.save(save_path, image_list, allow_pickle=False)
print(f'Saved reference batch to {save_path}')
