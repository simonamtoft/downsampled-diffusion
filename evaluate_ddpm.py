import os
import json
import torch
import wandb
from models import Unet, DDPM, DownsampleDDPM
from utils import get_dataloader, get_color_channels, \
    compute_vlb, create_generator_loader, Evaluator, \
    SAMPLE_DIR, CHECKPOINT_DIR, DATA_DIR
import tensorflow.compat.v1 as tf
import numpy as np

# ONLY CHANGE STUFF HERE
saved_model = 'chq_x3_t100_d_3'
saved_sample = saved_model
# saved_sample = 'cifar_full_255'
fid_samples = 10000

# Directories etc.
WANDB_PROJECT = 'ddpm-test'
device = 'cuda'

# load saved state dict of model and its config file
samples = np.load(os.path.join(SAMPLE_DIR, f'{saved_sample}.npy'))
save_data = torch.load(os.path.join(CHECKPOINT_DIR, f'{saved_model}.pt'))
config = save_data['config']
if 'ema_model' in config:
    model_state_dict = save_data['ema_model']
else:
    model_state_dict = save_data['model']

# get data
train_loader, _ = get_dataloader(config, data_root=DATA_DIR, device=device, train=True, val_split=0, train_transform=False)
test_loader = get_dataloader(config, data_root=DATA_DIR, device=device, train=False, train_transform=False)
tmp = create_generator_loader(train_loader)
tmp = list(tmp)
g_data = create_generator_loader(train_loader)

# print min-max values
print('\n\t\tMin\t\tMax')
print(f'Sample:\t{samples.min():.2f}\t{samples.max():.2f}')
print(f'Data:\t{np.min(tmp):.2f}\t{np.max(tmp):.2f}')

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

# setup
print(f'\nEvaluating the checkpoint {saved_model} with configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')
# wandb.init(project=WANDB_PROJECT, config=config, resume='allow', id=config['wandb_id'])
metrics = {}

### COMPUTE METRICS ###
# compute VLB
metrics['vlb'] = compute_vlb(model, test_loader, device)

# compute other metrics using Evaluator
config = tf.ConfigProto(
    allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
)
config.gpu_options.allow_growth = True
evaluator = Evaluator(tf.Session(config=config))
print("warming up TensorFlow...")
evaluator.warmup()
print("computing reference batch activations...")
ref_acts = evaluator.read_activations(g_data)
print("computing/reading reference batch statistics...")
ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)
print("computing sample batch activations...")
sample_acts = evaluator.read_activations(samples)
print("computing/reading sample batch statistics...")
sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)
print("Computing evaluations...")
metrics['is'] = evaluator.compute_inception_score(sample_acts[0])
metrics['fid'] = sample_stats.frechet_distance(ref_stats)
metrics['sfid'] = sample_stats_spatial.frechet_distance(ref_stats_spatial)
prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
metrics['precision'] = prec
metrics['recall'] = recall

# Display resulting metrics
print('\nResults:')
print(metrics)
# wandb.finish()