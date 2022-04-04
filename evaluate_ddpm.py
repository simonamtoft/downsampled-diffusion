import os
import json
import torch
import wandb
from models import Unet, DDPM, DownsampleDDPM
from utils import get_dataloader, get_color_channels, \
    create_generator_ddpm, create_generator_dddpm, \
    compute_fid, compute_vlb, create_generator_loader
import tensorflow.compat.v1 as tf
from fid.evaluator import Evaluator
from itertools import chain
import numpy as np

WANDB_PROJECT = 'ddpm-test'
FID_DIR = './results/fid_stats'
DATA_ROOT = '../data'
device = 'cuda'
saved_model = 'cifar_simple_ema_3'
fid_samples = 10000

# load saved state dict of model and its config file
save_data = torch.load(f'./results/checkpoints/{saved_model}.pt')
samples = np.load(f'./results/samples/{saved_model}.npy')
config = save_data['config']
if 'ema_model' in config:
    model_state_dict = save_data['ema_model']
else:
    model_state_dict = save_data['model']

# get data
train_loader, _ = get_dataloader(config, data_root=DATA_ROOT, device=device, train=True, val_split=0)
test_loader = get_dataloader(config, data_root=DATA_ROOT, device=device, train=False, train_transform=False)
g_data = create_generator_loader(train_loader)

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


### COMPUTE METRICS ###
metrics = {}
# wandb.init(project=WANDB_PROJECT, config=config, resume='allow', id=config['wandb_id'])
print(f'\nEvaluating the checkpoint {saved_model} with configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')

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
metrics['vlb'] = compute_vlb(model, test_loader, device)


# Display resulting metrics
print('\nResults:')
print(metrics)
print(json.dumps(metrics, sort_keys=False, indent=4) + '\n')
# wandb.finish()