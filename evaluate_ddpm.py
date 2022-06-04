import os
import json
import torch
from models import Unet, DDPM, DownsampleDDPM
from utils import get_dataloader, get_color_channels, \
    Evaluator, compute_test_losses, get_model_state_dict, \
    SAMPLE_DIR, CHECKPOINT_DIR, DATA_DIR, REFERENCE_DIR
import tensorflow.compat.v1 as tf
import numpy as np

# ONLY CHANGE STUFF HERE
saved_model = 'celeba_x2'
saved_sample = saved_model
fid_samples = 50000

# Directories etc.
WANDB_PROJECT = 'ddpm-test'
device = 'cuda'

# load saved data
save_data = torch.load(os.path.join(CHECKPOINT_DIR, f'{saved_model}.pt'))
model_state_dict = get_model_state_dict(save_data)

# fix config if missing
config = save_data['config']
if config['model'] == 'dddpm':
    if 'force_latent' not in config:
        config['force_latent'] = False

# get name of reference batch
if config['dataset'] == 'mnist':
    reference_batch = 'mnist_32_10k.npy'
elif config['dataset'] == 'cifar10':
    if fid_samples == 10000:
        reference_batch = 'cifar10_10k.npy'
    elif fid_samples == 50000:
        reference_batch = 'cifar10_50k.npy'
elif config['dataset'] == 'celeba':
    if fid_samples == 10000:
        reference_batch = 'celeba_10k.npy'
    elif fid_samples == 50000:
        reference_batch = 'celeba_50k.npy'
elif config['dataset'] == 'celeba_hq_64':
    if fid_samples == 10000:
        reference_batch = 'celeba_hq_64_10k.npy'
    elif fid_samples == 24000:
        reference_batch = 'celeba_hq_64_24k.npy'
elif config['dataset'] == 'celeba_hq':
    reference_batch = 'celeba_hq_256_10k.npy'

# load samples and reference images and test data
samples = np.load(os.path.join(SAMPLE_DIR, f'{saved_sample}.npy'))
reference = np.load(os.path.join(REFERENCE_DIR, reference_batch))
test_loader = get_dataloader(config, data_root=DATA_DIR, device=device, train=False, train_transform=False)

# print min-max values
print('\n\t\tMin\t\tMax')
print(f'Sample:\t{samples.min():.2f}\t{samples.max():.2f}')
print(f'Data:\t{np.min(reference):.2f}\t{np.max(reference):.2f}')

# Setup DDPM model
print(f'\nLoading model checkpoint {saved_model}')
print(f'Trained for {save_data["step"]} steps with configuration dict:')
print(json.dumps(config, sort_keys=False, indent=4) + '\n')
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
print(f'\nComputing results using {fid_samples} samples')
print(f'Reference batch: {reference_batch}')
print(f'Samples: {saved_sample}')
metrics = {}
vlb, L_simple = compute_test_losses(model, test_loader, device)
metrics['vlb'] = vlb
metrics['L_simple'] = L_simple

# compute other metrics using Evaluator
config = tf.ConfigProto(
    allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
)
config.gpu_options.allow_growth = True
evaluator = Evaluator(tf.Session(config=config))
# print("warming up TensorFlow...")
evaluator.warmup()
# print("computing reference batch activations...")
ref_acts = evaluator.read_activations(reference)
# print("computing/reading reference batch statistics...")
ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)
# print("computing sample batch activations...")
sample_acts = evaluator.read_activations(samples)
# print("computing/reading sample batch statistics...")
sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)
# print("Computing evaluations...")
metrics['is'] = evaluator.compute_inception_score(sample_acts[0])
metrics['fid'] = sample_stats.frechet_distance(ref_stats)
metrics['sfid'] = sample_stats_spatial.frechet_distance(ref_stats_spatial)
prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
metrics['precision'] = prec
metrics['recall'] = recall

# Display resulting metrics
print('\nResults:')
print(json.dumps(metrics, sort_keys=False, indent=4) + '\n')
