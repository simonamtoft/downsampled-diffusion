import os
import json
from utils import Evaluator, REFERENCE_DIR
import tensorflow.compat.v1 as tf
import numpy as np

# Compute the performance metrics between
# the two datasets defined.
# ds_1 is acting as the reference batch
# ds_2 is acting as the generative model
device = 'cuda'
n = '10k'
ds_1 = f'celeba_hq_real_{n}.npy'
ds_2 = f'celeba_hq_256_{n}.npy'

# load samples and reference images and test data
dataset_1 = np.load(os.path.join(REFERENCE_DIR, ds_1))
dataset_2 = np.load(os.path.join(REFERENCE_DIR, ds_2))

# print min-max values
print('\n\t\t\tMin\t\tMax')
print(f'Dataset 1:\t{dataset_1.min():.2f}\t{dataset_1.max():.2f}')
print(f'Dataset 2:\t{dataset_2.min():.2f}\t{dataset_2.max():.2f}')

### COMPUTE METRICS ###
metrics = {}
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
evaluator = Evaluator(tf.Session(config=config))
evaluator.warmup()

acts_1 = evaluator.read_activations(dataset_1)
stats_1, stats_spatial_1 = evaluator.read_statistics(acts_1)

acts_2 = evaluator.read_activations(dataset_2)
stats_2, stats_spatial_2 = evaluator.read_statistics(acts_2)

metrics['fid'] = stats_2.frechet_distance(stats_1)
metrics['sfid'] = stats_spatial_2.frechet_distance(stats_spatial_1)
prec, recall = evaluator.compute_prec_recall(acts_1[0], acts_2[0])
metrics['precision'] = prec
metrics['recall'] = recall

# Display resulting metrics
print('\nResults:')
print(f'({ds_1} vs. {ds_2})')
print(json.dumps(metrics, sort_keys=False, indent=4) + '\n')
