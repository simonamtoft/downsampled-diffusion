import os
import json
import tensorflow.compat.v1 as tf
import numpy as np
from utils import Evaluator, SAMPLE_DIR

saved_model = 'fengnima'
fid_samples = 10000
reference_batch = 'celeba_hq_10k.npy'

# load samples and reference images and test data
samples = np.load(os.path.join(SAMPLE_DIR, f'{saved_model}.npy'))
reference = np.load(os.path.join(SAMPLE_DIR, reference_batch))

# print min-max values
print('\n\t\tMin\t\tMax')
print(f'Sample:\t{samples.min():.2f}\t{samples.max():.2f}')
print(f'Data:\t{np.min(reference):.2f}\t{np.max(reference):.2f}')

### COMPUTE METRICS ###
metrics = {}
config = tf.ConfigProto(
    allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
)
config.gpu_options.allow_growth = True
evaluator = Evaluator(tf.Session(config=config))
print("warming up TensorFlow...")
evaluator.warmup()
print("computing reference batch activations...")
ref_acts = evaluator.read_activations(reference)
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
print(json.dumps(metrics, sort_keys=False, indent=4) + '\n')
