import os
import torch
import numpy as np
from fid.inception import InceptionV3
from fid.fid_score import compute_statistics_of_generator, \
    load_statistics, calculate_frechet_distance


def create_generator_ddpm(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for _ in range(num_iters):
        with torch.no_grad():
            samples = model.sample(batch_size)
        yield samples.float()


def create_generator_dddpm(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for _ in range(num_iters):
        with torch.no_grad():
            samples, _ = model.sample(batch_size)
        yield samples.float()


def compute_fid(config, g, fid_samples, fid_dir, device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, inception_model, config['batch_size'], dims, device, fid_samples)
    path = os.path.join(fid_dir, f"{config['dataset']}_{fid_samples}.npz")
    m0, s0 = load_statistics(path)
    return calculate_frechet_distance(m0, s0, m, s)


def compute_vlb(model, test_loader, device):
    vlb = []
    for x, _ in iter(test_loader):
        x = x.to(device)
        losses = model.calc_vlb(x)
        vlb.append(losses['vlb'])
    return torch.stack(vlb, dim=1).mean()
