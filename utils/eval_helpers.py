import os
import torch
import numpy as np
# from fid.inception import InceptionV3
# from fid.fid_score import compute_statistics_of_generator, \
#     load_statistics, calculate_frechet_distance
from .utils import min_max_norm_image


def create_generator_loader(dataloader):
    for batch in dataloader:
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.float().numpy() * 255.
        # batch = batch.float().numpy()
        yield np.moveaxis(batch, 1, -1)
        # yield batch.numpy()


def compute_vlb(model, test_loader, device):
    vlb = []
    for x, _ in iter(test_loader):
        x = x.to(device)
        losses = model.calc_vlb(x)
        vlb.append(losses['vlb'])
    vlb = torch.stack(vlb, dim=1).mean().cpu().numpy()
    return vlb


def fix_samples(samples):
    samples = min_max_norm_image(samples) * 255.
    samples = samples.cpu().numpy()
    samples = np.moveaxis(samples, 1, -1)
    return samples
