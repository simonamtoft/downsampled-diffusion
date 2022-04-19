import torch
import numpy as np
from .utils import min_max_norm_image


def create_generator_loader(dataloader):
    for batch in dataloader:
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.float().numpy() * 255.
        yield np.moveaxis(batch, 1, -1)


def compute_vlb(model, test_loader, device):
    vlb = []
    for x, _ in iter(test_loader):
        x = x.to(device)
        losses = model.calc_vlb(x)
        vlb.append(losses['vlb'])
    vlb = torch.stack(vlb, dim=1).mean().cpu().numpy().item()
    return vlb


def compute_test_losses(model, test_loader, device):
    vlb = []
    L_simple = []
    for x, _ in iter(test_loader):
        x = x.to(device)
        losses = model.test_losses(x)
        vlb.append(losses['vlb'])
        L_simple.append(losses['L_simple'])
    vlb = torch.stack(vlb, dim=1).mean().cpu().numpy().item()
    L_simple = torch.stack(L_simple, dim=0).mean().cpu().numpy().item()
    return vlb, L_simple


def fix_samples(samples):
    samples = min_max_norm_image(samples) * 255.
    samples = samples.cpu().numpy()
    samples = np.moveaxis(samples, 1, -1)
    return samples
