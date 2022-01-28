import torch.nn.functional as F


def l1_loss(noise, recon):
    """Computes the l1 loss between noise and reconstruction."""
    return (noise - recon).abs().mean()

def l2_loss(noise, recon):
    """Computes the l2 loss between noise and reconstruction."""
    return F.mse_loss(noise, recon)