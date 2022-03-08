import numpy as np
import torch
import torch.nn as nn

from .ddpm import DDPM
from models.downsampled.wrapper import get_downsampling, \
    get_upsampling


class DownsampleDDPM(DDPM):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__(config, denoise_model, device, color_channels)

        # when to compute reconsctrution
        self.t_rec_max = int(self.timesteps - 1) if config['t_rec_max'] == -1 else config['t_rec_max']

        # original shape
        shape = (self.in_channels, self.image_size, self.image_size)

        # latent shape
        unet_in = config['unet_in']
        self.dim_reduc = np.power(2, config['n_downsamples']).astype(int)
        z_size = int(self.image_size/self.dim_reduc)
        self.sample_shape = (unet_in, z_size, z_size)

        # Instantiate down-up sample network
        assert unet_in >= self.in_channels, f'Input channels to DDPM-Unet {unet_in} should be equal or larger to data color channels {self.in_channels}.'
        self.downsample = get_downsampling(config, shape)
        self.upsample = get_upsampling(config, shape)

    @torch.no_grad()
    def reconstruct(self, x:torch.tensor, n:int) -> tuple:
        """
        Reconstructs x_hat from x in downsampled space and upsamples at the end.
        Method only used for visualization and not for computing gradients etc.
        
        Returns
            x_recon (torch.tensor): A tensor of n reconstructions of x in the 
                                    original image space. The reconstructions 
                                    are made from increasing noisy x_t for
                                    timescales t linearly between 0 to T.
            z_hat   (torch.tensor): A tensor of n reconstructions of x in the 
                                    latent space. The reconstructions 
                                    are made from increasing noisy z_t for
                                    timescales t linearly between 0 to T.
        """
        assert x.shape[0] >= n
        x = x[:n]

        # define linear timescales from 0 to T for n steps
        t = torch.linspace(0, self.timesteps - 1, n, device=self.device, dtype=torch.long)

        # downsample the input
        z = self.downsample(x)

        # sample noisy z_t from the q distribution
        eps = torch.randn_like(z)
        z_t = self.q_sample(z, t, eps)

        # predict added noise for each z_t
        eps_hat = self.latent_model(z_t, t)

        # compute latent space reconstruction
        z_hat = self.predict_x_from_eps(z_t, t, eps_hat)

        # upsample reconstruction
        x_recon = self.upsample(z_hat)
        return x_recon, z_hat

    @torch.no_grad()
    def sample(self, batch_size:int=16) -> tuple:
        """
        Sample a batch of images in latent space, and upsample these to original
        image space.
        
        Returns:
            x_sample (torch.tensor):    A tensor of batch_size samples in original image space.
            z_sample (torch.tensor):    A tensor of batch_size samples in latent space.
        """
        z_sample = self.p_sample_loop((batch_size, *self.sample_shape))
        x_sample = self.upsample(z_sample)
        return x_sample, z_sample

    def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
        """Train loss computations for the Downsample DDPM architecture."""
        # downsample the input
        z = self.downsample(x)

        # Generate noise
        eps = torch.randn_like(z)

        # sample noisy z from q distribution for step t
        z_t = self.q_sample(z, t, eps)

        # predict the noise for step t
        eps_hat = self.latent_model(z_t, t)

        # create reconstrution from model output at step t and upsample
        z_hat = self.predict_x_from_eps(z_t, t, eps_hat)
        x_hat = self.upsample(z_hat)

        # compute latent loss
        loss_latent = self.get_loss(eps, eps_hat).mean(dim=[1, 2, 3])

        # compute the reconstruction loss
        # set it to 0 when t >= t_rec_max
        loss_recon = self.get_loss(x, x_hat).mean(dim=[1, 2, 3])
        zeros = torch.zeros_like(loss_recon).detach()
        cond = (t < self.t_rec_max).detach()
        loss_recon = torch.where(cond, loss_recon, zeros)

        # compute objective
        obj = (loss_latent + loss_recon).mean()

        return obj, {
            'latent': loss_latent.mean(),
            'recon': loss_recon.mean()
        }


class DownsampleDDPMAutoencoder(DownsampleDDPM):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__(config, denoise_model, device, color_channels)

    def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
        """Training loss computations for the Autoencoder implementation for down-up sampling."""
        # downsample the input
        z = self.downsample(x)

        # Generate noise
        eps = torch.randn_like(z)

        # sample noisy z from q distribution for step t and t=1
        z_t = self.q_sample(z, t, eps)

        # denoise the noisy z at step t and t=1
        eps_hat = self.latent_model(z_t, t)

        # upsample the latent
        x_hat = self.upsample(z)

        # compute losses
        loss_latent = self.get_loss(eps, eps_hat).mean(dim=[1, 2, 3])
        loss_recon = self.get_loss(x, x_hat).mean(dim=[1, 2, 3])
        return (loss_latent, loss_recon)
