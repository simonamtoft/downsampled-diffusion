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
    def reconstruct(self, x:torch.tensor) -> torch.tensor:
        """
        Reconstructs x_hat from x in downsampled space and upsamples at the end.
        Method only used for visualization and not for computing gradients etc.
        """
        # Get the first diffusion step (t=0)
        t_0 = torch.full((x.shape[0],), 0, device=self.device, dtype=torch.long)
        
        # downsample the input
        z = self.downsample(x)

        # Generate noise
        eps = torch.randn_like(z)
        
        # sample noisy z from q distribution for t=1
        z_0 = self.q_sample(z, t_0, eps)
        
        # get model output eps for z_0 and get reconstruction
        eps_hat = self.latent_model(z_0, t_0)
        z_hat = self.predict_x_from_eps(z_0, t_0, eps_hat)
        
        # upsample reconstruction and return
        x_recon = self.upsample(z_hat)
        return x_recon

    @torch.no_grad()
    def p_sample_loop(self, shape:tuple) -> torch.tensor:
        # Create random noise in downsampled image space
        img = torch.randn(shape, device=self.device)
                
        # Pass backwards through the DDPM
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((shape[0],), i, device=self.device, dtype=torch.long))

        # return the upsampled result
        img = self.upsample(img)
        return img

    # def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
    #     """Train loss computations for the Downsample DDPM architecture."""

    #     # generate noise
    #     eps = torch.randn_like(x)

    #     # downsample the input and noise
    #     z = self.downsample(x)
    #     eps_z = self.downsample(eps)

    #     # sample noisy z from q distribution for step t
    #     z_t = self.q_sample(z, t, eps_z)

    #     # predict the noise for step t
    #     eps_z_hat = self.latent_model(z_t, t)

    #     # create reconstrution from model output at step t and upsample
    #     eps_hat = self.upsample(eps_z_hat)

    #     # compute latent loss
    #     loss_latent = self.get_loss(eps, eps_hat).mean(dim=[1, 2, 3])

    #     # compute the reconstruction loss
    #     loss_recon = torch.tensor([0.]).to(self.device)
        
    #     # compute objective
    #     obj = (loss_latent + loss_recon).mean()

    #     return obj, {
    #         'latent': loss_latent.mean(),
    #         'recon': loss_recon.mean()
    #     }

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
