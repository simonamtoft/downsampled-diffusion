import numpy as np
import torch
import torch.nn as nn

from utils import min_max_norm_image
from models.downsampled import get_downsampling, \
    get_upsampling

from .ddpm import DDPM


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
        z = self.rescaled_downsample(x)

        # sample noisy z_t from the q distribution
        eps = torch.randn_like(z)
        z_t = self.q_sample(z, t, eps)

        # predict added noise for each z_t
        eps_hat = self.latent_model(z_t, t)

        # compute latent space reconstruction
        z_recon = self.predict_x_from_eps(z_t, t, eps_hat)

        # upsample reconstruction
        x_recon = self.upsample(z_recon)
        return x_recon, z_recon

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

    def rescaled_downsample(self, x:torch.tensor) -> torch.tensor:
        """Downsample input x to z-space and rescale output z to be in [-1, 1]"""
        # downsample input
        z = self.downsample(x)
        
        # rescale to [-1, 1] as DDPM expects
        z = min_max_norm_image(z) * 2. - 1.
        return z

    def loss_recon(self, x:torch.tensor, x_hat:torch.tensor, t:torch.tensor) -> torch.tensor:
        loss = self.get_loss(x, x_hat).mean(dim=[1, 2, 3])
        loss = torch.where(t < self.t_rec_max, loss, torch.zeros_like(loss))
        return loss

    def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
        """Train loss computations for the Downsample DDPM architecture."""
        
        # downsample x to z rescaled to [-1, 1]
        z = self.rescaled_downsample(x)
        
        # foward pass through DDPM
        eps = torch.randn_like(z)
        z_t = self.q_sample(z, t, eps)
        eps_hat = self.latent_model(z_t, t)
        L_ddpm = self.loss_ddpm(eps, eps_hat, t)
        
        # compute image / reconstruction loss
        z_hat = self.predict_x_from_eps(z_t, t, eps_hat)
        x_hat = self.upsample(z_hat)
        L_rec = self.loss_recon(x, x_hat, t)

        # compute objective
        obj = (L_ddpm + L_rec).mean()
        return obj, {
            'latent': L_ddpm.mean(),
            'recon': L_rec.mean()
        }
    
    # def t_sample(self, n:int) -> torch.tensor:
    #     """Sample n t's uniformly between [0, T], apart from t = 0 which is weighted higher."""
    #     w = torch.ones(self.timesteps)
    #     # w[0] = int(self.timesteps * 0.01)   # sample t = 0, 1% of the time.
    #     # self.t_w = w / w.sum()
    #     return torch.multinomial(w, n, replacement=True).to(self.device).long()


class DownsampleDDPMAutoencoder(DownsampleDDPM):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__(config, denoise_model, device, color_channels)

    def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
        """Training loss computations for the Autoencoder implementation for down-up sampling."""
        # downsample the input
        z = self.rescaled_downsample(x)

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
