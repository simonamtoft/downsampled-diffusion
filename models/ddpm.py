import numpy as np
import torch
from torch import nn
from functools import partial

from .losses import l1_loss, l2_loss
from .helpers import default, extract, \
    noise_like, cosine_beta_schedule


class DDPM(nn.Module):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__()
        
        self.denoise = denoise_model
        self.in_channels = color_channels
        self.device = device
        
        # extract fields from config 
        self.image_size = config['image_size']
        self.timesteps = config['timesteps']

        # define loss computation
        if config['loss_type'] == 'l1':
            self.get_loss = l1_loss
        elif config['loss_type'] == 'l2':
            self.get_loss = l2_loss
        else:
            raise NotImplementedError()
        
        # Initialize betas (variances)
        betas = cosine_beta_schedule(self.timesteps)
        
        # Compute alphas from betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        # ensure variables are pytorch tensors with dtype float32
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        # register buffers for alphas and betas
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
    # def q_mean_variance(self, x_start, t):
    #     """Compute and return the mean, variance and log(variance) for the q distribution for a specific step t in the diffusion model."""
    #     mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    #     variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
    #     log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    #     return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):

        # Get mean and log variance of the model
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, clip_denoised)
        
        # generate noise
        noise = noise_like(x.shape, self.device, repeat_noise)
        
        # no noise when t == 0
        batch_size = x.shape[0]
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
        
        # compute sample from p
        x_t = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        
        return x_t
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Generate images by sampling"""
        
        # start with an image of completely random noise
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        # go through the ddpm in reverse order (from t=T to t=0)
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        """Sample a batch"""
        return self.p_sample_loop((batch_size, self.in_channels, self.image_size, self.image_size))

    # @torch.no_grad()
    # def interpolate(self, x1, x2, t=None, lam=0.5):
    #     b, *_, device = *x1.shape, x1.device
    #     t = default(t, self.timesteps - 1)

    #     assert x1.shape == x2.shape

    #     t_batched = torch.stack([torch.tensor(t, device=device)] * b)
    #     xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

    #     img = (1 - lam) * xt1 + lam * xt2
    #     for i in reversed(range(0, t)):
    #         img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

    #     return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        
        # Generate noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        # sample noisy x from q distribution for step t
        x_noisy = self.q_sample(x_start, t, noise)
        
        # denoise the noisy x at step t
        x_recon = self.denoise(x_noisy, t)

        # compute loss
        loss = self.get_loss(noise, x_recon)

        return loss

    def forward(self, x, *args, **kwargs):
        # select a random timestep t for each x in batch        
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
        
        # denoise x at timestep t, and compute loss
        loss = self.p_losses(x, t, *args, **kwargs)       
        return loss


class DownsampleDDPM(DDPM):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__(config, denoise_model, device, color_channels)

        # number of channels inside the down-up sample network
        self.channels = [int(config['unet_chan'] / 2), config['unet_chan']]
        
        # number of downsamples
        n_downsamples = 1
        self.dim_reduc = np.power(2, n_downsamples)
        
        # Instantiate downsample network
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=2, padding=0, stride=2)
        )

        # Instantiate upsample network
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.channels[1], self.channels[0], kernel_size=4, stride=2, padding=1),
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.in_channels, kernel_size=1, padding=0),
        )
    
    @torch.no_grad()
    def p_sample_loop(self, shape:tuple):
        # Create random noise in downsampled space
        img = torch.randn(shape, device=self.device)
        # img = self.downsample(img)
        
        # Pass backwards through the DDPM
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((shape[0],), i, device=self.device, dtype=torch.long))
            
        # return the upsampled result
        img = self.upsample(img)
        return img
    
    @torch.no_grad()
    def sample(self, batch_size:int=16):
        img_size = int(self.image_size/self.dim_reduc)
        return self.p_sample_loop((batch_size, self.channels[1], img_size, img_size))
    
    def p_losses(self, x_start:torch.tensor, t:torch.tensor, noise:torch.tensor=None) -> tuple:
        # Instantiate t=1 for each x in the batch
        t_1 = torch.ones((x_start.shape[0],), device=self.device).long()
        
        # downsample the input
        z_start = self.downsample(x_start)
        
        # Generate noise
        noise_z = torch.randn_like(z_start)

        # sample noisy z from q distribution for step t and t=1
        z_t = self.q_sample(z_start, t, noise_z)
        z_1 = self.q_sample(z_start, t_1, noise_z)
        
        # denoise the noisy z at step t and t=1
        z_recon = self.denoise(z_t, t)
        z_1_recon = self.denoise(z_1, t_1)
        
        # upsample the reconstructed z at t=1
        x_1_recon = self.upsample(z_1_recon)

        # compute losses
        loss_latent = self.get_loss(noise_z, z_recon)
        loss_recon = self.get_loss(x_start, x_1_recon)
        return (loss_latent, loss_recon)