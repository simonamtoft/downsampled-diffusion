"""
DDPM implementation, which is a mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
thanks a lot for open-sourcing :) 
"""
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from .updown_sampling import get_downsampling, \
    get_upsampling
from .losses import discretized_gaussian_log_likelihood, \
    l1_loss, l2_loss, normal_kl
from .helpers import extract, noise_like, \
    make_beta_schedule, flat_bits, \
    get_identity_like

OBJETIVE_NAMES = ['simple', 'hybrid', 'vlb']


class DDPM(nn.Module):
    def __init__(self, config:dict, latent_model:nn.Module, device:str, color_channels:int=3):
        super().__init__()
        self.in_channels = color_channels   # Number of input channels for the latent model
        self.latent_model = latent_model    # The latent model. Predicts either means or noise.
        self.device = device                # The device to run on (cuda or cpu).

        # extract fields from config 
        self.image_size = config['image_size']
        self.timesteps = config['T']

        # setup shape for sampling
        self.sample_shape = (self.in_channels, self.image_size, self.image_size)

        # determine whether to clip denoised to range or not
        self.clip_denoised = True
        self.clip_range = (-1., 1.)

        # define which objective function to use
        self.L = config['loss_type']
        self.lambda_ = 0.0001
        assert self.L in OBJETIVE_NAMES
        # if self.L == 'l1':
        #     self.get_loss = l1_loss
        # elif self.L == 'l2':
        # self.get_loss = l2_loss
        # else:
        #     raise NotImplementedError(f'Loss type {self.L} not implemented for DDPM.')

        # Initialize betas (variances)
        betas = make_beta_schedule(config['beta_schedule'], self.timesteps)
        
        # Compute alphas from betas
        alphas = 1. - betas                                         # alpha_t
        alphas_cumprod = np.cumprod(alphas, axis=0)                 # alpha_bar_t
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])    # alpha_bar_{t-1}
        
        # compute variances and mean coefficients for the posterior q(x_{t-1} | x_t, x)
        posterior_variance = (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) * betas
        coef_x0 = np.sqrt(alphas_cumprod_prev) * betas / (1. - alphas_cumprod)
        coef_xt = np.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
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

        # calculations for posterior q(x_{t-1} | x_t, x)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))
        ))
        self.register_buffer('posterior_mean_coef1', to_torch(coef_x0))
        self.register_buffer('posterior_mean_coef2', to_torch(coef_xt))

    def q_mean_variance(self, x:torch.tensor, t:torch.tensor):
        """
        Get the distribution q(x_t | x).
        
        Args:
            x (torch.tensor):   The noiseless input (N x C x H x W).
            t (torch.tensor):   Number of diffusion steps (t=0 is the first step).
            
        Returns:
            A tuple (mean, variance, log_variance) consisting of the mean,
            variance and log of the variance of the posterior distribution q(x_t | x). 
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x.shape) * x
        variance = extract(1. - self.alphas_cumprod, t, x.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def reconstruct(self, x:torch.tensor):
        """Reconstructs x_hat from the noiseless input x"""
        # set t=0 for each x
        t_0 = torch.full((x.shape[0],), 0, device=self.device, dtype=torch.long)
        
        # generate Gaussian noise: eps ~ N(0, 1)
        eps = torch.randn_like(x)

        # sample noisy x from q distribution for t=0
        x_0 = self.q_sample(x, t_0, eps)

        # return reconstruction
        eps_hat = self.latent_model(x_0, t_0)
        x_recon = self.predict_x_from_eps(x_0, t_0, eps_hat)
        return x_recon

    def predict_x_from_eps(self, x_t:torch.tensor, t:torch.tensor, eps:torch.tensor):
        """Predict the noiseless x from a noisy x_t along with the output of the latent model, eps."""
        assert x_t.shape == eps.shape
        x = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        if self.clip_denoised:
            x.clamp_(*self.clip_range)
        return x

    def q_posterior(self, x:torch.tensor, x_t:torch.tensor, t:torch.tensor):
        """
        Compute the mean and variance of the diffusion postertior:
            q(x_{t-1} | x_t, x)

        Args:
            x (torch.tensor):   The noiseless x
            x_t (torch.tensor): The x at diffusion step t
            t (torch.tensor):   The diffusion step t, where t=0 is the first step.

        Returns:
            A tuple (mean, variance, log_variance), that is the mean, variance 
            and log of the variance of the posterior distribution q(x_{t-1} | x_t, x).
        """
        assert x.shape == x_t.shape

        # compute mean of the posterior q
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        # Compute variance and log variance of the posterior
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, x_t:torch.tensor, t:torch.tensor):
        """
        Apply the model to get p(x_{t-1} | x_t).
        
        Args:
            x_t (torch.tensor):  The (N x C x H x W) tensor at time t.
            t (torch.tensor):  A one-dimensional tensor of timesteps.
        
        Returns:
            A tuple of (mean, variance, log_variance) of the distribution p(x_{t-1} | x_t).
        """
        eps_hat = self.latent_model(x_t, t)
        x_recon = self.predict_x_from_eps(x_t, t, eps_hat)
        mean, variance, log_variance = self.q_posterior(x_recon, x_t, t)
        return mean, variance, log_variance

    @torch.no_grad()
    def p_sample(self, x_t:torch.tensor, t:torch.tensor, repeat_noise:bool=False):
        """
        Sample x_{t-1} from the model from the given timestep t.

        Args:
            x_t (torch.tensor): The current tensor at timestep t.
            t (torch.tensor):   The step value

        Returns:
            torch.tensor: A random sample from the model.
        """

        # Get mean and log variance of the model
        mean, _, log_variance = self.p_mean_variance(x_t, t)
        
        # generate Gaussian noise: eps ~ N(0, 1)
        eps = noise_like(x_t.shape, self.device, repeat_noise)
        
        # no noise when t == 0
        batch_size = x_t.shape[0]
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))
        
        # compute and return sample from p
        return mean + nonzero_mask * (0.5 * log_variance).exp() * eps
    
    @torch.no_grad()
    def p_sample_loop(self, shape:tuple):
        """
        Generate samples from the model.
        
        Args:
            shape (tuple):  The shape of the samples (N x C x H x W)
            
        Returns:
            torch.tensor:   A non-differentiable batch of samples.
        """
        
        # start with an image of completely random noise
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        # go through the ddpm in reverse order (from t=T to t=0)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def sample(self, batch_size:int=16):
        """Sample a batch of images from model."""
        return self.p_sample_loop((batch_size, *self.sample_shape))

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

    def q_sample(self, x:torch.tensor, t:torch.tensor, eps:torch.tensor):
        """
        Diffuse the data x for a given number of diffusion steps t.
        This is done by x_t ~ q(x_t | x)
        
        Args:
            x:      The initial data batch (N x C x H x W).
            t:      The number of diffusion steps minus one (0 means one step).
            eps:    Random gaussian noise of same shape as x.
        
        Returns:
            A noisy version of x.
        """
        assert x.shape == eps.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x 
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * eps
        )

    def losses(self, x:torch.tensor, t:torch.tensor):
        """
        Compute the objective for a single training step. 
        Either returns L_simple, L_vlb or L_hybrid, according to self.L

        Args:
            x (torch.tensor):   The input data of shape (N x C x H x W).
            t (torch.tensor):   A batch of t's of size N.

        Returns:
            A tuple (L_simple, L_vlb), where L_simple is the L2 loss, and
            L_vlb is the variational lower bound for a single timestep t for
            each element in the batch.
        """
        
        # Generate noise
        # eps = default(eps, lambda: torch.randn_like(x))
        eps = torch.randn_like(x)

        # sample noisy x from q distribution for step t
        x_t = self.q_sample(x, t, eps)

        # get the model output for diffusion step t
        eps_hat = self.latent_model(x_t, t)
        
        # compute the objective function
        obj = l2_loss(eps, eps_hat)
        # if self.L == 'simple':
        # elif self.L == 'vlb':
        #     obj = self.vlb_terms(x, x_t, t)
        # elif self.L == 'hybrid':
        #     L_simple = l2_loss(eps, eps_hat)
        #     L_vlb = self.vlb_terms(x, x_t, t)
        #     obj = L_simple + self.lambda_ * L_vlb
        return obj
    
    def vlb_terms(self, x:torch.tensor, x_t:torch.tensor, t:torch.tensor):
        """
        Get a term for the variational lower-bound except for t=T.     
            L_t = KL( q(x{t-1} | xt, x) || p(x{t-1} | xt) )
            L_0 = -log p(x | x1)
        Resulting units are nats for binary data and bits/dim for color data.

        Args:
            x (torch.tensor):   The input data of shape (N x C x H x W).
            x_t (torch.tensor): The noisy version of the input after 
                                t number of diffusion steps.
            t (torch.tensor):   A batch of timestep indicies.
        
        Returns:
            A shape (N) tensor of negative log-likelihoods.
        """

        # compute true and predicted means and log variances
        true_mean, _, true_log_var = self.q_posterior(x, x_t, t)
        pred_mean, _, pred_log_var = self.p_mean_variance(x_t, t)

        # detach means if loss is hybrid
        # such that vlb part only optimizes variances
        if self.L == 'hybrid':
            true_mean = true_mean.detach()
            pred_mean = pred_mean.detach()
        
        # turn variances to a diagonal matrix
        # I = get_identity_like(x)
        # I_1 = get_ones_like(x)
        # true_log_var = true_log_var * I
        # pred_log_var = pred_log_var * I

        # compute kl in bits/dim
        # KL( q(x{t-1} | xt, x0) || p(x{t-1} | xt) )
        kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl = flat_bits(kl)

        # compute negative log-likelihood
        nll = -discretized_gaussian_log_likelihood(
            x, means=pred_mean, log_scales=0.5*pred_log_var
        )
        nll = flat_bits(nll)

        # Return the loss where
        #   if t == 0: vlb = L_0 (discrete NLL)
        #   else:      vlb = L_t (KLD) 
        vlb = torch.where((t == 0), nll, kl)
        return vlb

    @torch.no_grad()
    def calc_prior(self, x:torch.tensor):
        """
        Calculate the prior KL term L_T for the VLB measured in bits/dim.
        
        Args:
            x (torch.tensor): The (N x C x H x W) input tensor.
        
        Returns:
            A batch of (N) KL values for L_T, one for each batch element.
        """
        # define t=T for each x in batch
        t = torch.full((x.shape[0],), self.timesteps-1, device=self.device, dtype=torch.long)
        
        # compute mean and log variance of q distribution
        mean, _, log_var = self.q_mean_variance(x, t)
        
        # compute prior KL
        L_T = normal_kl(mean, log_var, 0., 0.)
        return flat_bits(L_T)
    
    @torch.no_grad()
    def calc_vlb(self, x:torch.tensor):
        """
        Computes the entire variational lower-bound for the 
        entire Markov chain, measured in bits/dim for color images 
        and nats for binary images.
        
        Args:
            x (torch.tensor): The noiseless (N x C x H x W) input tensor.
            
        Returns:
            The total VLB per batch element.
        """

        batch_size = x.shape[0]
        
        # compute variational lower bound
        vlb = []
        for t in list(range(self.timesteps))[::-1]:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            eps = torch.randn_like(x)
            x_t = self.q_sample(x, t_batch, eps)
            
            # calculate vlb for timestep t
            vlb_ = self.vlb_terms(x, x_t, t_batch)  # try scalar t instead of t_batch
            vlb.append(vlb_)
        vlb = torch.stack(vlb, dim=1)
        prior = self.calc_prior(x)
        total = vlb.sum(dim=1) + prior
        return total

    def forward(self, x:torch.tensor):
        # select a random timestep t for each x in batch        
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
        
        # compute and return the training objective 
        return self.losses(x, t)


class DownsampleDDPM(DDPM):
    def __init__(self, config:dict, denoise_model:nn.Module, device:str, color_channels:int=3):
        super().__init__(config, denoise_model, device, color_channels)
        
        # original image shape
        shape = (self.in_channels, self.image_size, self.image_size)
        
        # latent image shape
        self.dim_reduc = np.power(2, config['n_downsamples']).astype(int)
        sample_channels = (
            self.in_channels * self.dim_reduc 
            if 'convolutional' in config['mode'] 
            else self.in_channels
        )
        z_size = int(self.image_size/self.dim_reduc)
        self.sample_shape = (sample_channels, z_size, z_size)
        
        # Instantiate downsample network
        self.downsample = get_downsampling(config['mode'], shape, config['n_downsamples'])

        # Instantiate upsample network
        self.upsample = get_upsampling(config['mode'], shape, config['n_downsamples'])

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

    def losses(self, x:torch.tensor, t:torch.tensor) -> tuple:
        """Train loss computations for the Downsample DDPM architecture."""
        # set t=0 for each x
        t_0 = torch.full((x.shape[0],), 0, device=self.device, dtype=torch.long)
        
        # downsample the input
        z_0 = self.downsample(x)

        # Generate noise
        eps = torch.randn_like(z_0)

        # sample noisy z from q distribution for step t and t=1
        z_t = self.q_sample(z_0, t, eps)
        z_0 = self.q_sample(z_0, t_0, eps)

        # denoise the noisy z at step t
        eps_hat = self.latent_model(z_t, t)
        eps_hat_0 = self.latent_model(z_0, t_0)

        # create reconstrution from model output at step t and upsample
        # z_hat = self.predict_x_from_eps(z_t, t, eps_hat_t)
        z_hat = self.predict_x_from_eps(z_0, t_0, eps_hat_0)
        x_hat = self.upsample(z_hat)

        # compute losses
        loss_latent = self.get_loss(eps, eps_hat)
        loss_recon = self.get_loss(x, x_hat)
        return (loss_latent, loss_recon)


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
        loss_latent = self.get_loss(eps, eps_hat)
        loss_recon = self.get_loss(x, x_hat)
        return (loss_latent, loss_recon)
