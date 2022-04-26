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
from torch import tensor
from functools import partial

from utils import flat_bits, reduce_mean, reduce_sum
from models.utils import discretized_gaussian_log_likelihood, \
    l1_loss, l2_loss, normal_kl, extract, noise_like
from .beta_schedule import make_beta_schedule

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
        self.sample_shape = [self.in_channels, self.image_size, self.image_size]

        # determine whether to clip denoised to range or not
        self.clip_denoised = True
        self.clip_range = (-1., 1.)

        # define which objective function to use
        self.L = config['loss_type']
        self.lambda_ = 0.0001
        assert self.L in OBJETIVE_NAMES
        
        # compute loss as L2 (MSE), and flatten with mean or sum
        self.get_loss = partial(l2_loss, reduction='none')  # don't take mean
        if config['loss_flat'] == 'mean':
            self.flatten_loss = reduce_mean
        elif config['loss_flat'] == 'sum':
            self.flatten_loss = reduce_sum
        else:
            raise ValueError(f'Can only do mean or sum for flatten of loss, but {config["loss_flat"]} was desired..')
        
        # Initialize betas (variances)
        betas = make_beta_schedule(config['beta_schedule'], self.timesteps)
        assert (betas > 0).all() and (betas <= 1).all(), 'betas must be in (0, 1]'

        # Compute alphas from betas
        alphas = 1. - betas                                         # alpha_t
        alphas_cumprod = np.cumprod(alphas, axis=0)                 # alpha_bar_t
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])    # alpha_bar_{t-1}

        # compute variances and mean coefficients for the posterior q(x_{t-1} | x_t, x)
        # equation 6 and 7 in the DDPM paper
        posterior_variance = (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) * betas
        coef_x0 = np.sqrt(alphas_cumprod_prev) * betas / (1. - alphas_cumprod)
        coef_xt = np.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # clip the log variances since the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_var_clip = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

        # ensure variables are pytorch tensors with dtype float32
        to_torch = partial(tensor, dtype=torch.float32)
        
        # register buffers for alphas and betas
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # register buffer calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # register buffer calculations for posterior q(x_{t-1} | x_t, x)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(posterior_log_var_clip))
        self.register_buffer('posterior_mean_coef1', to_torch(coef_x0))
        self.register_buffer('posterior_mean_coef2', to_torch(coef_xt))
        
        # define weights to compute L_vlb from L_simple
        vlb_weights = (
            self.betas ** 2 / (
                2 * self.posterior_variance 
                * to_torch(alphas) 
                * (1 - self.alphas_cumprod)
            )
        )
        vlb_weights[0] = vlb_weights[1]
        self.register_buffer('vlb_weights', vlb_weights, persistent=False)
        assert not torch.isnan(self.vlb_weights).all()

    def q_mean_variance(self, x:tensor, t:tensor):
        """
        Get the distribution q(x_t | x).
        Equation 4 in the DDPM paper.
        
        Args:
            x (tensor):   The noiseless input (N x C x H x W).
            t (tensor):   Number of diffusion steps (t=0 is the first step).
            
        Returns:
            A tuple (mean, variance, log_variance) consisting of the mean,
            variance and log of the variance of the posterior distribution q(x_t | x). 
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x.shape) * x
        variance = extract(1. - self.alphas_cumprod, t, x.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def reconstruct(self, x:tensor, n:int) -> tensor:
        """
        Reconstructs x_hat from the noiseless input x, 
        for increasing time-scales 0 to T linearly spaced.
        """
        assert x.shape[0] >= n
        x = x[:n]
        
        # define linear timescales from 0 to T for n steps
        t = torch.linspace(0, self.timesteps - 1, n, device=self.device, dtype=torch.long)

        # generate Gaussian noise: eps ~ N(0, 1)
        eps = torch.randn_like(x)

        # sample noisy x from q distribution for t=0
        x_0 = self.q_sample(x, t, eps)

        # return reconstruction
        eps_hat = self.latent_model(x_0, t)
        x_recon = self.predict_x_from_eps(x_0, t, eps_hat, clip=False)  # true before
        return x_recon

    def predict_x_from_eps(self, x_t:tensor, t:tensor, eps:tensor, clip:bool=True):
        """Predict original data from noise (eps) and noisy data x_t for step t."""
        assert x_t.shape == eps.shape
        x = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        if clip:
            x.clamp_(*self.clip_range)
        return x

    def q_posterior(self, x:tensor, x_t:tensor, t:tensor):
        """
        Compute the mean and variance of the diffusion postertior:
            q(x_{t-1} | x_t, x)

        Args:
            x (tensor):   The noiseless x
            x_t (tensor): The x at diffusion step t
            t (tensor):   The diffusion step t, where t=0 is the first step.

        Returns:
            A tuple (mean, variance, log_variance), that is the mean, variance 
            and log of the variance of the posterior distribution q(x_{t-1} | x_t, x).
        """
        assert x.shape == x_t.shape

        # compute mean of the posterior q
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        # Compute variance and log variance of the posterior
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, x_t:tensor, t:tensor):
        """
        Apply the model to get p(x_{t-1} | x_t).
        
        Args:
            x_t (tensor):  The (N x C x H x W) tensor at time t.
            t (tensor):  A one-dimensional tensor of timesteps.
        
        Returns:
            A tuple of (mean, variance, log_variance) of the distribution p(x_{t-1} | x_t).
        """
        eps_hat = self.latent_model(x_t, t)
        x_recon = self.predict_x_from_eps(x_t, t, eps_hat, clip=True)
        mean, variance, log_variance = self.q_posterior(x_recon, x_t, t)
        return mean, variance, log_variance

    @torch.no_grad()
    def p_sample(self, x_t:tensor, t:tensor, repeat_noise:bool=False):
        """
        Sample x_{t-1} from the model from the given timestep t.

        Args:
            x_t (tensor): The current tensor at timestep t.
            t (tensor):   The step value

        Returns:
            tensor: A random sample from the model.
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
    def p_sample_loop(self, shape:tuple, every:int=1):
        """
        Generate samples from the model.
        
        Args:
            shape (tuple):  The shape of the samples (N x C x H x W)
            
        Returns:
            tensor:   A non-differentiable batch of samples.
        """
        
        # start with an image of completely random noise
        img = torch.randn(shape, device=self.device)

        # go through the ddpm in reverse order (from t=T to t=0)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def sample(self, batch_size:int=16, every:int=1):
        """Sample a batch of images from model."""
        return self.p_sample_loop((batch_size, *self.sample_shape), every)

    def q_sample(self, x:tensor, t:tensor, eps:tensor):
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
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * eps
        )

    def loss_ddpm(self, eps:tensor, eps_hat:tensor, t:tensor) -> tensor:
        """Compute the loss for the DDPM, either as simple, vlb or hybrid loss."""
        # Compute difference between noise and model output
        # and reduce to a single value per batch element
        loss = self.flatten_loss(self.get_loss(eps, eps_hat))

        # compute objective function
        if self.L == 'simple':
            obj = loss.mean()
        elif self.L == 'vlb':
            obj = (self.vlb_weights[t] * loss).mean()
        elif self.L == 'hybrid':
            obj = (loss + self.lambda_ * self.vlb_weights[t] * loss).mean()
        return obj
    
    def losses(self, x:tensor, t:tensor):
        """
        Compute the objective for a single training/validation step. 
        Either returns L_simple, L_vlb or L_hybrid, according to self.L
        To get results in bits/dim, divide by log(2).

        Args:
            x (tensor):   The input data of shape (N x C x H x W).
            t (tensor):   A batch of t's of size N.

        Returns:
            A tuple (L_simple, L_vlb), where L_simple is the L2 loss, and
            L_vlb is the variational lower bound for a single timestep t for
            each element in the batch.
        """
        
        # Generate noise
        eps = torch.randn_like(x)

        # sample noisy x from q distribution for step t
        x_t = self.q_sample(x, t, eps)

        # get the model output for diffusion step t
        eps_hat = self.latent_model(x_t, t)

        return self.loss_ddpm(eps, eps_hat, t)
    
    def vlb_terms(self, x:tensor, x_t:tensor, t:tensor):
        """
        Get a term for the variational lower-bound except for t=T.     
            L_t = KL( q(x{t-1} | xt, x) || p(x{t-1} | xt) )
            L_0 = -log p(x | x1)
        Resulting units are nats for binary data and bits/dim for color data.

        Args:
            x (tensor):   The input data of shape (N x C x H x W).
            x_t (tensor): The noisy version of the input after 
                                t number of diffusion steps.
            t (tensor):   A single timestep index (same for entire batch).
        
        Returns:
            A shape (N) tensor of negative log-likelihoods.
            
        Reference:
        Original TensorFlow implementation is done by Jonathan Ho.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L257
        """

        # compute true and predicted means and log variances
        true_mean, _, true_log_var = self.q_posterior(x, x_t, t)
        pred_mean, _, pred_log_var = self.p_mean_variance(x_t, t)

        # detach means if loss is hybrid
        # such that vlb part only optimizes variances
        if self.L == 'hybrid':
            true_mean = true_mean.detach()
            pred_mean = pred_mean.detach()

        # compute kl in bits/dim
        # KL( q(x{t-1} | xt, x0) || p(x{t-1} | xt) )
        kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl = flat_bits(kl)
        
        # compute negative log-likelihood
        # L_0 = -log p(x_0 | x_1)
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
    def calc_prior(self, x:tensor):
        """
        Calculate the prior KL term L_T for the VLB.
        
        Args:
            x (tensor): The (N x C x H x W) input tensor.
        
        Returns:
            A batch of (N) KL values for L_T, one for each batch element.
        
        Reference:
        Original TensorFlow implementation is done by Jonathan Ho.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L306
        """
        # define t=T for each x in batch
        t = torch.full((x.shape[0],), self.timesteps-1, device=self.device, dtype=torch.long)
        
        # compute mean and log variance of q distribution
        mean, _, log_var = self.q_mean_variance(x, t)
        
        # compute prior KL
        L_T = normal_kl(mean, log_var, 0., 0.)
        return flat_bits(L_T)
    
    @torch.no_grad()
    def calc_vlb(self, x:tensor):
        """
        Computes the entire variational lower-bound for the 
        entire Markov chain, measured in bits/dim for color images 
        and nats for binary images.
        
        Args:
            x (tensor): The noiseless (N x C x H x W) input tensor.
            
        Returns:
            The total VLB per batch element.
        """
        
        # compute terms L_0, ..., L_{T-1}
        vlb_t = []
        for t in list(range(self.timesteps))[::-1]:
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            eps = torch.randn_like(x)
            x_t = self.q_sample(x, t_batch, eps)
            
            # calculate vlb for timestep t
            vlb_ = self.vlb_terms(x, x_t, t_batch)  # try scalar t instead of t_batch
            vlb_t.append(vlb_)

        # vlb for each timestep for each batch
        vlb_t = torch.stack(vlb_t, dim=1)
        
        # compute the prior (L_T) for each batch
        prior = self.calc_prior(x)
        
        # sum vlb and prior
        vlb = vlb_t.sum(dim=1) + prior
        
        return {
            'vlb_t': vlb_t,
            'prior': prior,
            'vlb': vlb
        }

    def t_sample(self, n:int) -> tensor:
        """Sample n t's uniformly between [0, T]"""
        return torch.randint(0, self.timesteps, (n,), device=self.device).long()
    
    def forward(self, x:tensor):
        # select a random timestep t for each x in batch
        t = self.t_sample(x.shape[0])

        # compute and return the training objective 
        return self.losses(x, t)
