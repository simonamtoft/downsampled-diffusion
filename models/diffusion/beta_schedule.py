import torch
import numpy as np


def make_beta_schedule(schedule:str, n_timestep:int, linear_start:float=1e-4, linear_end:float=2e-2, cosine_s:float=8e-3):
    """
    Different beta schedule implementations for the linear and cosine schedules.
    
    References:
    https://github.com/openai/improved-diffusion/
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    if schedule == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / n_timestep
        beta_start = scale * linear_start
        beta_end = scale * linear_end
        return np.linspace(
            beta_start, beta_end, n_timestep, dtype=np.float64
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()
