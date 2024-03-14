import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from utils.tools import expand_dims

def forward_diffuse(x_0, t, scheduler_dict, forward_steps = 1000):
    """
    Perform the forward diffusion process, as per the one-step formulation in the DDPM paper.
    Args:
        - x_0: input image. Shape: (batch_size, channel, height, width)
        - t: time step. Shape: (batch_size)
        - scheduler_dict: dictionary containing the parameters for the diffusion process.
            - betas: the beta values. Shape: (forward_steps, 1, 1, 1)
            - alphas: the alpha values. Shape: (forward_steps, 1, 1, 1)
            - sqrt_recip_alphas: the square root of the reciprocal of the alpha values. Shape: (forward_steps, 1, 1, 1)
            - alphas_cumprod: the cumulative product of the alpha values. Shape: (forward_steps, 1, 1, 1)
            - sqrt_alphas_cumprod: the square root of the cumulative product of the alpha values. Shape: (forward_steps, 1, 1, 1)
            - sqrt_one_minus_alphas_cumprod: the square root of the cumulative product of the one minus alpha values. Shape: (forward_steps, 1, 1, 1)
            - variances: the variances. Shape: (forward_steps, 1, 1, 1)
        - forward_steps: the number of forward steps. Default: 1000

    Returns:
        - x_t: the image at time step t. Shape: (batch_size, channel, height, width)
        - noise: the generated noise. Shape: (batch_size, channel, height, width)
    """
    assert len(x_0.shape) == 4, "Input image must have shape (batch_size, channel, height, width)"
    batch_size, channel, height, width = x_0.shape
    assert (t < forward_steps).all() and (t >= 0).all()
    assert t.shape == (batch_size,)
    noise = torch.randn(x_0.shape, device = x_0.device)
    x_t = x_0 * scheduler_dict["sqrt_alphas_cumprod"][t] + noise * scheduler_dict["sqrt_one_minus_alphas_cumprod"][t]
    return x_t, noise

# DDPM Sampling process
def DDPM_denoising_step(x_t, t, scheduler_dict, model):
    """
    Denoising process. Note: The model predicts epsilon_theta, the additive noise between x_0 and x_t.
    x_t: (batch, channel, height, width)
    t: (batch)
    scheduler_dict: dict containing the betas, alphas, variances, etc.
    model: main model (predicts epsilon_theta)
    """
    with torch.no_grad():
        batch_size, channels, height, width = x_t.shape
        epsilon_theta = model(x_t, t)
        ratio = scheduler_dict["betas"][t] / scheduler_dict["sqrt_one_minus_alphas_cumprod"][t]

        # Shape: (B, 1, 1, 1)
        std_t = scheduler_dict["variances"][t].sqrt()
        assert std_t.shape == (batch_size, 1, 1, 1)
        # Only add noise if t != 0
        z = torch.randn(x_t.shape, device = epsilon_theta.device) * expand_dims(t > 0) # binary mask    
        x_t_minus_1 = scheduler_dict["sqrt_recip_alphas"][t] * (x_t - ratio * epsilon_theta) + std_t * z

        assert std_t.shape == (batch_size, 1, 1, 1)
        assert z.shape == x_t.shape
        assert t.shape == (batch_size,)
        return x_t_minus_1
    
def DDPM_denoising_process(shape, scheduler_dict, model, forward_steps = 1000, device = None):
    im = torch.randn(shape, device = device) # Noise sampled from N(0, I)
    ims = [im]
    for step in tqdm(list(reversed(range(0, forward_steps)))):
        im = DDPM_denoising_step(im, torch.ones(shape[0], device = device, dtype=torch.long) * step,
                                 scheduler_dict, model)
        ims.append(im)
    return torch.stack(ims)

#to be adjusted for skipping steps
def DDIM_denoising_step(x_t, t, scheduler_dict, model, sigma = 0):
    with torch.no_grad():
        # TODO: Non-deterministic sampling with nonzero sigma
        batch_size, channels, height, width = x_t.shape
        epsilon_theta = model(x_t, t)
        first_term = scheduler_dict["alphas_cumprod_prev"].sqrt() * (x_t - scheduler_dict["betas"][t] * epsilon_theta) / scheduler_dict["alphas"][t].sqrt()
        second_term = (1 - scheduler_dict["alphas_prev"]).sqrt() * epsilon_theta
        # Shape: (B, 1, 1, 1)
        x_t_minus_1 = first_term + second_term
        assert t.shape == (batch_size,)
        return x_t_minus_1

#to be adjusted for skipping steps
def DDIM_denoising_process(shape, scheduler_dict, model, forward_steps = 1000,device = None):
    im = torch.randn(shape, device = device) # Noise sampled from N(0, I)
    ims = [im]
    for step in tqdm(list(reversed(range(0, forward_steps, 4)))):
        im = DDIM_denoising_step(im, torch.ones(shape[0], device = device, dtype=torch.long) * step,
                                 scheduler_dict, model)
        ims.append(im)
    return torch.stack(ims)
