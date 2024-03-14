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
    
def DDPM_denoising_process(shape, scheduler_dict, model, forward_steps = 1000, device = None,
                           noise = None):
    if noise is None: im = torch.randn(shape, device = device) # Noise sampled from N(0, I)
    else: im = noise
    ims = [im]
    for step in tqdm(list(reversed(range(0, forward_steps)))):
        im = DDPM_denoising_step(im, torch.ones(shape[0], device = device, dtype=torch.long) * step,
                                 scheduler_dict, model)
        ims.append(im)
    return torch.stack(ims)

def DDIM_denoising_step(x_tau, tau, scheduler_dict, model, sigmas):
    """
    DDIM denoising process, as per the DDIM paper. Note that t is replaced by tau, since our reverse diffusion process
    no longer takes exactly the same number of steps as our forward diffusion process.
    """
    with torch.no_grad():
        batch_size, channels, height, width = x_tau.shape
        epsilon_theta = model(x_tau, tau)
        numerator = x_tau - scheduler_dict["sqrt_one_minus_alphas_cumprod"][tau] * epsilon_theta
        first_term = scheduler_dict["alphas_cumprod_prev"][tau].sqrt() * numerator / scheduler_dict["sqrt_alphas_cumprod"][tau]
        second_term = torch.sqrt(1 - scheduler_dict["alphas_cumprod_prev"][tau] - sigmas[tau]**2) * epsilon_theta

        z = torch.randn(x_tau.shape, device = epsilon_theta.device) * expand_dims(tau > 0) # binary mask  

        third_term = sigmas[tau] * z
        # Shape: (B, 1, 1, 1)
        x_t_minus_1 = first_term + second_term + third_term
        assert tau.shape == (batch_size,)
        return x_t_minus_1

def DDIM_denoising_process(shape, scheduler_dict, model, eta = 0, backward_steps = 100, forward_steps = 1000, device = None,
                           noise = None):
    """
    Important args:
        - shape: shape of the image. (batch, channel, height, width)
        - scheduler_dict: See forward_diffuse.
        - eta: DDIM sampling noise schedule. An eta of 0 makes the forward process deterministic.
        - noise: For interpolation. If not None, noise is randomly sampled. Shape: (batch, channel, height, width)
    """
    assert forward_steps % backward_steps == 0, "forward_steps must be divisible by backward_steps (interpolation not yet implemented)"
    steps_forward = forward_steps // backward_steps
    if noise is None: im = torch.randn(shape, device = device) # Noise sampled from N(0, I)
    else: im = noise

    # Add alphas_cumprod_prev to scheduler_dict
    temp_scheduler_dict = scheduler_dict.copy()
    alphas_cumprod_prev = torch.cat([torch.tensor([1. for i in range(steps_forward)], device = device, dtype = torch.float32), 
                                     scheduler_dict["alphas_cumprod"][:-steps_forward].squeeze()], dim = 0)
    alphas_cumprod_prev = expand_dims(alphas_cumprod_prev)
    temp_scheduler_dict["alphas_cumprod_prev"] = alphas_cumprod_prev

    # Generate sigma schedule
    first_elem = torch.sqrt((1 - scheduler_dict["alphas_cumprod_prev"]) / (1 - scheduler_dict["alphas_cumprod"]))
    second_elem = torch.sqrt(1 - scheduler_dict["alphas_cumprod"] / scheduler_dict["alphas_cumprod_prev"])
    sigmas = eta * first_elem * second_elem
    assert sigmas.shape == (forward_steps, 1, 1, 1)

    ims = [im]
    for step in tqdm(list(reversed(range(0, forward_steps, steps_forward)))):
        im = DDIM_denoising_step(im, torch.ones(shape[0], device = device, dtype=torch.long) * step,
                                 temp_scheduler_dict, model, sigmas = sigmas)
        ims.append(im)
    return torch.stack(ims)
