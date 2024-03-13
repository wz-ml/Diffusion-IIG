import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

def timestep_to_tensor(t, device = None):
    # Convert integer timestep to a tensor.
    return torch.ones(1, device = device, dtype = torch.int) * t

def expand_dims(x):
    """
    If x is shape (batch_size), expand it to (batch_size, 1, 1, 1)
    """
    assert len(x.shape) == 1, f"Expected shape (batch_size), got {x.shape}"
    return x[:, None, None, None]

def show_tensor(x, normalization_mean = (0.5, 0.5, 0.5), normalization_std = (0.5, 0.5, 0.5)):
    inverse_transforms = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        # Undo normalization
        transforms.Lambda(lambda x: x * torch.tensor(normalization_std, device = x.device) + torch.tensor(normalization_mean, device = x.device)),
        transforms.Lambda(lambda x: x * 255.0), # Convert to 0-255
        transforms.Lambda(lambda x: torch.clamp(x, 0, 255)), # Clamp to 0-255
        transforms.Lambda(lambda x: x.cpu().numpy().astype(np.uint8)),
    ])
    return inverse_transforms(x)

def get_arrs(beta_schedule, device = None, forward_steps = 1000):
    betas = beta_schedule(forward_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim = 0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.]), alphas_cumprod[:-1]], dim = 0) # Shift one to the left
    sqrt_recip_alphas = torch.sqrt(1 / alphas)

    variances = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    if device is not None:
        betas = betas.to(device)
        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        variances = variances.to(device)
        sqrt_recip_alphas = sqrt_recip_alphas.to(device)

    # Expand dims to (forward_steps, 1, 1, 1) for broadcasting (easy mult w/ images)
    # (batch_size, 1, 1, 1) for batch of params, (batch_size, channel, height, width) for images
    betas = expand_dims(betas)
    alphas = expand_dims(alphas)
    sqrt_recip_alphas = expand_dims(sqrt_recip_alphas)
    alphas_cumprod = expand_dims(alphas_cumprod)
    sqrt_alphas_cumprod = expand_dims(sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = expand_dims(sqrt_one_minus_alphas_cumprod)
    variances = expand_dims(variances)
    
    scheduler_dict = {
        "betas": betas,
        "alphas": alphas,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "variances": variances
    }
    return scheduler_dict