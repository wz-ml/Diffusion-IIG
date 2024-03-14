import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utils.tools import expand_dims, show_tensor
from train.diffusion_process import forward_diffuse, DDPM_denoising_step, DDPM_denoising_process
from train.diffusion_process import DDIM_denoising_step, DDIM_denoising_process

def visualize_DDPM_denoising(scheduler_dict, model, shape, forward_steps = 1000, steps_to_show = 10, rows_to_show = 1,
                             device = None, noise = None):
    fig, ax = plt.subplots(rows_to_show, steps_to_show + 1, figsize=(15, 3 * rows_to_show), squeeze = False)
    assert len(shape) == 3, "Expected shape (channels, height, width)"
    HEIGHT, WIDTH, CHANNELS = shape
    ims = DDPM_denoising_process((rows_to_show, HEIGHT, WIDTH, CHANNELS), scheduler_dict = scheduler_dict,
                                 forward_steps = forward_steps,
                                 device = device, model = model, noise = noise)
    # ims: (steps, batch_size, channels, height, width)
    show_every = forward_steps//steps_to_show

    for row_idx in range(rows_to_show):
        ax[row_idx, 0].imshow(show_tensor(ims[row_idx][0]))
        ax[row_idx, 0].set_title("t="+str(forward_steps))
        for step in range(0, forward_steps, show_every):
            ax[row_idx, step // show_every].imshow(show_tensor(ims[step][row_idx]))
            ax[row_idx, steps_to_show - (step // show_every)].set_title(f"t={step}")
        ax[row_idx, steps_to_show].imshow(show_tensor(ims[-1][row_idx]))
        ax[row_idx, steps_to_show].set_title("t=0")
    plt.show()
    return ims

#to be adjusted for skipping steps
def visualize_DDIM_denoising(scheduler_dict, model, shape, forward_steps = 1000, backward_steps = 100, steps_to_show = 10, rows_to_show = 1,
                             device = None, noise = None):
    fig, ax = plt.subplots(rows_to_show, steps_to_show + 1, figsize=(15, 3 * rows_to_show), squeeze = False)
    assert len(shape) == 3, "Expected shape (channels, height, width)"
    HEIGHT, WIDTH, CHANNELS = shape
    ims = DDIM_denoising_process((rows_to_show, HEIGHT, WIDTH, CHANNELS), scheduler_dict = scheduler_dict,
                                 forward_steps = forward_steps,
                                 backward_steps = backward_steps,
                                 device = device, model = model, noise = noise)
    # ims: (steps, batch_size, channels, height, width)
    show_every = (backward_steps)//steps_to_show

    for row_idx in range(rows_to_show):
        ax[row_idx, 0].imshow(show_tensor(ims[row_idx][0]))
        ax[row_idx, 0].set_title("t="+str(backward_steps))
        for step in range(0, int(backward_steps), show_every):
            ax[row_idx, step // show_every].imshow(show_tensor(ims[step][row_idx]))
            ax[row_idx, steps_to_show - (step // show_every)].set_title(f"t={step}")
        ax[row_idx, steps_to_show].imshow(show_tensor(ims[-1][row_idx]))
        ax[row_idx, steps_to_show].set_title("t=0")
    plt.show()
    return ims