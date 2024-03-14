import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from inspect import isfunction
from functools import partial
from tqdm import tqdm, trange
from einops import rearrange

import sys, os
sys.path.append("../") # important for relative imports to work

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--save_path', type=str, default="../saved_models/ddpm_cifar10_quadratic_schedule_6_epochs.pth")
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

from utils.datamodule import CIFAR10Datamodule, MNISTDatamodule, AnimeDataModule
if args.dataset == "cifar":
    train_dm = CIFAR10Datamodule("data", 32, train = True)
    val_dm = CIFAR10Datamodule("data", 32, train = False)
elif args.dataset == "mnist":
    train_dm = MNISTDatamodule("data", 32, train = True)
    val_dm = MNISTDatamodule("data", 32, train = False)
elif args.dataset == "anime":
    train_dm = AnimeDataModule("data/anime_faces",batch_size=32, train = True)
    val_dm = AnimeDataModule("data/anime_faces",batch_size=32, train = False)
else:
    raise Exception("Invalid dataset")

from train.beta_schedules import *
import numpy as np
from utils.tools import get_arrs, timestep_to_tensor, show_tensor
from train.diffusion_process import forward_diffuse

FORWARD_STEPS = 1000
scheduler_dict = get_arrs(sigmoid_beta_schedule, device, forward_steps=FORWARD_STEPS)

_, HEIGHT, WIDTH, CHANNELS = next(iter(train_dm.dataloader))[0].shape
print("Typical image shape:", (HEIGHT, WIDTH, CHANNELS))

from models.unet import KarrasUNet
import numpy as np

# From scratch
model = KarrasUNet(dim = 128, dim_mults = (1, 2, 4, 4, 8))

# Load model
# model = torch.load("../saved_models/ddpm_cifar10_quadratic_schedule_6_epochs.pth")

model = model.to(device)
# Multi GPU training
# model = nn.DataParallel(model, device_ids=[2, 3])
im = next(iter(train_dm.dataloader))[0][:1].to(device)
epsilon_theta = model(im, timestep_to_tensor(0, device = device))
assert epsilon_theta.shape == im.shape

# Main Training Loop
NUM_EPOCHS = args.epochs

from train.diffusion_process import DDPM_denoising_step, DDPM_denoising_process
from train.vis_utils import visualize_DDPM_denoising

from torch import optim
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 0)

loss_ema = None
for epoch in range(NUM_EPOCHS):
    with tqdm(train_dm.dataloader) as pbar:
        for x, class_label in pbar:
            x = x.to(device); class_label = class_label.to(device)
            # TODO: Condition on class label
            t = torch.randint(0, FORWARD_STEPS, (x.shape[0],), device = device) # batch of t
            x_t, noise = forward_diffuse(x, t, scheduler_dict)
            epsilon_theta = model(x_t, t)
            # loss = F.mse_loss(epsilon_theta, noise) # TODO: Experiment with l1 loss
            # Huber loss:
            loss = F.smooth_l1_loss(epsilon_theta, noise)
            loss.backward()
            loss_ema = round(loss_ema * 0.95 + loss.item() * 0.05 if loss_ema else loss.item(), 3)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"Training loss": loss_ema})

    validation_loss = []
    with torch.no_grad():
        with tqdm(val_dm.dataloader) as pbar:
            for x, class_label in pbar:
                x = x.to(device); class_label = class_label.to(device)
                t = torch.randint(0, FORWARD_STEPS, (x.shape[0],), device = device)
                x_t, noise = forward_diffuse(x, t, scheduler_dict)
                epsilon_theta = model(x_t, t)
                loss = F.smooth_l1_loss(epsilon_theta, noise)
                validation_loss.append(loss.item())
    mean_val_loss = sum(validation_loss) / len(validation_loss)
    print(f"Epoch {epoch}. Validation loss: {mean_val_loss:.3f}")

# Save model
torch.save(model, args.save_path)