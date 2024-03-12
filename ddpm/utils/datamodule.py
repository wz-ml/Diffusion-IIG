"""
Datamodules for training and evaluation.
Each datamodule is a class that contains the following:
- data_path: path to the dataset
- batch_size: batch size
- train: whether the datamodule is for training or evaluation
- shuffle: whether to shuffle the dataset

Benchmarks:
    - MNIST
    - CIFAR10
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch as t
import numpy as np
from PIL import Image
import random

class Datamodule():
    """Generic class for datamodule. Don't use!"""
    def __init__(self, data_path, batch_size, train=True, shuffle=True):
        self.data_path = os.path.join("..", data_path)
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle
        self.transform = None

class MNISTDatamodule(Datamodule):
    def __init__(self, data_path, batch_size, train=True, shuffle=True, transform = None):
        super().__init__(data_path, batch_size, train, shuffle)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # Repeat across 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        self.dataset = datasets.MNIST(self.data_path, train=self.train, transform=self.transform, download=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

class CIFAR10Datamodule(Datamodule):
    def __init__(self, data_path, batch_size, train=True, shuffle=True):
        super().__init__(data_path, batch_size, train, shuffle)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(self.data_path, train=self.train, transform=self.transform, download=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)