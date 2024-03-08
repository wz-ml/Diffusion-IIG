import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file defines different building blocks for the
U-net architecture that can be used, with the goal of
being as modular as possible (hotswap in blocks in
and out). 

I use snake_case convention for variable, function and method names
I use PascalCase for class names
"""

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        # *args and **kwargs let us use this for any function that has args beyond input tensor
        skip_connection = self.fn(x, *args, **kwargs) + x
        
        return skip_connection

class DoubleConv2dUnweighted(nn.Module):
    # note that this is DIFFERENT from nn.conv2d
    # we do (conv -> BN -> relu) twice
    # the actual paper uses some weighted conv

    def __init__(self, dim_in, dim_out, dim_mid=None):
        super().__init__()
        if not dim_mid:
            dim_mid = dim_out
        self.DoubleConv2dUnweighted = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_mid, dim_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.DoubleConv2dUnweighted(x)
    
class DownScale(nn.Module):
    # maxpool -> double conv
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.maxpool_conv2d = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2dUnweighted(dim_in, dim_out)
        )

    def forward(self, x):
        return self.maxpool_conv2d(x)
    
class UpScale(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2dUnweighted(dim_in, dim_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x

class OutConv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
