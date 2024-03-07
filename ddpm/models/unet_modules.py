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

class Conv2dUnweighted(nn.Module):
    # note that this is DIFFERENT from nn.conv2d
    # we do (conv -> BN -> relu) twice
    # the actual paper uses some weighted conv

    def __init__(self, dim_in, dim_out, dim_mid=None):
        super().__init__()
        if not dim_mid:
            dim_mid = dim_out
        self.Conv2dUnweighted = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=3, padding=1, bias=False),
            # need to finish the rest
        )

    
class UpScale(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()

    def forward(self, x):
        # need to complete
        return None
