import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

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
    # this is different from DownSample. I take DownSample from annotated diffusion
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.maxpool_conv2d = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2dUnweighted(dim_in, dim_out)
        )

    def forward(self, x):
        return self.maxpool_conv2d(x)
    
class UpScale(nn.Module):
    # can also do bilinear interpolation but I opted for conv2 transpose
    # this is different from UpSample. I take UpSample from annotated diffusion
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

class UpSample(nn.Module):
    # This is taken from annotated diffusion
    def __init__(self, dim_in, dim_out):
        super(UpSample, self).__init__()
        self.upsample = nn.Sequential(
            nn.UpSample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding=1),
        )

    def forward(self, x):
        return self.upsample(x)

class DownSample(nn.Module):
    # This is taken from annotated diffusion
    def __init__(self, dim_in, dim_out=None):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim_in * 4, default(dim_out, dim_in), 1),
        )


"""
The following describes an implementation of a standard (unweighted) WideResNet
block. Annotated diffusion uses a 'weight standardized' conv2d. We should try both.
"""

# I define this below purely to differentiate and so code is more readable.
def exists(x):
    return x is not None


class UnweightedConv2d(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class WideUnweightedBlock(nn.Module):
    # as opposed to wide WEIGHTED block
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = UnweightedConv2d(dim_in, dim_out, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x* (scale + 1) + shift

        x = self.act(x)
        return x
        
class WideUnweightedResNetBlock(nn.Module):

    def __init__(self, dim_in, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(time_emb_dim, dim_out * 2)
            )
            if exists(time_emb_dim) else None
        )
