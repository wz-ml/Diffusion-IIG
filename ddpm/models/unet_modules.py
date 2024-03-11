import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from functools import partial
import types
import math

"""
This file defines different building blocks for the
U-net architecture that can be used, with the goal of
being as modular as possible (hotswap in blocks in
and out). 

I use snake_case convention for variable, function and method names
I use PascalCase for class names
"""

#Network helpers are defined below
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

"""
More standard Unet blocks are defined below
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
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding=1),
        )

    def forward(self, x):
        return self.upsample(x)

class DownSample(nn.Module):
    # This is taken from annotated diffusion
    def __init__(self, dim_in, dim_out=None):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            # doing some debugging
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim_in * 4, default(dim_out, dim_in), 1),
        )

    def forward(self, x): 
        print('intermediate x shape', x.shape) # DEBUG
        return self.downsample(x)


"""
The following describes an implementation of a standard (unweighted) WideResNet
block. Annotated diffusion uses a 'weight standardized' conv2d. We should try both.
"""

# I define this below purely to differentiate and so code is more readable.
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

        self.block1 = WideUnweightedBlock(dim_in, dim_out)
        self.block2 = WideUnweightedBlock(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

"""
The following is an implementation largely taken from annotated diffusion that uses
a weight standardized convolutional layer which apparently works better with
group normalization
https://arxiv.org/abs/1903.10520
FORWARD PASS HAS NOT BEEN TESTED YET
"""

class WeightStandardizedConv2d(nn.Conv2d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class WideWeightedBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        else:
            None

        x = self.act(x)
        return x

class WideWeightedResNetBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(),
                          nn.Linear(time_emb_dim, dim_out * 2)
            )
            if exists(time_emb_dim) else None
        )
        
        self.block1 = WideWeightedBlock(dim_in, dim_out, groups=groups)
        self.block2 = WideWeightedBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            print("Time emb shape", time_emb.shape) # DEBUG
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

"""
Different position embeddings. We should try a couple of them probably.
Need to test if works as attended. !!!
"""

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        print("time shape", time.shape)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, max_time_steps, dim):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.embeddings = nn.Embedding(max_time_steps, dim)

    def forward(self, time):
        return self.embeddings(time)

class FourierPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_time_steps):
        super().__init__()
        self.dim = dim
        self.max_time_steps = max_time_steps
        self.omega = 2 * math.pi / self.max_time_steps

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.zeros((time.shape[0], self.dim), device=device)
        for i in range(half_dim):
            freq = self.omega * (2 ** i)
            embeddings[:, 2 * i] = torch.sin(freq * time)
            embeddings[:, 2 * i + 1] = torch.cos(freq * time)

        return embeddings

class RelativePositionEmbeddings(nn.Module):
    def __init__(self, max_rel_pos, dim):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.embeddings = nn.Embedding(2 * max_rel_pos + 1, dim)

    def forward(self, time):
        device = time.device
        batch_size, seq_len = time.shape
        rel_pos = torch.arange(-self.max_rel_pos,
                               self.max_rel_pos + 1,
                               device=device).repeat(seq_len, 1)

        rel_pos = rel_pos.transpose(0, 1)
        return self.embeddings(rel_pos + self.max_rel_pos)

"""
Attention variants
- attention
- linear attention
- crossAttention
- attention and linear attention are from annotated diffusion
NOTE: Have not tested thoroughly yet
"""

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim_x, dim_y, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkx = nn.Conv2d(dim_x, hidden_dim * 2, 1, bias=False)
        self.to_qky = nn.Conv2d(dim_y, hidden_dim * 2, 1, bias=False)
        self.to_v = nn.Conv2d(dim_y, hidden_dim, 1, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim_x, 1)

    def forward(self, x, y):
        bx, cx, hw, wx = x.shape
        by, cy, hy, wy = y.shape

        qkx = self.to_qkx(x).chunk(2, dim=1)
        qky, vy = self.to_qky(y), self.to_v(y)

        qkx = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkx)
        qky, vk = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), (qky, vk))

        qkx = qkx[0] * self.scale
        qky = qky * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", qkx, qky)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, vk)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=hx, y=wx)

        return self.to_out(out)
