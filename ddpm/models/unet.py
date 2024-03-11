from models.unet_modules import *
import torch
import torch.nn as nn
from functools import partial

"""
This is a base UNet, fairly faithful to the original implementation
from Ronnneberger (2015)
"""
class BaseUNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BaseUNet, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.start = (DoubleConv2dUnweighted(dim_in, 64))
        self.down_scale1 = (DownScale(64, 128))
        self.down_scale2 = (DownScale(128, 256))
        self.down_scale3 = (DownScale(256, 512))
        self.down_scale4 = (DownScale(512, 1024))
        self.up_scale1 = (UpScale(1024, 512))
        self.up_scale2 = (UpScale(512, 256))
        self.up_scale3 = (UpScale(256, 128))
        self.up_scale4 = (UpScale(128, 64))
        self.final = (nn.Conv2d(64, dim_out, kernel_size=1))

    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down_scale1(x1)
        x3 = self.down_scale2(x2)
        x4 = self.down_scale3(x3)
        x5 = self.down_scale4(x4)
        
        x = self.up_scale1(x5, x4)
        x = self.up_scale2(x, x3)
        x = self.up_scale3(x, x2)
        x = self.up_scale4(x, x1)
        
        outputs = self.final(x)
        
        return outputs

"""
This is closer to the UNet that Karras et. al propose and use in
https://arxiv.org/abs/2312.02696
right now its base from annotated diffusion. But the idea is we can drop
different blocks in and out from what I've implemented in unet_modules
"""
class KarrasUNet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(WideWeightedResNetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        print("Dim shape:", dim) # DEBUG

        # drop in different position embeddings here!
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))), # drop in attention here
                        DownSample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        UpSample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        print("output shape", x.shape) # DEBUG

        return self.final_conv(x)
    

        

        
