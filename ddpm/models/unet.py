from unet_modules import *


class UNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(UNet, self).__init__()
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
