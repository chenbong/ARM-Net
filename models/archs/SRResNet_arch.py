import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.arch_util import USConv2d



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True, True])
        self.conv2 = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True, True])

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class US_MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(US_MSRResNet, self).__init__()

        self.upscale = upscale

        self.conv_first = USConv2d(in_nc, nf, 3, 1, 1, bias=True, us=[False, True])
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = USConv2d(nf, nf * 4, 3, 1, 1, bias=True, us=[True, True])
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = USConv2d(nf, nf * 9, 3, 1, 1, bias=True, us=[True, True])
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = USConv2d(nf, nf * 4, 3, 1, 1, bias=True, us=[True, True])
            self.upconv2 = USConv2d(nf, nf * 4, 3, 1, 1, bias=True, us=[True, True])
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = USConv2d(nf, nf, 3, 1, 1, bias=True, us=[True, True])
        self.conv_last = USConv2d(nf, out_nc, 3, 1, 1, bias=True, us=[True, False])

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
